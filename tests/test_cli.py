from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import pytest

from startup_adherence import cli, text_cli
from startup_adherence.domain.profile import criterion_ids, load_profile
from startup_adherence.storage import RunStore, read_json, read_jsonl

PROFILES = Path(__file__).parents[1] / "profiles"


class FakeJev:
    calls: ClassVar[list[tuple[str, str, str]]] = []

    def __init__(self, key, *, timeout=60, input_style="pages"):
        assert key == "test-key"
        assert input_style in {"pages", "text"}

    def evaluate(self, *, subject, pages, profile, model):
        self.calls.append((subject, profile["id"], model))
        scores = dict.fromkeys(criterion_ids(profile), 0.6)
        return scores, {
            "answers": {key: {"noul": value} for key, value in scores.items()}
        }


def _evidence(tmp_path, names=("one",)):
    root = tmp_path / "evidence"
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "evidence.jsonl").write_text(
            json.dumps({"url": f"https://{name}.test/", "text": "evidence"}) + "\n"
        )
    sites = tmp_path / "sites.json"
    sites.write_text(
        json.dumps([{"name": name, "url": f"https://{name}.test/"} for name in names])
    )
    return root, sites


def test_paid_classification_needs_approval_even_in_smoke(
    tmp_path, monkeypatch, capsys
):
    root, sites = _evidence(tmp_path)
    FakeJev.calls = []
    monkeypatch.setattr(cli, "JevClient", FakeJev)
    monkeypatch.setattr(cli, "scrape_site", lambda **kwargs: root / kwargs["site_name"])
    monkeypatch.setattr(cli, "write_state", lambda *args: None)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    monkeypatch.setattr("sys.stdin.readline", lambda: "n\n")
    assert (
        cli.main(
            [
                "--mode",
                "smoke",
                "--sites-file",
                str(sites),
                "--evidence-root",
                str(root),
            ]
        )
        == 0
    )
    assert not FakeJev.calls
    assert "no paid API calls" in capsys.readouterr().out


def test_zero_percent_and_offline_modes_need_no_keys_or_network(tmp_path, monkeypatch):
    root, sites = _evidence(tmp_path)
    monkeypatch.delenv(cli.JEV_API_KEY_ENV, raising=False)
    monkeypatch.setattr(
        cli, "JevClient", lambda *_: (_ for _ in ()).throw(AssertionError("network"))
    )
    assert (
        cli.main(
            [
                "--mode",
                "classify",
                "--percentage",
                "0",
                "--sites-file",
                str(sites),
                "--evidence-root",
                str(root),
            ]
        )
        == 0
    )


def test_profile_specific_pipeline_cli_and_export(tmp_path, monkeypatch):
    root, sites = _evidence(tmp_path)
    FakeJev.calls = []
    monkeypatch.setattr(cli, "JevClient", FakeJev)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    p = PROFILES / "gsd_patient_journey_mapping.json"
    common = [
        "--sites-file",
        str(sites),
        "--evidence-root",
        str(root),
        "--profile",
        str(p),
    ]
    assert (
        cli.main(["--mode", "classify", "--percentage", "100", "--yes", *common]) == 0
    )
    assert FakeJev.calls == [("one", "gsd_patient_journey_mapping", "jev-latest")]
    monkeypatch.delenv(cli.JEV_API_KEY_ENV)
    monkeypatch.setattr(
        cli,
        "JevClient",
        lambda *_: (_ for _ in ()).throw(AssertionError("offline network")),
    )
    assert cli.main(["--mode", "score", *common]) == 0
    output = tmp_path / "result.csv"
    assert cli.main(["--mode", "export", "--output", str(output), *common]) == 0
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["fit_score"] == "60.0"
    assert "core__functional_de_para_fit" in rows[0]
    assert "specific_counterpart" not in rows[0]


def test_bulk_failure_does_not_cancel_other_sites(tmp_path, monkeypatch, capsys):
    root, sites = _evidence(tmp_path, ("one", "two"))

    class SometimesFake(FakeJev):
        def evaluate(self, *, subject, pages, profile, model):
            if subject == "one":
                raise RuntimeError("broken provider")
            return super().evaluate(
                subject=subject, pages=pages, profile=profile, model=model
            )

    monkeypatch.setattr(cli, "JevClient", SometimesFake)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    assert (
        cli.main(
            [
                "--mode",
                "classify",
                "--percentage",
                "100",
                "--yes",
                "--workers",
                "2",
                "--sites-file",
                str(sites),
                "--evidence-root",
                str(root),
            ]
        )
        == 1
    )
    assert "completed=1, failed=1" in capsys.readouterr().out
    assert (
        root / "two" / "profiles" / "digital_twin" / "1.0.0" / "current.json"
    ).exists()


def test_export_groups_missing_profile_results_and_does_not_write_csv(
    tmp_path, monkeypatch, capsys
):
    root, sites = _evidence(tmp_path, ("one", "two", "three"))
    monkeypatch.setattr(cli, "JevClient", FakeJev)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    common = ["--sites-file", str(sites), "--evidence-root", str(root)]
    assert (
        cli.main(
            [
                "--mode",
                "classify",
                "--site",
                "one",
                "--percentage",
                "100",
                "--yes",
                *common,
            ]
        )
        == 0
    )
    capsys.readouterr()
    destination = tmp_path / "scores.csv"
    assert cli.main(["--mode", "export", "--output", str(destination), *common]) == 1
    output = capsys.readouterr().out
    assert "selected=3, completed=1, failed=2" in output
    assert "[saved_result] 2 site(s) without a completed run" in output
    assert "two, three" in output
    assert "CSV EXPORT NOT WRITTEN" in output
    assert "Traceback" not in output
    assert not destination.exists()

    assert (
        cli.main(
            ["--mode", "export", "--traceback", "--output", str(destination), *common]
        )
        == 1
    )
    assert "Traceback" not in capsys.readouterr().out


def test_only_missing_classifies_remaining_sites_without_repeating_paid_calls(
    tmp_path, monkeypatch, capsys
):
    root, sites = _evidence(tmp_path, ("one", "two", "three"))
    FakeJev.calls = []
    monkeypatch.setattr(cli, "JevClient", FakeJev)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    common = ["--sites-file", str(sites), "--evidence-root", str(root)]
    paid = ["--mode", "classify", "--percentage", "100", "--yes"]
    assert cli.main([*paid, "--site", "one", *common]) == 0
    assert cli.main([*paid, "--only-missing", *common]) == 0
    assert [name for name, _, _ in FakeJev.calls] == ["one", "two", "three"]
    assert "already_completed=1, to_classify=2" in capsys.readouterr().out
    assert cli.main([*paid, "--only-missing", *common]) == 0
    assert "already_completed=3, to_classify=0" in capsys.readouterr().out
    assert len(FakeJev.calls) == 3

    destination = tmp_path / "scores.csv"
    assert cli.main(["--mode", "export", "--output", str(destination), *common]) == 0
    with destination.open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_only_missing_rejects_other_modes():
    with pytest.raises(SystemExit, match="2"):
        cli.parse_args(["--mode", "export", "--only-missing"])


def test_jev_failure_reports_stage_cause_and_traceback(tmp_path, monkeypatch, capsys):
    root, sites = _evidence(tmp_path)

    class BrokenJev(FakeJev):
        def evaluate(self, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(cli, "JevClient", BrokenJev)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    assert (
        cli.main(
            [
                "--mode",
                "classify",
                "--percentage",
                "100",
                "--yes",
                "--traceback",
                "--sites-file",
                str(sites),
                "--evidence-root",
                str(root),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "[one] [jev] RuntimeError: provider unavailable" in output
    assert "[one] traceback (jev):" in output
    assert "in evaluate" in output
    profile = load_profile(PROFILES / "digital_twin.json")
    runs = list((RunStore(root / "one", profile).root / "runs").iterdir())
    assert read_json(runs[0] / "run.json")["failed_stage"] == "jev"


def test_crawl_failure_reports_stage_and_traceback(tmp_path, monkeypatch, capsys):
    _, sites = _evidence(tmp_path)

    def broken_crawl(**kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(cli, "scrape_site", broken_crawl)
    assert (
        cli.main(
            [
                "--mode",
                "crawl",
                "--force-crawl",
                "--traceback",
                "--sites-file",
                str(sites),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "[one] [crawl] OSError: connection reset" in output
    assert "broken_crawl" in output


def test_duplicate_sites_are_classified_once_and_exported_once(
    tmp_path, monkeypatch, capsys
):
    root, sites = _evidence(tmp_path, ("one", "two"))
    sites.write_text(
        json.dumps(
            [
                {"name": "one", "url": "https://one.test/"},
                {"name": "one", "url": "https://other.test/"},
                {"name": "one alias", "url": "http://www.one.test/?utm_source=list"},
                {"name": "two", "url": "https://two.test/"},
                {"name": "two", "url": "https://two.test/"},
            ]
        ),
        encoding="utf-8",
    )
    FakeJev.calls = []
    monkeypatch.setattr(cli, "JevClient", FakeJev)
    monkeypatch.setenv(cli.JEV_API_KEY_ENV, "test-key")
    common = ["--sites-file", str(sites), "--evidence-root", str(root)]

    assert (
        cli.main(["--mode", "classify", "--percentage", "100", "--yes", *common]) == 0
    )
    assert sorted(call[0] for call in FakeJev.calls) == ["one", "two"]
    output = tmp_path / "scores.csv"
    assert cli.main(["--mode", "export", "--output", str(output), *common]) == 0
    with output.open(encoding="utf-8-sig", newline="") as stream:
        assert [row["site_name"] for row in csv.DictReader(stream)] == ["one", "two"]
    captured = capsys.readouterr()
    assert "Skipped 3 duplicate site entries" in captured.err
    assert "selected=2, completed=2" in captured.out


def test_duplicate_url_alias_can_be_selected_without_repeated_job():
    configured = [
        {"name": "one", "url": "https://example.test/path/"},
        {"name": "alias", "url": "http://www.example.test/path?utm_campaign=x"},
        {"name": "two", "url": "https://other.test/"},
    ]
    assert cli.select_sites(configured, ["alias", "alias"]) == configured[:1]


@pytest.mark.parametrize("no_progress", [False, True])
def test_crawl_console_shows_site_and_batch_progress_by_default(
    tmp_path, monkeypatch, capsys, no_progress
):
    root, sites = _evidence(tmp_path, ("one", "two"))
    crawl_options = []

    def scrape(**kwargs):
        crawl_options.append(kwargs)
        if kwargs["site_name"] == "two":
            raise RuntimeError("request failed")
        return root / kwargs["site_name"]

    monkeypatch.setattr(cli, "scrape_site", scrape)
    monkeypatch.setattr(cli, "write_state", lambda *args: None)
    args = ["--mode", "crawl", "--force-crawl", "--sites-file", str(sites)]
    if no_progress:
        args.append("--no-progress")
    assert cli.main(args) == 1
    output = capsys.readouterr().out
    assert ("[one] crawl START" in output) == (not no_progress)
    assert ("CRAWL PROGRESS: 2/2" in output) == (not no_progress)
    assert "selected=2, completed=1, failed=1" in output
    assert {item["verbose"] for item in crawl_options} == ({-1} if no_progress else {1})


def test_site_directory_collision_still_fails_for_distinct_sites():
    with pytest.raises(ValueError, match="collide on disk"):
        cli.select_sites(
            [
                {"name": "a b", "url": "https://one.test/"},
                {"name": "a-b", "url": "https://two.test/"},
            ],
            None,
        )


def test_text_cli_uses_shared_core_and_stores_response_before_replay(
    tmp_path, monkeypatch, capsys
):
    FakeJev.calls = []
    monkeypatch.setattr(text_cli, "JevClient", FakeJev)
    monkeypatch.setenv("TYPESAFE_PSN_DIG_TWIN_CLASS", "test-key")
    source = tmp_path / "input.md"
    source.write_text("Product offers ongoing care navigation", encoding="utf-8")
    responses = tmp_path / "raw.jsonl"
    p = PROFILES / "gsd_patient_journey_mapping.json"
    assert (
        text_cli.main(
            [
                "--profile",
                str(p),
                "--input",
                str(source),
                "--yes",
                "--runs-root",
                str(tmp_path / "runs"),
                "--responses-output",
                str(responses),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["fit_score"] == 60.0
    profile = load_profile(p)
    source_digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    input_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = tmp_path / "runs" / "input" / source_digest / input_digest
    run, _, persisted = RunStore(saved, profile).current()
    assert persisted["source"] == str(source)
    assert persisted["input_sha256"] == input_digest
    assert read_json(run / "run.json")["status"] == "completed"
    monkeypatch.delenv("TYPESAFE_PSN_DIG_TWIN_CLASS")
    assert text_cli.main(["--profile", str(p), "--responses", str(responses)]) == 0
    assert json.loads(capsys.readouterr().out)["offline_rescore"]


def test_text_chunks_are_distinct_evidence_units(tmp_path, monkeypatch, capsys):
    class VariedFake(FakeJev):
        count = 0

        def evaluate(self, *, subject, pages, profile, model):
            self.count += 1
            value = 0.2 if self.count == 1 else 0.8
            scores = dict.fromkeys(criterion_ids(profile), value)
            return scores, {"answers": {key: {"noul": value} for key in scores}}

    monkeypatch.setattr(text_cli, "JevClient", VariedFake)
    monkeypatch.setenv("TYPESAFE_PSN_DIG_TWIN_CLASS", "test-key")
    input_file = tmp_path / "input.txt"
    input_file.write_text("abcdeFGHIJ", encoding="utf-8")
    p = PROFILES / "digital_twin.json"
    root = tmp_path / "runs"
    assert (
        text_cli.main(
            [
                "--profile",
                str(p),
                "--input",
                str(input_file),
                "--chunk-chars",
                "5",
                "--yes",
                "--runs-root",
                str(root),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["aggregation"]["evidence_units"] == 2
    saved = next(root.rglob("responses.jsonl"))
    assert [row["evidence_unit"] for row in read_jsonl(saved)] == [
        "input_chunk_1",
        "input_chunk_2",
    ]


@pytest.mark.parametrize("mode", ["crawl", "score", "overview", "export"])
def test_parse_modes(mode):
    assert cli.parse_args(["--mode", mode]).mode == mode
