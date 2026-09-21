from __future__ import annotations

import json
import threading
from typing import Any

import pytest

import orchestrator


def _write_evidence(site_dir) -> None:
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
        encoding="utf-8",
    )


def _configure_test_site(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [{"name": "site1", "url": "https://example.com/"}],
    )


def test_crawl_mode_never_requires_key_or_calls_classifier(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        _write_evidence(site_dir)
        return site_dir

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("crawl mode must not call Jev")

    monkeypatch.delenv(orchestrator.JEV_API_KEY_ENV, raising=False)
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "crawl"])

    assert calls["scraper"]["max_pages"] == 0
    output = capsys.readouterr().out
    assert "no Jev or DeepL API calls will be made" in output
    assert "1 Jev request(s) would be required" in output


def test_smoke_mode_preserves_page_and_chunk_limits(tmp_path, monkeypatch) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        _write_evidence(site_dir)
        return site_dir

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "digital_twin_enabling",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "smoke"])

    assert calls["scraper"]["max_pages"] == orchestrator.SMOKE_MAX_PAGES
    assert calls["classifier"]["max_chunks"] == orchestrator.SMOKE_MAX_CHUNKS
    assert calls["classifier"]["api_key"] == "test-secret"


def test_classify_mode_prompts_for_percentage_without_scraping(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fail_scrape_site(**kwargs):
        del kwargs
        raise AssertionError("classify mode must not scrape")

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    answers = iter(["50", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fail_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify"])

    assert calls["classifier"]["evidence_dir"] == site_dir
    assert calls["classifier"]["max_chunks"] == 0
    assert calls["classifier"]["evidence_percentage"] == 50


def test_zero_percentage_cancels_without_jev_call(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("zero percent must not call Jev")

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr("builtins.input", lambda _: "0")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "classify"])

    assert "no API calls were made" in capsys.readouterr().out


def test_site_intent_selects_only_requested_sites(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        calls.append(name)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    orchestrator.main(["--mode", "crawl", "--site", "site2"])

    assert calls == ["site2"]


def test_classify_uses_safe_directory_for_display_name(tmp_path, monkeypatch) -> None:
    site_dir = tmp_path / "evidence" / "Thoth-BioSimulations"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {
                "name": "Thoth BioSimulations",
                "url": "https://www.thothbiosimulations.ca",
            }
        ],
    )
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify", "--percentage", "100", "--yes"])

    assert calls["classifier"]["site_name"] == "Thoth BioSimulations"
    assert calls["classifier"]["evidence_dir"] == site_dir


def test_bulk_rejects_evidence_directory_collisions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "Site One", "url": "https://one.example/"},
            {"name": "Site-One", "url": "https://two.example/"},
        ],
    )

    with pytest.raises(ValueError, match="same evidence directory"):
        orchestrator.main(["--mode", "crawl", "--workers", "2"])


def test_crawl_bulk_processes_sites_concurrently(tmp_path, monkeypatch) -> None:
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        calls.append(name)
        barrier.wait(timeout=2)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    orchestrator.main(["--mode", "crawl", "--sites", "all", "--workers", "2"])

    assert set(calls) == {"site1", "site2"}


def test_classify_bulk_prompts_once_and_applies_percentage_per_site(
    tmp_path, monkeypatch, capsys
) -> None:
    sites = [
        {"name": "site1", "url": "https://one.example/"},
        {"name": "site2", "url": "https://two.example/"},
    ]
    for site in sites:
        _write_evidence(tmp_path / "evidence" / site["name"])
    prompts: list[str] = []
    answers = iter(["50", "y"])
    calls: dict[str, dict[str, Any]] = {}

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    def fake_classify_site(**kwargs):
        calls[kwargs["site_name"]] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(orchestrator, "SITES", sites)
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify", "--workers", "2"])

    assert len(prompts) == 2
    assert set(calls) == {"site1", "site2"}
    assert all(call["evidence_percentage"] == 50 for call in calls.values())
    output = capsys.readouterr().out
    assert "Sites ready: 2" in output
    assert "Jev requests: 2" in output


def test_bulk_failure_does_not_cancel_other_sites(
    tmp_path, monkeypatch, capsys
) -> None:
    completed: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        if name == "site1":
            raise RuntimeError("simulated failure")
        completed.append(name)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    with pytest.raises(orchestrator.BulkExecutionError):
        orchestrator.main(["--mode", "crawl", "--workers", "2"])

    assert completed == ["site2"]
    output = capsys.readouterr().out
    assert "selected=2, completed=1, failed=1" in output
    assert "[site1] failed: RuntimeError: simulated failure" in output


def test_translation_requires_confirmation_before_api_calls(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        return site_dir

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "jev-secret")
    monkeypatch.setenv(orchestrator.DEEPL_API_KEY_ENV, "deepl-secret:fx")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "smoke", "--translation", "auto"])

    assert calls["classifier"]["translation_mode"] == "auto"
    assert calls["classifier"]["translation_target"] == "EN"
    assert calls["classifier"]["deepl_api_key"] == "deepl-secret:fx"


def test_rejected_translation_confirmation_cancels_all_api_calls(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)

    def fake_scrape_site(**kwargs):
        del kwargs
        return site_dir

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("rejected confirmation must not call APIs")

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "jev-secret")
    monkeypatch.setenv(orchestrator.DEEPL_API_KEY_ENV, "deepl-secret:fx")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "smoke", "--translation", "auto"])

    assert "no API calls were made" in capsys.readouterr().out
