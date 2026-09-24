from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from startup_adherence.application.classification import (
    classify_saved,
    current_result,
    plan_classification,
    replay,
)
from startup_adherence.application.reporting import (
    export_csv,
    fields,
    overview,
    rows_for_results,
)
from startup_adherence.domain.profile import criterion_ids, load_profile
from startup_adherence.storage import RunStore, read_json, read_jsonl

PROFILES = Path(__file__).parents[1] / "profiles"


class FakeJev:
    def __init__(self, fail_at=0):
        self.calls = []
        self.fail_at = fail_at

    def evaluate(self, *, subject, pages, profile, model):
        self.calls.append((subject, pages, profile["id"], model))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("provider unavailable")
        values = dict.fromkeys(criterion_ids(profile), 0.8)
        return values, {
            "answers": {key: {"noul": value} for key, value in values.items()},
            "provider_version": "fixture",
        }


def site(tmp_path, texts=("one", "two"), *, page_errors=None):
    directory = tmp_path / "evidence" / "company"
    directory.mkdir(parents=True)
    (directory / "evidence.jsonl").write_text(
        "".join(
            json.dumps({"url": f"https://company.test/{i}", "text": text}) + "\n"
            for i, text in enumerate(texts)
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"page_errors": page_errors or [], "crawl_limited": False}) + "\n",
        encoding="utf-8",
    )
    return directory


def classify(directory, profile, fake=None, **kwargs):
    return classify_saved(
        site_name="company",
        site_dir=directory,
        profile=profile,
        plan=plan_classification(directory / "evidence.jsonl", chunk_chars=10),
        jev_client=fake or FakeJev(),
        **kwargs,
    )


def test_same_website_two_profiles_have_independent_replay_and_results(tmp_path):
    directory = site(tmp_path)
    digital = load_profile(PROFILES / "digital_twin.json")
    gsd = load_profile(PROFILES / "gsd_patient_journey_mapping.json")
    a = classify(directory, digital)
    b = classify(directory, gsd)
    assert a["profile"]["id"] == "digital_twin"
    assert b["profile"]["id"] == "gsd_patient_journey_mapping"
    assert a["run_id"] != b["run_id"]
    assert current_result(directory, digital)["run_id"] == a["run_id"]
    assert (
        replay(site_name="company", site_dir=directory, profile=gsd)["fit_score"]
        == b["fit_score"]
    )
    first, state, _ = RunStore(directory, digital).current()
    assert state["requests_completed"] == state["requests_planned"] == 1
    saved = read_jsonl(first / "responses.jsonl")
    assert saved[0]["raw_response"]["provider_version"] == "fixture"
    assert saved[0]["question_set_sha256"] == state["question_set_sha256"]


def test_weight_change_replays_offline_without_overwriting_original(tmp_path):
    directory = site(tmp_path)
    original = load_profile(PROFILES / "digital_twin.json")
    old_result = classify(directory, original)
    amended = copy.deepcopy(original)
    amended["score_aggregation"].update(geometric_weight=0.1, bottleneck_weight=0.9)
    with pytest.raises(ValueError, match="run --mode score"):
        current_result(directory, amended)
    derived = replay(site_name="company", site_dir=directory, profile=amended)
    assert current_result(directory, amended) == derived
    assert current_result(directory, original) == old_result
    new_version = copy.deepcopy(amended)
    new_version["version"] = "1.1.0"
    revised = replay(site_name="company", site_dir=directory, profile=new_version)
    assert revised["profile"]["version"] == "1.1.0"
    assert current_result(directory, new_version) == revised
    new_questions = copy.deepcopy(original)
    new_questions["criteria"][0]["instructions"] += " extra"
    with pytest.raises(ValueError, match="questions differ"):
        replay(site_name="company", site_dir=directory, profile=new_questions)


def test_failed_new_run_preserves_previous_complete_result_and_partial_log(tmp_path):
    directory = site(tmp_path, ("A" * 10, "B" * 10))
    p = load_profile(PROFILES / "digital_twin.json")
    previous = classify(directory, p)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        classify(directory, p, FakeJev(fail_at=2))
    assert current_result(directory, p) == previous
    runs = list((RunStore(directory, p).root / "runs").iterdir())
    failed = [run for run in runs if read_json(run / "run.json")["status"] == "failed"]
    assert len(failed) == 1
    assert read_json(failed[0] / "run.json")["requests_completed"] == 1
    assert len(read_jsonl(failed[0] / "responses.jsonl")) == 1


def test_partial_crawl_is_visible_but_sitemap_failures_do_not_mark_partial(tmp_path):
    p = load_profile(PROFILES / "digital_twin.json")
    good = site(tmp_path)
    (good / "manifest.json").write_text(
        json.dumps(
            {
                "errors": [{"url": "sitemap.xml", "error": "404"}],
                "page_errors": [],
                "crawl_limited": False,
            }
        )
    )
    assert not classify(good, p)["evidence_is_partial"]
    (good / "manifest.json").write_text(
        json.dumps(
            {
                "errors": [{"url": "page", "error": "timeout"}],
                "page_errors": [{"url": "page", "error": "timeout"}],
                "crawl_limited": False,
            }
        )
    )
    assert classify(good, p)["evidence_is_partial"]


def test_generic_export_and_overview_include_profile_criteria(tmp_path):
    directory = site(tmp_path)
    p = load_profile(PROFILES / "gsd_patient_journey_mapping.json")
    result = classify(directory, p)
    rows = rows_for_results(
        [{"name": "company", "url": "https://company.test/"}], {"company": result}, p
    )
    output = tmp_path / "export.csv"
    assert export_csv(rows, p, output) == 1
    assert "core__functional_de_para_fit" in fields(p)
    assert "aux__patient_caregiver_education__flag" in fields(p)
    assert "specific_counterpart" not in output.read_text(encoding="utf-8-sig")
    assert overview(rows, p)["metrics"]["fit_score"]["mean"] == 80.0


def test_replay_rejects_corrupted_question_identity(tmp_path):
    directory = site(tmp_path)
    p = load_profile(PROFILES / "digital_twin.json")
    classify(directory, p)
    run, _, _ = RunStore(directory, p).current()
    records = read_jsonl(run / "responses.jsonl")
    records[0]["question_set_sha256"] = "wrong"
    (run / "responses.jsonl").write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="Incompatible"):
        replay(site_name="company", site_dir=directory, profile=p)


def test_planned_evidence_cannot_change_before_paid_calls(tmp_path):
    directory = site(tmp_path)
    p = load_profile(PROFILES / "digital_twin.json")
    plan = plan_classification(directory / "evidence.jsonl", chunk_chars=10)
    (directory / "evidence.jsonl").write_text("changed", encoding="utf-8")
    fake = FakeJev()
    with pytest.raises(ValueError, match="Evidence changed"):
        classify_saved(
            site_name="company",
            site_dir=directory,
            profile=p,
            plan=plan,
            jev_client=fake,
        )
    assert not fake.calls


def test_dynamic_csv_names_do_not_collide_with_fixed_columns(tmp_path):
    directory = site(tmp_path)
    p = load_profile(PROFILES / "gsd_patient_journey_mapping.json")
    p["criteria"][0]["id"] = "fit_score"
    result = classify(directory, p)
    rows = rows_for_results(
        [{"name": "company", "url": "https://company.test"}], {"company": result}, p
    )
    assert rows[0]["fit_score"] == rows[0]["core__fit_score"] == 80.0
    assert len(fields(p)) == len(set(fields(p)))


def test_translation_is_recorded_and_replay_never_calls_provider(tmp_path):
    directory = site(tmp_path, ("hello",))
    p = load_profile(PROFILES / "digital_twin.json")
    calls = []

    def translator(chunk, **kwargs):
        calls.append(kwargs["mode"])
        translated = {
            "pages": [{**page, "text": "translated"} for page in chunk["pages"]]
        }
        return (
            translated,
            [{"status": "translated", "billed_characters": 5}],
            True,
        )

    result = classify_saved(
        site_name="company",
        site_dir=directory,
        profile=p,
        plan=plan_classification(
            directory / "evidence.jsonl", chunk_chars=10, translation="deepl"
        ),
        jev_client=FakeJev(),
        translation="deepl",
        deepl_key="fixture",
        translator=translator,
    )
    run, _, _ = RunStore(directory, p).current()
    assert calls == ["deepl"]
    assert result["translation"]["billed_characters"] == 5
    assert read_jsonl(run / "translations.jsonl")[0]["status"] == "translated"
    assert (
        replay(site_name="company", site_dir=directory, profile=p)["fit_score"]
        == result["fit_score"]
    )
