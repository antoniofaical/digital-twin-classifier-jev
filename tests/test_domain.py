from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import ValidationError

from startup_adherence.domain.profile import (
    criterion_ids,
    load_profile,
    profile_sha256,
    question_set_sha256,
    validate_profile,
)
from startup_adherence.domain.scoring import build_fit_result, normalize_records
from startup_adherence.evidence.chunks import evidence_chunks, select_evidence_chunks

PROFILES = Path(__file__).parents[1] / "profiles"


def profile():
    return load_profile(PROFILES / "digital_twin.json")


def test_packaged_default_profile_and_schema_match_authoring_files():
    for name in ("digital_twin.json", "profile.schema.json"):
        packaged = json.loads(
            files("startup_adherence.profiles").joinpath(name).read_text()
        )
        authored = json.loads((PROFILES / name).read_text())
        assert packaged == authored


@pytest.mark.parametrize(
    "filename",
    ["digital_twin.json", "gsd_patient_journey_mapping.json", "profile.template.json"],
)
def test_shipped_profiles_validate(filename):
    assert load_profile(PROFILES / filename)["id"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(id="BAD-ID"),
        lambda p: p.update(version="../../escape"),
        lambda p: p.update(unexpected="ignored"),
        lambda p: p["criteria"].append(
            {"id": "none", "role": "auxiliary", "instructions": ""}
        ),
    ],
)
def test_schema_rejects_invalid_profiles(change):
    p = profile()
    change(p)
    with pytest.raises((ValidationError, ValueError)):
        validate_profile(p)


def test_weights_change_profile_identity_but_not_jev_questions():
    old = profile()
    new = copy.deepcopy(old)
    new["score_aggregation"]["bottleneck_weight"] = 0.9
    assert profile_sha256(new) != profile_sha256(old)
    assert question_set_sha256(new) == question_set_sha256(old)
    new["instruction"] += " Additional policy."
    assert question_set_sha256(new) != question_set_sha256(old)


def test_score_is_generic_and_auxiliary_never_changes_core():
    p = load_profile(PROFILES / "gsd_patient_journey_mapping.json")
    core = criterion_ids(p, "core")
    aux = criterion_ids(p, "auxiliary")

    def score(auxiliary):
        return build_fit_result(
            subject="company",
            profile=p,
            records=[
                {
                    "request": 1,
                    "urls": ["https://example.test/"],
                    "probabilities": {
                        **dict.fromkeys(core, 0.7),
                        **dict.fromkeys(aux, auxiliary),
                    },
                }
            ],
        )

    assert score(0.0)["fit_score"] == score(1.0)["fit_score"] == 70.0
    assert score(0.0)["auxiliary_flags"] != score(1.0)["auxiliary_flags"]


def test_digital_twin_legacy_formula_is_preserved():
    p = profile()
    scores = {
        criterion["id"]: value
        for criterion, value in zip(p["criteria"], (0.9, 0.8, 0.4, 0.7, 0.0, 0.9))
    }
    result = build_fit_result(
        subject="A",
        profile=p,
        records=[{"request": 1, "urls": ["https://a.test"], "probabilities": scores}],
    )
    import math

    expected = 100 * (
        0.6 * math.exp(sum(math.log(x) for x in (0.9, 0.8, 0.4, 0.7)) / 4) + 0.4 * 0.4
    )
    assert result["fit_score"] == round(expected, 2)


def test_duplicate_url_keeps_strongest_unit_without_repeating_weight():
    p = profile()
    records = [
        {
            "request": i,
            "urls": ["https://a.test/"],
            "probabilities": dict.fromkeys(criterion_ids(p), value),
        }
        for i, value in enumerate((0.2, 0.9, 0.6), 1)
    ]
    result = build_fit_result(subject="A", records=records, profile=p)
    assert result["aggregation"]["evidence_units"] == 1
    assert result["fit_score"] == 90.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_probability_validation_fails_closed(invalid):
    with pytest.raises(ValueError):
        normalize_records([{"probabilities": {"x": invalid}}], ("x",))


def test_chunking_and_sampling_preserve_text_and_urls(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "".join(
            json.dumps({"url": f"https://a.test/{i}", "text": letter * 10}) + "\n"
            for i, letter in enumerate("ABCD")
        ),
        encoding="utf-8",
    )
    assert (
        "".join(
            page["text"]
            for chunk in evidence_chunks(evidence, 10)
            for page in chunk["pages"]
        )
        == "A" * 10 + "B" * 10 + "C" * 10 + "D" * 10
    )
    selected, indices, strategy, total = select_evidence_chunks(
        evidence, 10, evidence_percentage=50
    )
    assert (indices, strategy, total) == ([1, 4], "evenly_spaced_chunks", 4)
    assert [chunk["pages"][0]["text"] for chunk in selected] == ["A" * 10, "D" * 10]
    single, _, strategy, _ = select_evidence_chunks(evidence, 10, max_chunks=1)
    assert strategy == "balanced_across_pages"
    assert len({page["url"] for page in single[0]["pages"]}) == 4


def test_text_chunks_preserve_all_input_characters(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    original = "  Alpha\n  Beta  \n"
    evidence.write_text(json.dumps({"url": "input", "text": original}) + "\n")
    chunks = list(evidence_chunks(evidence, 5, preserve_whitespace=True))
    assert (
        "".join(page["text"] for chunk in chunks for page in chunk["pages"]) == original
    )
