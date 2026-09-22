from __future__ import annotations

from pathlib import Path

import pytest

import fit_engine


def _profile() -> dict:
    return {
        "profile_schema_version": 1,
        "id": "candidate_fit",
        "version": "1.0.0",
        "name": "Candidate fit",
        "instruction": "Judge only the supplied evidence.",
        "criteria": [
            {
                "id": "relevance",
                "role": "core",
                "weight": 2.0,
                "instructions": "Is the candidate relevant?",
            },
            {
                "id": "readiness",
                "role": "core",
                "weight": 1.0,
                "instructions": "Is the candidate ready?",
            },
            {
                "id": "explicit_claim",
                "role": "auxiliary",
                "threshold": 0.7,
                "instructions": "Does the candidate explicitly claim a fit?",
            },
        ],
        "evidence_aggregation": {"top_weights": [0.6, 0.4]},
        "score_aggregation": {"method": "weighted_mean"},
    }


def test_profile_drives_questions_weights_and_generic_result() -> None:
    profile = _profile()
    records = [
        {
            "request": 1,
            "evidence_unit": "first",
            "probabilities": {
                "relevance": 0.9,
                "readiness": 0.5,
                "explicit_claim": 0.8,
            },
        }
    ]

    fit_engine.validate_profile(profile)
    questions = fit_engine.profile_questions(profile)
    result = fit_engine.build_fit_result(
        subject="example",
        records=records,
        profile=profile,
    )

    assert set(questions) == {"relevance", "readiness", "explicit_claim"}
    assert result["fit_result_schema_version"] == 1
    assert result["profile"]["id"] == "candidate_fit"
    assert result["fit_score"] == pytest.approx(76.67)
    assert result["criterion_scores"] == {
        "relevance": 90.0,
        "readiness": 50.0,
    }
    assert result["auxiliary_flags"] == {"explicit_claim": True}
    assert result["main_strength"] == "relevance"
    assert result["main_gap"] == "readiness"


def test_evidence_aggregation_keeps_distinct_generic_chunks() -> None:
    profile = _profile()
    records = [
        {
            "request": number,
            "evidence_unit": f"chunk_{number}",
            "probabilities": {
                "relevance": probability,
                "readiness": probability,
                "explicit_claim": probability,
            },
        }
        for number, probability in enumerate((0.9, 0.5, 0.1), start=1)
    ]

    result = fit_engine.build_fit_result(
        subject="example",
        records=records,
        profile=profile,
    )

    assert result["fit_score"] == 74.0
    assert result["aggregation"]["evidence_units"] == 3
    assert [
        evidence["score"] for evidence in result["criterion_evidence"]["relevance"]
    ] == [90.0, 50.0]


def test_profile_can_average_all_evidence_units() -> None:
    profile = _profile()
    profile["evidence_aggregation"]["method"] = "mean"
    records = [
        {
            "request": number,
            "evidence_unit": f"chunk_{number}",
            "probabilities": {
                "relevance": probability,
                "readiness": probability,
                "explicit_claim": probability,
            },
        }
        for number, probability in enumerate((0.9, 0.5, 0.1), start=1)
    ]

    result = fit_engine.build_fit_result(
        subject="example",
        records=records,
        profile=profile,
    )

    assert result["fit_score"] == 50.0
    assert result["aggregation"]["evidence_method"] == "mean"
    assert len(result["criterion_evidence"]["relevance"]) == 3


def test_profile_rejects_duplicate_criteria() -> None:
    profile = _profile()
    profile["criteria"].append(dict(profile["criteria"][0]))

    with pytest.raises(ValueError, match="Duplicate criterion id"):
        fit_engine.validate_profile(profile)


def test_digital_twin_profile_is_versioned_and_valid() -> None:
    profile = fit_engine.load_profile(
        Path(__file__).parents[1] / "profiles" / "digital_twin.json"
    )

    assert profile["id"] == "digital_twin"
    assert profile["version"] == "1.0.0"
    assert fit_engine.criterion_ids(profile, "core") == (
        "specific_counterpart",
        "individualized_data_link",
        "repeated_synchronization",
        "simulation_prediction",
    )
