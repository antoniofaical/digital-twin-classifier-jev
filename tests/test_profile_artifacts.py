from __future__ import annotations

import json
from pathlib import Path

import fit_engine

PROFILES = Path(__file__).parents[1] / "profiles"


def test_profile_schema_tracks_runtime_aggregation_methods() -> None:
    schema = json.loads((PROFILES / "profile.schema.json").read_text(encoding="utf-8"))

    evidence_methods = set(
        schema["$defs"]["evidence_aggregation"]["properties"]["method"]["enum"]
    )
    simple_methods = set(
        schema["$defs"]["simple_score_aggregation"]["properties"]["method"]["enum"]
    )
    bottleneck_method = schema["$defs"]["bottleneck_score_aggregation"]["properties"][
        "method"
    ]["const"]

    assert evidence_methods == fit_engine.SUPPORTED_EVIDENCE_AGGREGATIONS
    assert simple_methods | {bottleneck_method} == fit_engine.SUPPORTED_AGGREGATIONS


def test_shipped_profiles_and_template_pass_runtime_validation() -> None:
    for filename in (
        "digital_twin.json",
        "gsd_patient_journey_mapping.json",
        "profile.template.json",
    ):
        profile = fit_engine.load_profile(PROFILES / filename)
        assert fit_engine.question_set_sha256(profile)


def test_profile_request_template_is_valid_json_with_required_context() -> None:
    request = json.loads(
        (PROFILES / "profile_request.template.json").read_text(encoding="utf-8")
    )

    assert request["request_schema_version"] == 1
    assert request["target_definition"]
    assert request["decision_use"]
    assert request["evidence_description"]
    assert request["required_dimensions"]


def test_authoring_docs_reference_validation_and_machine_contract() -> None:
    guide = (PROFILES / "HOW_TO_CREATE_PROFILES.md").read_text(encoding="utf-8")
    prompt = (PROFILES / "PROFILE_GENERATION_PROMPT.md").read_text(encoding="utf-8")

    assert "--validate-profile" in guide
    assert "question_set_sha256" in guide
    assert "profile.schema.json" in guide
    assert "JSON only" in prompt
