"""Versioned profile contract and question identity."""

from __future__ import annotations

import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SUPPORTED_PROFILE_SCHEMA_VERSION = 1
SUPPORTED_AGGREGATIONS = {
    "weighted_mean",
    "weighted_geometric",
    "weighted_geometric_bottleneck",
    "minimum",
}
SUPPORTED_EVIDENCE_AGGREGATIONS = {"top_weighted", "mean", "maximum"}


def profile_sha256(profile: dict[str, Any]) -> str:
    """Return a stable digest for the complete scoring contract."""
    canonical = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def question_set_sha256(profile: dict[str, Any]) -> str:
    """Return a digest of only the profile fields that affect Jev responses."""
    canonical = json.dumps(
        {
            "instruction": profile["instruction"],
            "questions": profile_questions(profile),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_profile(path: Path) -> dict[str, Any]:
    """Load and validate a fit profile from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Fit profile must be a JSON object: {path}")
    validate_profile(payload)
    return payload


def validate_profile(profile: dict[str, Any]) -> None:
    """Validate the stable, portable profile contract."""
    schema = json.loads(
        files("startup_adherence.profiles")
        .joinpath("profile.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(profile)
    if profile.get("profile_schema_version") != SUPPORTED_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported profile_schema_version: "
            f"{profile.get('profile_schema_version')!r}"
        )
    for field in ("id", "version", "name", "instruction"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f"Profile field {field!r} must be a non-empty string")

    criteria = profile.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Profile criteria must be a non-empty list")
    identifiers: set[str] = set()
    core_count = 0
    for index, criterion in enumerate(criteria, start=1):
        if not isinstance(criterion, dict):
            raise TypeError(f"Criterion {index} must be a JSON object")
        identifier = criterion.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Criterion {index} has no non-empty id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate criterion id: {identifier}")
        identifiers.add(identifier)
        if criterion.get("role") not in {"core", "auxiliary"}:
            raise ValueError(f"Criterion {identifier} role must be core or auxiliary")
        if criterion["role"] == "core":
            core_count += 1
            weight = criterion.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"Criterion {identifier} weight must be numeric")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError(f"Criterion {identifier} weight must be positive")
        instructions = criterion.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"Criterion {identifier} needs instructions")
        if "threshold" in criterion:
            threshold = criterion["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise TypeError(f"Criterion {identifier} threshold must be numeric")
            if not 0 <= float(threshold) <= 1:
                raise ValueError(
                    f"Criterion {identifier} threshold must be from zero to one"
                )
    if core_count == 0:
        raise ValueError("Profile must contain at least one core criterion")

    evidence = profile.get("evidence_aggregation", {})
    if not isinstance(evidence, dict):
        raise TypeError("evidence_aggregation must be a JSON object")
    evidence_method = evidence.get("method", "top_weighted")
    if evidence_method not in SUPPORTED_EVIDENCE_AGGREGATIONS:
        raise ValueError(f"Unsupported evidence aggregation method: {evidence_method}")
    top_weights = evidence.get("top_weights", [1.0])
    if not isinstance(top_weights, list) or not top_weights:
        raise ValueError("evidence_aggregation.top_weights must be a non-empty list")
    for weight in top_weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("Evidence weights must be numeric")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("Evidence weights must be positive")

    aggregation = profile.get("score_aggregation", {})
    if not isinstance(aggregation, dict):
        raise TypeError("score_aggregation must be a JSON object")
    method = aggregation.get("method", "weighted_mean")
    if method not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported score aggregation method: {method}")
    if method == "weighted_geometric_bottleneck":
        geometric_weight = aggregation.get("geometric_weight", 0.5)
        bottleneck_weight = aggregation.get("bottleneck_weight", 0.5)
        for name, weight in (
            ("geometric_weight", geometric_weight),
            ("bottleneck_weight", bottleneck_weight),
        ):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(weight)) or float(weight) < 0:
                raise ValueError(f"{name} must be non-negative")
        if float(geometric_weight) + float(bottleneck_weight) <= 0:
            raise ValueError("Score aggregation weights cannot both be zero")


def profile_questions(profile: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build the Jev question object from a validated profile."""
    return {
        criterion["id"]: {
            "type": "noul",
            "instructions": criterion["instructions"],
        }
        for criterion in profile["criteria"]
    }


def criterion_ids(profile: dict[str, Any], role: str | None = None) -> tuple[str, ...]:
    return tuple(
        criterion["id"]
        for criterion in profile["criteria"]
        if role is None or criterion["role"] == role
    )
