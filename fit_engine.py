"""Profile-driven fit scoring shared by generic and domain-specific entrypoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

FIT_RESULT_SCHEMA_VERSION = 1
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


def _evidence_unit_key(record: dict[str, Any], index: int) -> tuple[str, ...]:
    evidence_unit = str(record.get("evidence_unit", "")).strip()
    if evidence_unit:
        return ("unit", evidence_unit)
    raw_urls = record.get("urls", [])
    if not isinstance(raw_urls, list):
        raw_urls = []
    urls = tuple(sorted({str(url).strip() for url in raw_urls if str(url).strip()}))
    if urls:
        return ("urls", *urls)
    return ("record", str(record.get("request", index)))


def normalize_records(
    records: list[dict[str, Any]], criterion_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Validate probabilities and normalize them to floats."""
    if not records:
        raise ValueError("At least one Jev record is required")
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Jev record {index} must be a JSON object")
        probabilities = record.get("probabilities")
        if not isinstance(probabilities, dict):
            raise TypeError(f"Jev record {index} has no probabilities object")
        normalized: dict[str, float] = {}
        for criterion in criterion_names:
            try:
                probability = float(probabilities[criterion])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Jev record {index} has an invalid {criterion} probability"
                ) from exc
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Jev record {index} has an out-of-range {criterion} probability"
                )
            normalized[criterion] = probability
        normalized_records.append({**record, "probabilities": normalized})
    return normalized_records


def aggregate_evidence(
    records: list[dict[str, Any]],
    *,
    criterion_names: tuple[str, ...],
    top_weights: tuple[float, ...],
    method: str = "top_weighted",
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]], int]:
    """Aggregate the strongest independent evidence units for every criterion."""
    normalized_records = normalize_records(records, criterion_names)
    units: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for index, record in enumerate(normalized_records, start=1):
        key = _evidence_unit_key(record, index)
        unit = units.setdefault(key, {})
        sources = list(key[1:]) if key[0] in {"urls", "unit"} else []
        probabilities = record["probabilities"]
        for criterion in criterion_names:
            probability = probabilities[criterion]
            current = unit.get(criterion)
            if current is None or probability > current["probability"]:
                unit[criterion] = {
                    "probability": probability,
                    "request": record.get("request", index),
                    "sources": sources,
                }

    scores: dict[str, float] = {}
    supporting_evidence: dict[str, list[dict[str, Any]]] = {}
    for criterion in criterion_names:
        all_candidates = sorted(
            (unit[criterion] for unit in units.values()),
            key=lambda candidate: candidate["probability"],
            reverse=True,
        )
        if method == "maximum":
            candidates = all_candidates[:1]
            weights = (1.0,)
        elif method == "mean":
            candidates = all_candidates
            weights = tuple(1.0 for _candidate in candidates)
        else:
            candidates = all_candidates[: len(top_weights)]
            weights = top_weights[: len(candidates)]
        weight_total = sum(weights)
        scores[criterion] = (
            sum(
                candidate["probability"] * weight
                for candidate, weight in zip(candidates, weights, strict=True)
            )
            / weight_total
        )
        supporting_evidence[criterion] = [
            {
                "score": round(100 * candidate["probability"], 2),
                "weight": round(weight / weight_total, 6),
                "request": candidate["request"],
                "sources": candidate["sources"],
            }
            for candidate, weight in zip(candidates, weights, strict=True)
        ]
    return scores, supporting_evidence, len(units)


def combine_core_scores(profile: dict[str, Any], scores: dict[str, float]) -> float:
    """Combine normalized core scores according to the profile."""
    core = [
        (
            criterion["id"],
            float(criterion.get("weight", 1.0)),
            scores[criterion["id"]],
        )
        for criterion in profile["criteria"]
        if criterion["role"] == "core"
    ]
    weight_total = sum(weight for _identifier, weight, _score in core)
    weighted_mean = sum(weight * score for _id, weight, score in core) / weight_total
    if any(score == 0 for _identifier, _weight, score in core):
        weighted_geometric = 0.0
    else:
        weighted_geometric = math.exp(
            sum(weight * math.log(score) for _id, weight, score in core) / weight_total
        )
    minimum = min(score for _identifier, _weight, score in core)
    aggregation = profile.get("score_aggregation", {})
    method = aggregation.get("method", "weighted_mean")
    if method == "weighted_mean":
        return weighted_mean
    if method == "weighted_geometric":
        return weighted_geometric
    if method == "minimum":
        return minimum
    geometric_weight = float(aggregation.get("geometric_weight", 0.5))
    bottleneck_weight = float(aggregation.get("bottleneck_weight", 0.5))
    return (geometric_weight * weighted_geometric + bottleneck_weight * minimum) / (
        geometric_weight + bottleneck_weight
    )


def build_fit_result(
    *,
    subject: str,
    records: list[dict[str, Any]],
    profile: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one portable, profile-versioned fit result."""
    validate_profile(profile)
    metadata = metadata or {}
    all_criteria = criterion_ids(profile)
    top_weights = tuple(
        float(weight)
        for weight in profile.get("evidence_aggregation", {}).get("top_weights", [1.0])
    )
    aggregate, supporting_evidence, evidence_units = aggregate_evidence(
        records,
        criterion_names=all_criteria,
        top_weights=top_weights,
        method=profile.get("evidence_aggregation", {}).get("method", "top_weighted"),
    )
    core_ids = criterion_ids(profile, "core")
    auxiliary_ids = criterion_ids(profile, "auxiliary")
    criterion_scores = {
        identifier: round(100 * aggregate[identifier], 2) for identifier in core_ids
    }
    auxiliary_scores = {
        identifier: round(100 * aggregate[identifier], 2)
        for identifier in auxiliary_ids
    }
    auxiliary_flags = {
        criterion["id"]: aggregate[criterion["id"]] >= float(criterion["threshold"])
        for criterion in profile["criteria"]
        if criterion["role"] == "auxiliary" and "threshold" in criterion
    }
    result: dict[str, Any] = {
        **metadata,
        "fit_result_schema_version": FIT_RESULT_SCHEMA_VERSION,
        "profile": {
            "id": profile["id"],
            "version": profile["version"],
            "name": profile["name"],
            "sha256": profile_sha256(profile),
        },
        "subject": subject,
        "fit_score": round(100 * combine_core_scores(profile, aggregate), 2),
        "criterion_scores": criterion_scores,
        "auxiliary_scores": auxiliary_scores,
        "auxiliary_flags": auxiliary_flags,
        "main_strength": max(core_ids, key=aggregate.__getitem__),
        "main_gap": min(core_ids, key=aggregate.__getitem__),
        "aggregation": {
            "method": profile.get("score_aggregation", {}).get(
                "method", "weighted_mean"
            ),
            "evidence_method": profile.get("evidence_aggregation", {}).get(
                "method", "top_weighted"
            ),
            "top_evidence_weights": list(top_weights),
            "evidence_units": evidence_units,
        },
        "criterion_evidence": supporting_evidence,
    }
    return result
