"""Pure evidence aggregation and scoring for any validated profile."""

from __future__ import annotations

import math
from typing import Any

from .profile import criterion_ids, profile_sha256, validate_profile

FIT_RESULT_SCHEMA_VERSION = 1


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
                value = probabilities[criterion]
                if isinstance(value, bool):
                    raise TypeError("Boolean is not a probability")
                probability = float(value)
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
