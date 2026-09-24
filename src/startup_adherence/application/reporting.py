"""Profile-derived overview and CSV presentation."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ..domain.profile import criterion_ids, profile_sha256


def rows_for_results(
    sites: list[dict[str, str]],
    results: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for site in sites:
        if site["name"] not in results:
            continue
        result = results[site["name"]]
        if result.get("profile", {}).get("sha256") != profile_sha256(profile):
            raise ValueError(f"Score for {site['name']} uses a different profile")
        score = result["fit_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not 0 <= score <= 100
        ):
            raise ValueError(f"Invalid score for {site['name']}")
        row: dict[str, Any] = {
            "site_name": site["name"],
            "site_url": site["url"],
            "profile_id": profile["id"],
            "profile_version": profile["version"],
            "profile_sha256": profile_sha256(profile),
            "run_id": result.get("run_id", ""),
            "fit_score": score,
            "evidence_is_partial": result.get("evidence_is_partial", True),
            "evidence_chunks_available": result.get("evidence_chunks_available", ""),
            "evidence_chunks_sent": result.get("evidence_chunks_sent", ""),
            "main_strength": result["main_strength"],
            "main_gap": result["main_gap"],
        }
        for identifier in criterion_ids(profile, "core"):
            row[f"core__{identifier}"] = result["criterion_scores"][identifier]
        for identifier in criterion_ids(profile, "auxiliary"):
            row[f"aux__{identifier}"] = result["auxiliary_scores"][identifier]
            if identifier in result["auxiliary_flags"]:
                row[f"aux__{identifier}__flag"] = result["auxiliary_flags"][identifier]
        rows.append(row)
    return rows


def fields(profile: dict[str, Any]) -> list[str]:
    result = [
        "site_name",
        "site_url",
        "profile_id",
        "profile_version",
        "profile_sha256",
        "run_id",
        "fit_score",
    ]
    result.extend(f"core__{key}" for key in criterion_ids(profile, "core"))
    for criterion in profile["criteria"]:
        if criterion["role"] == "auxiliary":
            result.append(f"aux__{criterion['id']}")
            if "threshold" in criterion:
                result.append(f"aux__{criterion['id']}__flag")
    result.extend(
        [
            "evidence_is_partial",
            "evidence_chunks_available",
            "evidence_chunks_sent",
            "main_strength",
            "main_gap",
        ]
    )
    return result


def overview(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No scored sites")
    metrics = {}
    for key in [
        "fit_score",
        *(f"core__{key}" for key in criterion_ids(profile, "core")),
        *(f"aux__{key}" for key in criterion_ids(profile, "auxiliary")),
    ]:
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            raise ValueError(f"Invalid {key}")
        metrics[key] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_population": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "count": len(rows),
        "full_evidence": sum(not row["evidence_is_partial"] for row in rows),
        "partial_evidence": sum(bool(row["evidence_is_partial"]) for row in rows),
        "metrics": metrics,
        "gaps": dict(Counter(row["main_gap"] for row in rows)),
        "top_sites": sorted(
            ((row["site_name"], row["fit_score"]) for row in rows),
            key=lambda entry: (-entry[1], entry[0].casefold()),
        )[:5],
    }


def print_overview(rows: list[dict[str, Any]], profile: dict[str, Any]) -> None:
    summary = overview(rows, profile)
    print(
        f"STATISTICAL OVERVIEW ({summary['count']} site(s); {profile['id']}@{profile['version']})"
    )
    print(
        f"Evidence: full={summary['full_evidence']}, partial={summary['partial_evidence']}"
    )
    print(
        f"{'Criterion':<36} {'Mean':>7} {'Median':>7} {'Std(pop)':>9} {'Min':>7} {'Max':>7}"
    )
    for identifier, values in summary["metrics"].items():
        print(
            f"{identifier:<36} {values['mean']:>7.2f} {values['median']:>7.2f} "
            f"{values['std_population']:>9.2f} {values['min']:>7.2f} {values['max']:>7.2f}"
        )
    print("MAIN GAPS")
    for key, count in sorted(summary["gaps"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{key}: {count}")
    print("TOP 5 ADHERENCE")
    for index, (name, score) in enumerate(summary["top_sites"], 1):
        print(f"{index}. {name}: {score:.2f}/100")


def export_csv(
    rows: list[dict[str, Any]], profile: dict[str, Any], output: Path
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields(profile))
        writer.writeheader()
        writer.writerows(
            sorted(
                rows, key=lambda row: (-row["fit_score"], row["site_name"].casefold())
            )
        )
    temporary.replace(output)
    return len(rows)
