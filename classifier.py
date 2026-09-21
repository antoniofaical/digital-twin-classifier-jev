"""Classify saved website evidence with TypeSafe AI's Jev model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any

import requests

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

QUESTIONS = {
    "specific_counterpart": {
        "type": "noul",
        "instructions": (
            "Does the evidence clearly show a digital representation of one specific, "
            "identifiable physical or biological counterpart, rather than a generic model?"
        ),
    },
    "individualized_data_link": {
        "type": "noul",
        "instructions": (
            "Does measured data from that specific counterpart initialize or personalize "
            "its corresponding model?"
        ),
    },
    "repeated_synchronization": {
        "type": "noul",
        "instructions": (
            "Is the virtual representation repeatedly updated with new data from the same "
            "real-world counterpart?"
        ),
    },
    "simulation_prediction": {
        "type": "noul",
        "instructions": (
            "Does the virtual representation simulate scenarios, predict outcomes, or "
            "evaluate interventions?"
        ),
    },
    "self_claim": {
        "type": "noul",
        "instructions": (
            "Does the company explicitly call this product or technology a digital twin?"
        ),
    },
    "enabling_technology": {
        "type": "noul",
        "instructions": (
            "Even if a complete digital twin is not evidenced, is this clearly an enabling "
            "component such as virtual modeling, simulation, sensing, or data integration?"
        ),
    },
}

CORE_CRITERIA = (
    "specific_counterpart",
    "individualized_data_link",
    "repeated_synchronization",
    "simulation_prediction",
)


def evidence_chunks(evidence_file: Path, chunk_chars: int) -> Iterator[dict[str, Any]]:
    pages: list[dict[str, str]] = []
    size = 0
    with evidence_file.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            url = str(record.get("url", ""))
            for start in range(0, len(text), chunk_chars):
                part = text[start : start + chunk_chars]
                if pages and size + len(part) > chunk_chars:
                    yield {"pages": pages}
                    pages = []
                    size = 0
                pages.append({"url": url, "text": part})
                size += len(part)
    if pages:
        yield {"pages": pages}


def representative_evidence_chunk(
    evidence_file: Path, chunk_chars: int
) -> dict[str, Any]:
    """Sample all saved pages fairly within one bounded chunk."""
    records: list[dict[str, str]] = []
    with evidence_file.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if text:
                records.append({"url": str(record.get("url", "")), "text": text})

    offsets = [0] * len(records)
    samples = [""] * len(records)
    active = list(range(len(records)))
    remaining = chunk_chars

    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active: list[int] = []
        for index in active:
            if remaining == 0:
                break
            available = len(records[index]["text"]) - offsets[index]
            take = min(share, available, remaining)
            start = offsets[index]
            samples[index] += records[index]["text"][start : start + take]
            offsets[index] += take
            remaining -= take
            if offsets[index] < len(records[index]["text"]):
                next_active.append(index)
        active = next_active

    return {
        "pages": [
            {"url": record["url"], "text": sample}
            for record, sample in zip(records, samples, strict=True)
            if sample
        ]
    }


def evenly_spaced_indices(total: int, selected: int) -> list[int]:
    """Choose chunk indices spanning the start, middle, and end of the corpus."""
    if selected <= 0 or selected > total:
        raise ValueError("selected must be between one and total")
    if selected == 1:
        return [total // 2]
    return [
        round(position * (total - 1) / (selected - 1))
        for position in range(selected)
    ]


def decide(
    probabilities: dict[str, float],
    coherent_core_evidence: bool,
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[str, bool | None, bool]:
    core = [probabilities[name] for name in CORE_CRITERIA]
    if coherent_core_evidence and all(value >= positive_threshold for value in core):
        return "verified_digital_twin", True, False
    if probabilities["self_claim"] >= positive_threshold:
        return "claimed_digital_twin_unverified", None, True
    if all(value >= positive_threshold for value in core):
        return "manual_review_cross_chunk_evidence", None, True
    if probabilities["enabling_technology"] >= positive_threshold:
        if probabilities["simulation_prediction"] >= positive_threshold:
            return "virtual_model_or_biosimulation", False, True
        return "digital_twin_enabling", False, True
    if all(value <= negative_threshold for value in core):
        return "not_digital_twin", False, False
    return "manual_review", None, True


def classify_site(
    *,
    site_name: str,
    evidence_dir: Path,
    api_key: str,
    model: str = "jev-latest",
    chunk_chars: int = 20_000,
    max_chunks: int = 0,
    evidence_percentage: float | None = None,
    positive_threshold: float = 0.70,
    negative_threshold: float = 0.30,
    request_timeout: int = 60,
) -> dict[str, Any]:
    """Send selected evidence chunks to Jev and save the result."""
    if not api_key:
        raise ValueError("A Jev API key must be passed by orchestrator.py")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be a positive integer")
    if max_chunks < 0:
        raise ValueError("max_chunks must be zero or a positive integer")
    if evidence_percentage is not None and not 0 < evidence_percentage <= 100:
        raise ValueError(\n            "evidence_percentage must be greater than zero and at most 100"\n        )
    if evidence_percentage is not None and max_chunks:
        raise ValueError("Use evidence_percentage or max_chunks, not both")

    evidence_file = evidence_dir / "evidence.jsonl"
    if not evidence_file.exists():
        raise FileNotFoundError(f"Evidence not found: {evidence_file}")

    all_chunks = list(evidence_chunks(evidence_file, chunk_chars))
    if not all_chunks:
        raise ValueError(f"No textual evidence found in {evidence_file}")

    total_chunks = len(all_chunks)
    if evidence_percentage is not None:
        selected_count = min(
            total_chunks,
            ceil(total_chunks * evidence_percentage / 100),
        )
    elif max_chunks:
        selected_count = min(total_chunks, max_chunks)
    else:
        selected_count = total_chunks

    if selected_count == total_chunks:
        chunks = all_chunks
        source_chunk_numbers = list(range(1, total_chunks + 1))
        sampling_strategy = "all_chunks"
    elif selected_count == 1:
        chunks = [representative_evidence_chunk(evidence_file, chunk_chars)]
        source_chunk_numbers = list(range(1, total_chunks + 1))
        sampling_strategy = "balanced_across_pages"
    else:
        selected_indices = evenly_spaced_indices(total_chunks, selected_count)
        chunks = [all_chunks[index] for index in selected_indices]
        source_chunk_numbers = [index + 1 for index in selected_indices]
        sampling_strategy = "evenly_spaced_chunks"

    chunks_were_limited = len(chunks) < total_chunks

    crawl_was_limited = False
    manifest_file = evidence_dir / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        crawl_was_limited = bool(manifest.get("stopped_by_page_limit"))

    evidence_is_partial = chunks_were_limited or crawl_was_limited
    aggregate = {name: 0.0 for name in QUESTIONS}
    coherent_chunks: list[int] = []
    chunk_log = evidence_dir / "jev_chunks.jsonl"

    with chunk_log.open("w", encoding="utf-8") as log:
        for number, chunk in enumerate(chunks, start=1):
            response = requests.post(
                JEV_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "state": {
                        "company": site_name,
                        "instruction": (
                            "Judge only capabilities explicitly supported here."
                        ),
                        **chunk,
                    },
                    "questions": QUESTIONS,
                },
                timeout=request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            probabilities = {
                name: float(payload["answers"][name]["noul"]) for name in QUESTIONS
            }
            if all(probabilities[name] >= positive_threshold for name in CORE_CRITERIA):
                coherent_chunks.append(number)
            for name, probability in probabilities.items():
                aggregate[name] = max(aggregate[name], probability)
            log.write(
                json.dumps(
                    {
                        "request": number,
                        "source_chunk": source_chunk_numbers[number - 1]
                        if sampling_strategy == "evenly_spaced_chunks"
                        else None,
                        "sampling_strategy": sampling_strategy,
                        "urls": list(
                            dict.fromkeys(page["url"] for page in chunk["pages"])
                        ),
                        "probabilities": probabilities,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    classification, is_digital_twin, requires_review = decide(
        aggregate,
        coherent_core_evidence=bool(coherent_chunks),
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )
    provisional_classification = classification if evidence_is_partial else None
    if evidence_is_partial:
        classification = "partial_evidence_classification"
        is_digital_twin = None
        requires_review = True

    result = {
        "site_name": site_name,
        "classification": classification,
        "provisional_classification": provisional_classification,
        "is_digital_twin": is_digital_twin,
        "requires_human_review": requires_review,
        "evidence_is_partial": evidence_is_partial,
        "crawl_was_limited": crawl_was_limited,
        "chunks_were_limited": chunks_were_limited,
        "evidence_percentage_requested": evidence_percentage,
        "sampling_strategy": sampling_strategy,
        "source_chunk_numbers": source_chunk_numbers,
        "criterion_probabilities": aggregate,
        "chunks_supporting_all_core_criteria": coherent_chunks,
        "evidence_chunks_available": total_chunks,
        "evidence_chunks_sent": len(chunks),
        "model": model,
    }
    (evidence_dir / "classification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
