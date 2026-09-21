"""Classify saved website evidence with TypeSafe AI's Jev model."""

from __future__ import annotations

import json
from collections.abc import Iterator
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
    positive_threshold: float = 0.70,
    negative_threshold: float = 0.30,
    request_timeout: int = 60,
) -> dict[str, Any]:
    """Send every saved evidence chunk to Jev and save the final result."""
    if not api_key:
        raise ValueError("A Jev API key must be passed by orchestrator.py")
    evidence_file = evidence_dir / "evidence.jsonl"
    if not evidence_file.exists():
        raise FileNotFoundError(f"Evidence not found: {evidence_file}")

    chunks = list(evidence_chunks(evidence_file, chunk_chars))
    if not chunks:
        raise ValueError(f"No textual evidence found in {evidence_file}")

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
                        "chunk": number,
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
    result = {
        "site_name": site_name,
        "classification": classification,
        "is_digital_twin": is_digital_twin,
        "requires_human_review": requires_review,
        "criterion_probabilities": aggregate,
        "chunks_supporting_all_core_criteria": coherent_chunks,
        "evidence_chunks_sent": len(chunks),
        "model": model,
    }
    (evidence_dir / "classification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
