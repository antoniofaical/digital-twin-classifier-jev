"""Classify saved website evidence with TypeSafe AI's Jev model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from math import ceil, prod
from pathlib import Path
from typing import Any

import requests

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEEPL_FREE_ENDPOINT = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_ENDPOINT = "https://api.deepl.com/v2/translate"

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

AUXILIARY_CRITERIA = ("self_claim", "enabling_technology")
CLASSIFICATION_SCHEMA_VERSION = 2
TOP_EVIDENCE_WEIGHTS = (0.60, 0.25, 0.15)
SELF_CLAIM_THRESHOLD = 0.70

PRESERVED_RESULT_FIELDS = (
    "evidence_is_partial",
    "crawl_was_limited",
    "crawl_limit_reasons",
    "chunks_were_limited",
    "evidence_percentage_requested",
    "sampling_strategy",
    "source_chunk_numbers",
    "translation",
    "evidence_chunks_available",
    "evidence_chunks_sent",
    "model",
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
            page = {
                "url": str(record.get("url", "")),
                "language_hint": str(record.get("language_hint", "")),
            }
            for start in range(0, len(text), chunk_chars):
                part = text[start : start + chunk_chars]
                if pages and size + len(part) > chunk_chars:
                    yield {"pages": pages}
                    pages = []
                    size = 0
                pages.append({**page, "text": part})
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
                records.append(
                    {
                        "url": str(record.get("url", "")),
                        "language_hint": str(record.get("language_hint", "")),
                        "text": text,
                    }
                )

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
            {
                "url": record["url"],
                "language_hint": record["language_hint"],
                "text": sample,
            }
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
        round(position * (total - 1) / (selected - 1)) for position in range(selected)
    ]


def select_evidence_chunks(
    evidence_file: Path,
    chunk_chars: int,
    *,
    max_chunks: int = 0,
    evidence_percentage: float | None = None,
) -> tuple[list[dict[str, Any]], list[int], str, int]:
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
        return (
            all_chunks,
            list(range(1, total_chunks + 1)),
            "all_chunks",
            total_chunks,
        )
    if selected_count == 1:
        return (
            [representative_evidence_chunk(evidence_file, chunk_chars)],
            list(range(1, total_chunks + 1)),
            "balanced_across_pages",
            total_chunks,
        )

    selected_indices = evenly_spaced_indices(total_chunks, selected_count)
    return (
        [all_chunks[index] for index in selected_indices],
        [index + 1 for index in selected_indices],
        "evenly_spaced_chunks",
        total_chunks,
    )


def deepl_endpoint(api_key: str) -> str:
    return DEEPL_FREE_ENDPOINT if api_key.endswith(":fx") else DEEPL_PRO_ENDPOINT


def is_english_hint(language_hint: str) -> bool:
    normalized = language_hint.lower().replace("_", "-")
    return normalized == "en" or normalized.startswith("en-")


def translate_chunk(
    chunk: dict[str, Any],
    *,
    api_key: str,
    mode: str,
    target_language: str,
    request_timeout: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Translate selected page fragments while preserving URL provenance."""
    pages = [dict(page) for page in chunk["pages"]]
    candidate_indexes = [
        index
        for index, page in enumerate(pages)
        if mode == "deepl" or not is_english_hint(str(page.get("language_hint", "")))
    ]
    records: list[dict[str, Any]] = []

    for index, page in enumerate(pages):
        if index not in candidate_indexes:
            records.append(
                {
                    "page_index": index,
                    "url": page["url"],
                    "status": "skipped_english_hint",
                    "language_hint": page.get("language_hint", ""),
                    "source_characters": len(page["text"]),
                    "original_sha256": hashlib.sha256(
                        page["text"].encode("utf-8")
                    ).hexdigest(),
                }
            )

    if not candidate_indexes:
        return {"pages": pages}, records, False

    response = requests.post(
        deepl_endpoint(api_key),
        headers={
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "text": [pages[index]["text"] for index in candidate_indexes],
            "target_lang": target_language.upper(),
            "show_billed_characters": True,
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    translations = response.json().get("translations", [])
    if len(translations) != len(candidate_indexes):
        raise ValueError("DeepL returned an unexpected number of translations")

    for index, translation in zip(candidate_indexes, translations, strict=True):
        original_text = pages[index]["text"]
        translated_text = str(translation["text"])
        detected_language = str(translation.get("detected_source_language", ""))
        pages[index]["text"] = translated_text
        pages[index]["detected_source_language"] = detected_language
        pages[index]["translation_target_language"] = target_language.upper()
        records.append(
            {
                "page_index": index,
                "url": pages[index]["url"],
                "status": "translated",
                "language_hint": pages[index].get("language_hint", ""),
                "detected_source_language": detected_language,
                "target_language": target_language.upper(),
                "source_characters": len(original_text),
                "billed_characters": int(
                    translation.get("billed_characters", len(original_text))
                ),
                "original_sha256": hashlib.sha256(
                    original_text.encode("utf-8")
                ).hexdigest(),
                "translated_text": translated_text,
            }
        )
    records.sort(key=lambda record: record["page_index"])
    return {"pages": pages}, records, True


def load_jev_records(chunk_log: Path) -> list[dict[str, Any]]:
    """Load and validate saved Jev responses without making API calls."""
    if not chunk_log.is_file():
        raise FileNotFoundError(f"Jev chunk log not found: {chunk_log}")

    records: list[dict[str, Any]] = []
    with chunk_log.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {chunk_log} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"Jev record at line {line_number} must be a JSON object"
                )
            probabilities = record.get("probabilities")
            if not isinstance(probabilities, dict):
                raise TypeError(
                    f"Jev record at line {line_number} has no probabilities object"
                )
            normalized: dict[str, float] = {}
            for criterion in QUESTIONS:
                try:
                    probability = float(probabilities[criterion])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Jev record at line {line_number} has an invalid "
                        f"{criterion} probability"
                    ) from exc
                if not 0 <= probability <= 1:
                    raise ValueError(
                        f"Jev record at line {line_number} has an out-of-range "
                        f"{criterion} probability"
                    )
                normalized[criterion] = probability
            records.append({**record, "probabilities": normalized})

    if not records:
        raise ValueError(f"No Jev records found in {chunk_log}")
    return records


def _evidence_unit_key(record: dict[str, Any], index: int) -> tuple[str, ...]:
    raw_urls = record.get("urls", [])
    if not isinstance(raw_urls, list):
        raw_urls = []
    urls = tuple(sorted({str(url).strip() for url in raw_urls if str(url).strip()}))
    if urls:
        return ("urls", *urls)
    return ("record", str(record.get("request", index)))


def aggregate_criterion_scores(
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]], int]:
    """Aggregate the strongest independent evidence units for every criterion."""
    if not records:
        raise ValueError("At least one Jev record is required")

    units: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for index, record in enumerate(records, start=1):
        key = _evidence_unit_key(record, index)
        unit = units.setdefault(key, {})
        urls = list(key[1:]) if key[0] == "urls" else []
        probabilities = record.get("probabilities", {})
        for criterion in QUESTIONS:
            probability = float(probabilities[criterion])
            if not 0 <= probability <= 1:
                raise ValueError(f"{criterion} probability must be from zero to one")
            current = unit.get(criterion)
            if current is None or probability > current["probability"]:
                unit[criterion] = {
                    "probability": probability,
                    "request": record.get("request", index),
                    "urls": urls,
                }

    scores: dict[str, float] = {}
    supporting_evidence: dict[str, list[dict[str, Any]]] = {}
    for criterion in QUESTIONS:
        candidates = sorted(
            (unit[criterion] for unit in units.values()),
            key=lambda candidate: candidate["probability"],
            reverse=True,
        )[: len(TOP_EVIDENCE_WEIGHTS)]
        weights = TOP_EVIDENCE_WEIGHTS[: len(candidates)]
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
                "urls": candidate["urls"],
            }
            for candidate, weight in zip(candidates, weights, strict=True)
        ]
    return scores, supporting_evidence, len(units)


def adherence_score(core_scores: dict[str, float]) -> float:
    """Combine four required criteria, penalizing the weakest link."""
    values = [core_scores[name] for name in CORE_CRITERIA]
    geometric_mean = prod(values) ** (1 / len(values))
    return 0.60 * geometric_mean + 0.40 * min(values)


def build_adherence_result(
    *,
    site_name: str,
    records: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the versioned continuous-score result shared by live and replay runs."""
    metadata = metadata or {}
    aggregate, supporting_evidence, evidence_units = aggregate_criterion_scores(records)
    criterion_scores = {
        criterion: round(100 * aggregate[criterion], 2) for criterion in CORE_CRITERIA
    }
    auxiliary_scores = {
        criterion: round(100 * aggregate[criterion], 2)
        for criterion in AUXILIARY_CRITERIA
    }
    main_strength = max(CORE_CRITERIA, key=aggregate.__getitem__)
    main_gap = min(CORE_CRITERIA, key=aggregate.__getitem__)

    result: dict[str, Any] = {
        "classification_schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "site_name": site_name,
        "digital_twin_adherence_score": round(100 * adherence_score(aggregate), 2),
        "criterion_scores": criterion_scores,
        "auxiliary_scores": auxiliary_scores,
        "self_claim": aggregate["self_claim"] >= SELF_CLAIM_THRESHOLD,
        "main_strength": main_strength,
        "main_gap": main_gap,
    }
    for field in PRESERVED_RESULT_FIELDS:
        if field in metadata:
            result[field] = metadata[field]
    result.update(
        {
            "aggregation": {
                "method": "top_distinct_evidence_weighted_then_core_bottleneck",
                "top_evidence_weights": list(TOP_EVIDENCE_WEIGHTS),
                "core_formula": "100 * (0.60 * geometric_mean + 0.40 * minimum)",
                "self_claim_threshold": 100 * SELF_CLAIM_THRESHOLD,
                "evidence_units": evidence_units,
            },
            "criterion_evidence": supporting_evidence,
        }
    )
    return result


def write_classification_result(evidence_dir: Path, result: dict[str, Any]) -> None:
    temporary = evidence_dir / ".classification.json.tmp"
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(evidence_dir / "classification.json")


def rescore_site(*, site_name: str, evidence_dir: Path) -> dict[str, Any]:
    """Replace an existing categorical result using saved Jev responses only."""
    previous_file = evidence_dir / "classification.json"
    previous: dict[str, Any] = {}
    if previous_file.is_file():
        loaded = json.loads(previous_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError(f"Classification must be a JSON object: {previous_file}")
        previous = loaded

    records = load_jev_records(evidence_dir / "jev_chunks.jsonl")
    previous_sent = previous.get("evidence_chunks_sent")
    if previous_sent is not None and int(previous_sent) != len(records):
        raise ValueError(
            "Saved Jev response count does not match the previous classification: "
            f"responses={len(records)}, classification={previous_sent}"
        )
    metadata = {
        field: previous[field] for field in PRESERVED_RESULT_FIELDS if field in previous
    }
    evidence_file = evidence_dir / "evidence.jsonl"
    available_chunks = (
        sum(1 for _ in evidence_chunks(evidence_file, 20_000))
        if evidence_file.is_file()
        else len(records)
    )
    crawl_was_limited = False
    manifest_file = evidence_dir / "manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            crawl_was_limited = bool(
                manifest.get("crawl_limited", manifest.get("stopped_by_page_limit"))
            )
    metadata.setdefault(
        "evidence_is_partial",
        len(records) < available_chunks or crawl_was_limited,
    )
    metadata.setdefault("crawl_was_limited", crawl_was_limited)
    metadata.setdefault("evidence_chunks_available", available_chunks)
    metadata.setdefault("evidence_chunks_sent", len(records))
    metadata.setdefault("model", "jev-latest")
    if "sampling_strategy" not in metadata:
        metadata["sampling_strategy"] = records[0].get(
            "sampling_strategy", "saved_jev_responses"
        )
    if "source_chunk_numbers" not in metadata:
        metadata["source_chunk_numbers"] = [
            record.get("source_chunk")
            for record in records
            if record.get("source_chunk") is not None
        ]

    result = build_adherence_result(
        site_name=site_name,
        records=records,
        metadata=metadata,
    )
    write_classification_result(evidence_dir, result)
    return result


def classify_site(
    *,
    site_name: str,
    evidence_dir: Path,
    api_key: str,
    model: str = "jev-latest",
    chunk_chars: int = 20_000,
    max_chunks: int = 0,
    evidence_percentage: float | None = None,
    translation_mode: str = "off",
    translation_target: str = "EN",
    deepl_api_key: str = "",
    request_timeout: int = 60,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate optional evidence, send it to Jev, and save the result."""
    if not api_key:
        raise ValueError("A Jev API key must be passed by orchestrator.py")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be a positive integer")
    if max_chunks < 0:
        raise ValueError("max_chunks must be zero or a positive integer")
    if evidence_percentage is not None and not 0 < evidence_percentage <= 100:
        raise ValueError(
            "evidence_percentage must be greater than zero and at most 100"
        )
    if evidence_percentage is not None and max_chunks:
        raise ValueError("Use evidence_percentage or max_chunks, not both")
    if translation_mode not in {"off", "auto", "deepl"}:
        raise ValueError("translation_mode must be off, auto, or deepl")
    if translation_mode != "off" and not deepl_api_key:
        raise ValueError("A DeepL API key is required when translation is enabled")

    evidence_file = evidence_dir / "evidence.jsonl"
    if not evidence_file.exists():
        raise FileNotFoundError(f"Evidence not found: {evidence_file}")

    chunks, source_chunk_numbers, sampling_strategy, total_chunks = (
        select_evidence_chunks(
            evidence_file,
            chunk_chars,
            max_chunks=max_chunks,
            evidence_percentage=evidence_percentage,
        )
    )
    chunks_were_limited = len(chunks) < total_chunks

    crawl_was_limited = False
    crawl_limit_reasons: list[str] = []
    manifest_file = evidence_dir / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        crawl_was_limited = bool(
            manifest.get(
                "crawl_limited",
                manifest.get("stopped_by_page_limit"),
            )
        )
        raw_reasons = manifest.get("crawl_limit_reasons", [])
        if isinstance(raw_reasons, list):
            crawl_limit_reasons = [str(reason) for reason in raw_reasons]
        if not crawl_limit_reasons and manifest.get("stopped_by_page_limit"):
            crawl_limit_reasons = ["page_limit"]

    evidence_is_partial = chunks_were_limited or crawl_was_limited
    jev_records: list[dict[str, Any]] = []
    chunk_log = evidence_dir / "jev_chunks.jsonl"
    translation_log = evidence_dir / "translations.jsonl"
    translation_requests = 0
    translated_pages = 0
    skipped_english_pages = 0
    billed_characters = 0

    with (
        chunk_log.open("w", encoding="utf-8") as jev_log,
        translation_log.open("w", encoding="utf-8") as translation_output,
    ):
        for number, chunk in enumerate(chunks, start=1):
            if translation_mode != "off":
                chunk, translation_records, used_deepl = translate_chunk(
                    chunk,
                    api_key=deepl_api_key,
                    mode=translation_mode,
                    target_language=translation_target,
                    request_timeout=request_timeout,
                )
                translation_requests += int(used_deepl)
                for record in translation_records:
                    record["jev_request"] = number
                    translation_output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    if record["status"] == "translated":
                        translated_pages += 1
                        billed_characters += int(record["billed_characters"])
                    else:
                        skipped_english_pages += 1
                if used_deepl and progress_callback is not None:
                    progress_callback("deepl")

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
            record = {
                "request": number,
                "source_chunk": source_chunk_numbers[number - 1]
                if sampling_strategy != "balanced_across_pages"
                else None,
                "sampling_strategy": sampling_strategy,
                "translation_mode": translation_mode,
                "urls": list(dict.fromkeys(page["url"] for page in chunk["pages"])),
                "probabilities": probabilities,
            }
            jev_records.append(record)
            jev_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress_callback is not None:
                progress_callback("jev")

    result = build_adherence_result(
        site_name=site_name,
        records=jev_records,
        metadata={
            "evidence_is_partial": evidence_is_partial,
            "crawl_was_limited": crawl_was_limited,
            "crawl_limit_reasons": crawl_limit_reasons,
            "chunks_were_limited": chunks_were_limited,
            "evidence_percentage_requested": evidence_percentage,
            "sampling_strategy": sampling_strategy,
            "source_chunk_numbers": source_chunk_numbers,
            "translation": {
                "mode": translation_mode,
                "target_language": translation_target.upper(),
                "deepl_requests": translation_requests,
                "translated_page_fragments": translated_pages,
                "skipped_english_page_fragments": skipped_english_pages,
                "billed_characters": billed_characters,
            },
            "evidence_chunks_available": total_chunks,
            "evidence_chunks_sent": len(chunks),
            "model": model,
        },
    )
    write_classification_result(evidence_dir, result)
    return result
