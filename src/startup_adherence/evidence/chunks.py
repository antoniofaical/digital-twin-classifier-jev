"""Bounded evidence selection with source provenance."""

from __future__ import annotations

import json
from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any


def evidence_chunks(
    evidence_file: Path, chunk_chars: int, *, preserve_whitespace: bool = False
) -> Iterator[dict[str, Any]]:
    pages: list[dict[str, str]] = []
    size = 0
    with evidence_file.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            original = str(record.get("text", ""))
            if not original.strip():
                continue
            text = original if preserve_whitespace else original.strip()
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
    preserve_whitespace: bool = False,
) -> tuple[list[dict[str, Any]], list[int], str, int]:
    all_chunks = list(
        evidence_chunks(
            evidence_file, chunk_chars, preserve_whitespace=preserve_whitespace
        )
    )
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
