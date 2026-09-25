"""Profile-aware classification and offline replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..adapters.deepl import is_english_hint, translate_chunk
from ..domain.profile import (
    criterion_ids,
    profile_sha256,
    question_set_sha256,
    validate_profile,
)
from ..domain.scoring import build_fit_result, normalize_records
from ..evidence.chunks import select_evidence_chunks
from ..storage import (
    RunStore,
    append_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    timestamp,
    write_json,
)
from .failures import StageError


@dataclass(frozen=True)
class ClassificationPlan:
    chunks: list[dict[str, Any]]
    source_numbers: list[int]
    strategy: str
    total_chunks: int
    deepl_requests: int
    deepl_characters: int
    evidence_sha256: str

    @property
    def jev_requests(self) -> int:
        return len(self.chunks)


def plan_classification(
    evidence_file: Path,
    *,
    chunk_chars: int = 20_000,
    percentage: float | None = None,
    max_chunks: int = 0,
    translation: str = "off",
    preserve_whitespace: bool = False,
) -> ClassificationPlan:
    if translation not in {"off", "auto", "deepl"}:
        raise ValueError("Translation must be off, auto or deepl")
    if chunk_chars <= 0 or max_chunks < 0:
        raise ValueError("Invalid chunk limit")
    if percentage is not None and not 0 < percentage <= 100:
        raise ValueError("Percentage must be greater than zero and at most 100")
    if percentage is not None and max_chunks:
        raise ValueError("Choose a percentage or a maximum chunk count")
    chunks, numbers, strategy, total = select_evidence_chunks(
        evidence_file,
        chunk_chars,
        max_chunks=max_chunks,
        evidence_percentage=percentage,
        preserve_whitespace=preserve_whitespace,
    )
    requests = characters = 0
    if translation != "off":
        for chunk in chunks:
            pages = [
                page
                for page in chunk["pages"]
                if translation == "deepl"
                or not is_english_hint(str(page.get("language_hint", "")))
            ]
            requests += bool(pages)
            characters += sum(len(page["text"]) for page in pages)
    return ClassificationPlan(
        chunks,
        numbers,
        strategy,
        total,
        requests,
        characters,
        sha256_file(evidence_file),
    )


def classify_saved(
    *,
    site_name: str,
    site_dir: Path,
    profile: dict[str, Any],
    plan: ClassificationPlan,
    jev_client: Any,
    model: str = "jev-latest",
    translation: str = "off",
    target_language: str = "EN",
    deepl_key: str = "",
    timeout: int = 60,
    translator: Callable[..., Any] = translate_chunk,
    progress: Callable[[str], None] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    evidence_unit_prefix: str | None = None,
) -> dict[str, Any]:
    """Persist each accepted response before aggregation; promote only complete runs."""
    evidence_file = site_dir / "evidence.jsonl"
    validate_profile(profile)
    if sha256_file(evidence_file) != plan.evidence_sha256:
        raise ValueError("Evidence changed since classification was planned")
    if translation not in {"off", "auto", "deepl"}:
        raise ValueError("Invalid translation mode")
    if translation != "off" and not deepl_key:
        raise ValueError("DeepL key required for translation")
    if not plan.jev_requests:
        raise ValueError("No evidence chunks to classify")
    store = RunStore(site_dir, profile)
    _, directory, state = store.begin(
        evidence_sha256=plan.evidence_sha256,
        model=model,
        total=plan.jev_requests,
        parameters={
            "strategy": plan.strategy,
            "total_chunks": plan.total_chunks,
            "translation": translation,
            "target_language": target_language,
        },
    )
    records: list[dict[str, Any]] = []
    billed = 0
    try:
        for number, original in enumerate(plan.chunks, 1):
            chunk = original
            if translation != "off":
                try:
                    chunk, translations, used = translator(
                        chunk,
                        api_key=deepl_key,
                        mode=translation,
                        target_language=target_language,
                        request_timeout=timeout,
                    )
                except Exception as exc:
                    raise StageError("deepl", exc) from exc
                for item in translations:
                    append_jsonl(
                        directory / "translations.jsonl",
                        {
                            "request": number,
                            **item,
                        },
                    )
                    billed += int(item.get("billed_characters", 0))
                if used and progress:
                    progress("deepl")
            try:
                probabilities, raw = jev_client.evaluate(
                    subject=site_name,
                    pages=chunk["pages"],
                    profile=profile,
                    model=model,
                )
            except Exception as exc:
                raise StageError("jev", exc) from exc
            record = {
                "request": number,
                "source_chunk": None
                if plan.strategy == "balanced_across_pages"
                else plan.source_numbers[number - 1],
                "urls": list(dict.fromkeys(page["url"] for page in chunk["pages"])),
                "chunk_sha256": hashlib.sha256(
                    json.dumps(chunk, sort_keys=True, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "profile_id": profile["id"],
                "profile_version": profile["version"],
                "question_set_sha256": state["question_set_sha256"],
                "model": model,
                "probabilities": probabilities,
                "raw_response": raw,
            }
            if evidence_unit_prefix:
                record["evidence_unit"] = f"{evidence_unit_prefix}_{number}"
            append_jsonl(directory / "responses.jsonl", record)
            records.append(record)
            state["requests_completed"] = len(records)
            write_json(directory / "run.json", state)
            if progress:
                progress("jev")

        manifest_path = site_dir / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        limited = bool(
            manifest.get("crawl_limited", manifest.get("stopped_by_page_limit"))
        )
        errors = manifest.get("page_errors", [])
        metadata = {
            "run_id": state["run_id"],
            "model": model,
            "question_set_sha256": state["question_set_sha256"],
            "evidence_sha256": state["evidence_sha256"],
            "evidence_is_partial": limited
            or bool(errors)
            or plan.jev_requests < plan.total_chunks,
            "crawl_was_limited": limited,
            "crawl_limit_reasons": manifest.get("crawl_limit_reasons", []),
            "crawl_errors": len(errors),
            "evidence_chunks_available": plan.total_chunks,
            "evidence_chunks_sent": plan.jev_requests,
            "sampling_strategy": plan.strategy,
            "source_chunk_numbers": plan.source_numbers,
            "translation": {
                "mode": translation,
                "deepl_requests": plan.deepl_requests,
                "billed_characters": billed,
            },
        }
        if extra_metadata:
            if set(extra_metadata) & set(metadata):
                raise ValueError("Extra metadata overlaps classification fields")
            metadata.update(extra_metadata)
        result = build_fit_result(
            subject=site_name,
            records=records,
            profile=profile,
            metadata=metadata,
        )
        store.finish(directory, state, result)
        return result
    except Exception as exc:
        state.update(
            status="failed",
            failed_at=timestamp(),
            failed_stage=exc.stage if isinstance(exc, StageError) else "classification",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(directory / "run.json", state)
        raise


def replay(
    *, site_name: str, site_dir: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    store = RunStore(site_dir, profile)
    directory, state, previous = store.current()
    records = read_jsonl(directory / "responses.jsonl")
    if len(records) != state["requests_planned"]:
        raise ValueError("Incomplete saved responses")
    if any(
        record.get("question_set_sha256") != question_set_sha256(profile)
        or record.get("model") != state["model"]
        for record in records
    ):
        raise ValueError("Incompatible saved response identity")
    normalized = normalize_records(records, criterion_ids(profile))
    metadata = {
        key: value
        for key, value in previous.items()
        if key
        not in {
            "profile",
            "subject",
            "fit_score",
            "criterion_scores",
            "auxiliary_scores",
            "auxiliary_flags",
            "main_strength",
            "main_gap",
            "aggregation",
            "criterion_evidence",
            "fit_result_schema_version",
        }
    }
    metadata["offline_rescore"] = True
    result = build_fit_result(
        subject=site_name, records=normalized, profile=profile, metadata=metadata
    )
    write_json(directory / "derived" / f"{profile_sha256(profile)}.json", result)
    return result


def current_result(site_dir: Path, profile: dict[str, Any]) -> dict[str, Any]:
    directory, state, result = RunStore(site_dir, profile).current()
    if state["profile_sha256"] == profile_sha256(profile):
        return result
    derived = directory / "derived" / f"{profile_sha256(profile)}.json"
    if not derived.exists():
        raise ValueError("Profile scoring changed; run --mode score for offline replay")
    return read_json(derived)
