"""Explicit crawl cache and offline recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..domain.urls import canonicalize, in_scope
from ..evidence.chunks import evidence_chunks
from ..storage import read_json, read_jsonl, sha256_file, timestamp, write_json


def signature(site: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_url": site["url"],
        **{
            key: value
            for key, value in config.items()
            if key not in {"verbose", "log_callback"}
        },
    }


def write_state(
    site: dict[str, str],
    directory: Path,
    config: dict[str, Any],
    *,
    completed_at: str | None = None,
) -> None:
    evidence = directory / "evidence.jsonl"
    manifest = directory / "manifest.json"
    if not evidence.is_file() or not manifest.is_file():
        raise FileNotFoundError("Cannot complete crawl without evidence and manifest")
    write_json(
        directory / "crawl_state.json",
        {
            "schema_version": 2,
            "status": "completed",
            "site_name": site["name"],
            "completed_at": completed_at or timestamp(),
            "crawl_signature": signature(site, config),
            "evidence_sha256": sha256_file(evidence),
            "manifest_sha256": sha256_file(manifest),
        },
    )


def recent_crawl(
    site: dict[str, str],
    directory: Path,
    config: dict[str, Any],
    max_age_hours: float,
    now: datetime | None = None,
) -> bool:
    try:
        state = read_json(directory / "crawl_state.json")
        at = datetime.fromisoformat(state["completed_at"])
        if at.tzinfo is None:
            return False
        age = (now or datetime.now(UTC)) - at.astimezone(UTC)
        return bool(
            state.get("schema_version") == 2
            and state.get("status") == "completed"
            and state.get("site_name") == site["name"]
            and state.get("crawl_signature") == signature(site, config)
            and state.get("evidence_sha256")
            == sha256_file(directory / "evidence.jsonl")
            and state.get("manifest_sha256") == sha256_file(directory / "manifest.json")
            and timedelta(0) <= age <= timedelta(hours=max_age_hours)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def recover(
    site: dict[str, str], directory: Path, config: dict[str, Any], chunk_chars: int
) -> str:
    """Recover only a verifiably complete persisted site without network."""
    evidence = directory / "evidence.jsonl"
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    configured = (
        site["url"]
        if site["url"].startswith(("https://", "http://"))
        else "https://" + site["url"]
    )
    if manifest.get("site_name") != site["name"] or canonicalize(
        manifest.get("root_url", ""), keep_query=False
    ) != canonicalize(configured, keep_query=False):
        raise ValueError("Crawl identity mismatch")
    records = read_jsonl(evidence)
    if len(records) != manifest.get("pages_saved") or not records:
        raise ValueError("Crawl page count mismatch")
    if any(
        not isinstance(record.get("url"), str)
        or not in_scope(record["url"], configured, False)
        or not isinstance(record.get("text"), str)
        for record in records
    ):
        raise ValueError("Invalid crawl page content or URL scope")
    if not any(evidence_chunks(evidence, chunk_chars)):
        raise ValueError("No textual evidence")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise TypeError("Crawl manifest has no creation timestamp")
    at = datetime.fromisoformat(created_at)
    if at.tzinfo is None or at > datetime.now(UTC):
        raise ValueError("Invalid crawl manifest timestamp")
    write_state(site, directory, config, completed_at=created_at)
    return f"recovered {len(records)} pages"
