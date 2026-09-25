"""Transactional local artifacts for crawls and profile runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain.profile import profile_sha256, question_set_sha256, validate_profile


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object in {path}:{number}")
            records.append(value)
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


class RunStore:
    """One profile namespace per site; each attempt owns immutable results."""

    def __init__(self, site_dir: Path, profile: dict[str, Any]):
        validate_profile(profile)
        self.site_dir = site_dir
        self.profile = profile
        self.root = site_dir / "profiles" / profile["id"] / profile["version"]

    def begin(
        self,
        *,
        evidence_sha256: str,
        model: str,
        total: int,
        parameters: dict[str, Any],
    ) -> tuple[str, Path, dict[str, Any]]:
        run_id = uuid4().hex
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "profile_id": self.profile["id"],
            "profile_version": self.profile["version"],
            "profile_sha256": profile_sha256(self.profile),
            "question_set_sha256": question_set_sha256(self.profile),
            "evidence_sha256": evidence_sha256,
            "model": model,
            "requests_planned": total,
            "requests_completed": 0,
            "parameters": parameters,
            "created_at": timestamp(),
        }
        write_json(directory / "run.json", state)
        return run_id, directory, state

    def finish(
        self, directory: Path, state: dict[str, Any], result: dict[str, Any]
    ) -> None:
        write_json(directory / "result.json", result)
        state.update(status="completed", completed_at=timestamp())
        write_json(directory / "run.json", state)
        write_json(self.root / "current.json", {"run_id": state["run_id"]})

    def _read_current_at(
        self, root: Path
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        pointer = read_json(root / "current.json")
        run_id = pointer["run_id"]
        if (
            not isinstance(run_id, str)
            or len(run_id) != 32
            or not all(character in "0123456789abcdef" for character in run_id)
        ):
            raise ValueError("Invalid run pointer")
        directory = root / "runs" / run_id
        state = read_json(directory / "run.json")
        if state["status"] != "completed" or state["run_id"] != run_id:
            raise ValueError("Current run is not complete")
        if state["profile_id"] != self.profile["id"]:
            raise ValueError("Saved profile identity differs")
        if state["question_set_sha256"] != question_set_sha256(self.profile):
            raise ValueError("Saved questions differ from the selected profile")
        result = read_json(directory / "result.json")
        if result.get("profile", {}).get("sha256") != state["profile_sha256"]:
            raise ValueError("Saved result does not match its run profile")
        return directory, state, result

    def current(self) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        if (self.root / "current.json").exists():
            return self._read_current_at(self.root)
        candidates = []
        for version in self.root.parent.iterdir() if self.root.parent.exists() else []:
            if version == self.root or not (version / "current.json").is_file():
                continue
            try:
                candidate = self._read_current_at(version)
                candidates.append(candidate)
            except (OSError, KeyError, ValueError, TypeError):
                continue
        if candidates:
            return max(candidates, key=lambda item: item[1]["completed_at"])
        raise FileNotFoundError("No completed run with matching profile questions")
