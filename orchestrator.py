"""Configuration and entry point for scraping and Jev classification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO, TypeVar

from classifier import (
    classify_site,
    evidence_chunks,
    is_english_hint,
    select_evidence_chunks,
)
from scraper import (
    DEFAULT_MAX_CRAWL_SECONDS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_SITEMAPS,
    canonicalize,
    evidence_directory_name,
    scrape_site,
)

SITES = [
    {"name": "madidt", "url": "https://madidt.com/"},
]

EVIDENCE_ROOT = Path("evidence")
JEV_API_KEY_ENV = "TYPESAFE_PSN_DIG_TWIN_CLASS"
DEEPL_API_KEY_ENV = "DEEPL_API_KEY"

SMOKE_MAX_PAGES = 5
SMOKE_MAX_CHUNKS = 1
DEFAULT_CRAWL_MAX_AGE_HOURS = 24.0
CRAWL_STATE_VERSION = 1
CRAWL_STATE_FILENAME = "crawl_state.json"
GENERATED_ARTIFACT_FILENAMES = (
    "classification.json",
    CRAWL_STATE_FILENAME,
    ".crawl_state.json.tmp",
    "evidence.jsonl",
    "jev_chunks.jsonl",
    "manifest.json",
    "translations.jsonl",
)

SCRAPER_CONFIG = {
    "include_subdomains": False,
    "include_query_urls": False,
    "respect_robots": True,
    "request_timeout": 20,
    "max_requests": DEFAULT_MAX_REQUESTS,
    "max_queue_size": DEFAULT_MAX_QUEUE_SIZE,
    "max_crawl_seconds": DEFAULT_MAX_CRAWL_SECONDS,
    "max_sitemaps": DEFAULT_MAX_SITEMAPS,
}

CLASSIFIER_CONFIG = {
    "model": "jev-latest",
    "chunk_chars": 20_000,
    "positive_threshold": 0.70,
    "negative_threshold": 0.30,
    "request_timeout": 60,
}

T = TypeVar("T")

ANSI_COLORS = {
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
}
ANSI_RESET = "\033[0m"


class BulkExecutionError(RuntimeError):
    """Raised after all possible sites finish when one or more sites failed."""


@dataclass
class ProgressStage:
    label: str
    total: int
    color: str
    completed: int = 0


class TerminalProgress:
    """Render thread-safe progress stages on persistent terminal lines."""

    def __init__(
        self,
        total: int,
        *,
        enabled: bool,
        stream: TextIO | None = None,
        label: str = "Sites",
        color: str = "blue",
    ) -> None:
        self.total = total
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.color_enabled = enabled and "NO_COLOR" not in os.environ
        self.successful = 0
        self.failed = 0
        self.cancelled = 0
        self.reused = 0
        self.active = 0
        self._terminal_sites: set[str] = set()
        self._started_at = time.monotonic()
        self._stages = {
            "sites": ProgressStage(label=label, total=total, color=color),
        }
        self._last_line_count = 0
        self._running = False
        self._lock = threading.RLock()

    @property
    def completed(self) -> int:
        return len(self._terminal_sites)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._started_at = time.monotonic()
            self._draw_locked()

    def add_stage(self, key: str, *, label: str, total: int, color: str) -> None:
        if total < 0:
            raise ValueError("progress total must be zero or positive")
        with self._lock:
            if key in self._stages:
                raise ValueError(f"progress stage already exists: {key}")
            self._stages[key] = ProgressStage(
                label=label,
                total=total,
                color=color,
            )
            self._draw_locked()

    def advance_stage(self, key: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("progress amount must be zero or positive")
        with self._lock:
            stage = self._stages[key]
            stage.completed += amount
            if stage.completed > stage.total:
                raise ValueError(f"progress stage exceeds total: {key}")
            self._draw_locked()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self.active = 0
            if self.enabled:
                self._draw_locked()
                self.stream.write("\n")
                self.stream.flush()
            self._running = False

    def log(self, message: str) -> None:
        with self._lock:
            if self.enabled and self._running:
                self._clear_locked()
                self._last_line_count = 0
                self.stream.write(f"{message}\n")
                self._draw_locked()
                self.stream.flush()
            else:
                print(message, file=self.stream, flush=True)

    def set_active(self, active: int) -> None:
        with self._lock:
            self.active = max(0, active)
            self._draw_locked()

    def record_terminal(
        self,
        site_name: str,
        status: str,
        *,
        reused: bool = False,
    ) -> None:
        with self._lock:
            if site_name in self._terminal_sites:
                raise ValueError(f"site progress already recorded: {site_name}")
            if status == "completed":
                self.successful += 1
                self.reused += int(reused)
            elif status == "failed":
                self.failed += 1
            elif status == "cancelled":
                self.cancelled += 1
            else:
                raise ValueError(f"unknown terminal progress status: {status}")
            self._terminal_sites.add(site_name)
            self._stages["sites"].completed = self.completed
            if self.completed > self.total:
                raise ValueError("completed site progress exceeds selected sites")
            self._draw_locked()

    def _line(self, key: str, stage: ProgressStage) -> str:
        percentage = (
            100 if stage.total == 0 else round(100 * stage.completed / stage.total)
        )
        elapsed = max(0, int(time.monotonic() - self._started_at))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        details = f" {stage.completed}/{stage.total} {percentage:3d}%"
        if key == "sites":
            details += (
                f" completed={self.successful} reused={self.reused} "
                f"failed={self.failed}"
                f"{' cancelled=' + str(self.cancelled) if self.cancelled else ''} "
                f"active={self.active} "
                f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            )
        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        label = stage.label
        if self.color_enabled:
            color = ANSI_COLORS.get(stage.color, "")
            label = f"{color}{label}{ANSI_RESET}"
        prefix = f"{label}  "
        bar_width = min(
            40,
            max(0, columns - len(details) - len(stage.label) - len("  []") - 1),
        )
        if bar_width < 8:
            return f"{label}{details}"
        filled = min(
            bar_width,
            round(bar_width * stage.completed / max(1, stage.total)),
        )
        filled_bar = "█" * filled
        if self.color_enabled and filled_bar:
            color = ANSI_COLORS.get(stage.color, "")
            filled_bar = f"{color}{filled_bar}{ANSI_RESET}"
        bar = filled_bar + "-" * (bar_width - filled)
        return f"{prefix}[{bar}]{details}"

    def _clear_locked(self) -> None:
        if not self.enabled or not self._last_line_count:
            return
        for index in range(self._last_line_count):
            self.stream.write("\r\033[2K")
            if index < self._last_line_count - 1:
                self.stream.write("\033[1A")

    def _draw_locked(self) -> None:
        if not self.enabled or not self._running:
            return
        self._clear_locked()
        lines = [self._line(key, stage) for key, stage in self._stages.items()]
        self.stream.write("\n".join(lines))
        self.stream.flush()
        self._last_line_count = len(lines)


@dataclass(frozen=True)
class ClassificationPlan:
    site: dict[str, str]
    site_dir: Path
    classifier_config: dict[str, Any]
    chunks_available: int
    chunks_selected: int
    translation_requests: int
    translation_characters: int


@dataclass(frozen=True)
class CrawlRecoveryResult:
    status: str
    detail: str


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return number


def percentage_value(value: str) -> float:
    normalized = value.strip().removesuffix("%").strip().replace(",", ".")
    try:
        percentage = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("percentage must be from 0 to 100") from exc
    if not 0 <= percentage <= 100:
        raise argparse.ArgumentTypeError("percentage must be from 0 to 100")
    return percentage


def non_negative_number(value: str) -> float:
    try:
        number = float(value.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be zero or a positive number"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or a positive number")
    return number


def non_negative_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crawl limit must be zero or a positive integer"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError(
            "crawl limit must be zero or a positive integer"
        )
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl company websites and classify saved evidence with Jev."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "crawl", "classify"),
        default="smoke",
        help=(
            "smoke: bounded crawl plus one Jev call; "
            "crawl: full crawl with no API calls; "
            "classify: choose a percentage of previously saved evidence"
        ),
    )
    parser.add_argument(
        "--translation",
        choices=("off", "auto", "deepl"),
        default="off",
        help=(
            "off: send original text; auto: skip pages explicitly marked as English; "
            "deepl: translate every selected page fragment"
        ),
    )
    parser.add_argument(
        "--translation-target",
        default="EN",
        help="DeepL target-language code, default: EN",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--site",
        action="append",
        dest="site_names",
        default=[],
        metavar="NAME",
        help="select one configured site; repeat to select multiple sites",
    )
    selection.add_argument(
        "--sites",
        nargs="+",
        dest="site_names_group",
        metavar="NAME",
        help="select configured sites by name, or use 'all'",
    )
    parser.add_argument(
        "--sites-file",
        type=Path,
        help="load the configured site list from a JSON file instead of SITES",
    )
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=1,
        help="maximum number of sites processed concurrently, default: 1",
    )
    parser.add_argument(
        "--crawl-max-age-hours",
        type=non_negative_number,
        default=DEFAULT_CRAWL_MAX_AGE_HOURS,
        help=(
            "reuse a completed full crawl for this many hours, default: 24; "
            "only applies to crawl mode"
        ),
    )
    parser.add_argument(
        "--force-crawl",
        action="store_true",
        help="ignore a recent crawl and fetch every selected site again",
    )
    parser.add_argument(
        "--recover-existing-crawls",
        action="store_true",
        help=(
            "create crawl state for complete local evidence and manifests without "
            "network or API calls; only applies to crawl mode"
        ),
    )
    parser.add_argument(
        "--max-pages-per-site",
        type=non_negative_integer,
        default=DEFAULT_MAX_PAGES,
        help=f"maximum unique pages saved per site, default: {DEFAULT_MAX_PAGES}; 0 disables",
    )
    parser.add_argument(
        "--max-requests-per-site",
        type=non_negative_integer,
        default=DEFAULT_MAX_REQUESTS,
        help=(
            "maximum page requests attempted per site, default: "
            f"{DEFAULT_MAX_REQUESTS}; 0 disables"
        ),
    )
    parser.add_argument(
        "--max-queue-size",
        type=non_negative_integer,
        default=DEFAULT_MAX_QUEUE_SIZE,
        help=(
            "maximum pending URLs retained per site, default: "
            f"{DEFAULT_MAX_QUEUE_SIZE}; 0 disables"
        ),
    )
    parser.add_argument(
        "--max-crawl-seconds",
        type=non_negative_number,
        default=DEFAULT_MAX_CRAWL_SECONDS,
        help=(
            "maximum crawl duration per site in seconds, default: "
            f"{DEFAULT_MAX_CRAWL_SECONDS:g}; 0 disables"
        ),
    )
    parser.add_argument(
        "--max-sitemaps-per-site",
        type=non_negative_integer,
        default=DEFAULT_MAX_SITEMAPS,
        help=(
            "maximum sitemap documents fetched per site, default: "
            f"{DEFAULT_MAX_SITEMAPS}; 0 disables"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="show page progress; use -vv for HTTP, sitemap, robots, and error details",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable persistent progress bars",
    )
    parser.add_argument(
        "--percentage",
        type=percentage_value,
        help="evidence percentage for classify mode; otherwise prompt once for the bulk",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="approve the displayed bulk API plan without an interactive confirmation",
    )
    args = parser.parse_args(argv)
    if args.percentage is not None and args.mode != "classify":
        parser.error("--percentage can only be used with --mode classify")
    if args.force_crawl and args.mode != "crawl":
        parser.error("--force-crawl can only be used with --mode crawl")
    if args.recover_existing_crawls and args.mode != "crawl":
        parser.error("--recover-existing-crawls can only be used with --mode crawl")
    if args.recover_existing_crawls and args.force_crawl:
        parser.error("--recover-existing-crawls cannot be combined with --force-crawl")
    return args


def load_sites(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("sites")
    if not isinstance(payload, list):
        raise TypeError(
            "sites file must contain a JSON list or a {'sites': [...]} object"
        )
    sites: list[dict[str, str]] = []
    for index, value in enumerate(payload, start=1):
        if not isinstance(value, dict) or not value.get("name") or not value.get("url"):
            raise ValueError(f"sites file entry {index} must contain name and url")
        sites.append({"name": str(value["name"]), "url": str(value["url"])})
    return sites


def select_sites(
    configured_sites: list[dict[str, str]], requested_names: list[str]
) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for site in configured_sites:
        name = str(site.get("name", "")).strip()
        url = str(site.get("url", "")).strip()
        if not name or not url:
            raise ValueError(
                "every configured site must contain a non-empty name and url"
            )
        normalized = name.casefold()
        if normalized in by_name:
            raise ValueError(f"duplicate configured site name: {name}")
        by_name[normalized] = {"name": name, "url": url}

    if not requested_names or [name.casefold() for name in requested_names] == ["all"]:
        return list(by_name.values())
    if any(name.casefold() == "all" for name in requested_names):
        raise ValueError("'all' cannot be combined with individual site names")

    selected: list[dict[str, str]] = []
    selected_names: set[str] = set()
    unknown: list[str] = []
    for requested in requested_names:
        normalized = requested.casefold()
        site = by_name.get(normalized)
        if site is None:
            unknown.append(requested)
        elif normalized not in selected_names:
            selected.append(site)
            selected_names.add(normalized)
    if unknown:
        available = ", ".join(site["name"] for site in by_name.values())
        raise ValueError(
            f"unknown site(s): {', '.join(unknown)}. Available sites: {available}"
        )
    return selected


def evidence_directories(
    sites: list[dict[str, str]], evidence_root: Path
) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    owners: dict[str, str] = {}
    for site in sites:
        directory_name = evidence_directory_name(site["name"])
        normalized = directory_name.casefold()
        if normalized in owners:
            raise ValueError(
                f"site names '{owners[normalized]}' and '{site['name']}' resolve to "
                f"the same evidence directory: {directory_name}"
            )
        owners[normalized] = site["name"]
        directories[site["name"]] = evidence_root / directory_name
    return directories


def crawl_signature(
    site: dict[str, str], scraper_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "root_url": site["url"],
        "include_subdomains": bool(scraper_config["include_subdomains"]),
        "include_query_urls": bool(scraper_config["include_query_urls"]),
        "respect_robots": bool(scraper_config["respect_robots"]),
        "max_pages": int(scraper_config["max_pages"]),
        "max_requests": int(scraper_config["max_requests"]),
        "max_queue_size": int(scraper_config["max_queue_size"]),
        "max_crawl_seconds": float(scraper_config["max_crawl_seconds"]),
        "max_sitemaps": int(scraper_config["max_sitemaps"]),
    }


def crawl_signatures_compatible(
    stored: object,
    current: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    if stored == current:
        return True
    if not isinstance(stored, dict):
        return False
    legacy_keys = {
        "root_url",
        "include_subdomains",
        "include_query_urls",
        "respect_robots",
        "max_pages",
    }
    if set(stored) != legacy_keys:
        return False
    if any(stored[key] != current[key] for key in legacy_keys - {"max_pages"}):
        return False
    return (
        int(stored["max_pages"]) == 0
        and not manifest.get("stopped_by_page_limit")
        and not manifest.get("crawl_limited")
    )


def recent_crawl(
    *,
    site: dict[str, str],
    site_dir: Path,
    scraper_config: dict[str, Any],
    max_age_hours: float,
    now: datetime | None = None,
) -> tuple[bool, float | None]:
    """Trust only the explicit state written by this cache implementation."""
    state_file = site_dir / CRAWL_STATE_FILENAME
    if not state_file.exists():
        return False, None
    if (
        not (site_dir / "evidence.jsonl").exists()
        or not (site_dir / "manifest.json").exists()
    ):
        return False, None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        manifest = json.loads((site_dir / "manifest.json").read_text(encoding="utf-8"))
        completed_at = datetime.fromisoformat(str(state["completed_at"]))
    except (OSError, ValueError, TypeError, KeyError):
        return False, None
    if completed_at.tzinfo is None:
        return False, None
    if state.get("schema_version") != CRAWL_STATE_VERSION:
        return False, None
    if state.get("status") != "completed":
        return False, None
    if state.get("site_name") != site["name"]:
        return False, None
    if not crawl_signatures_compatible(
        state.get("crawl_signature"),
        crawl_signature(site, scraper_config),
        manifest,
    ):
        return False, None

    current_time = now or datetime.now(timezone.utc)
    age = current_time - completed_at.astimezone(timezone.utc)
    if age < timedelta(0):
        return False, None
    age_hours = age.total_seconds() / 3600
    return age_hours <= max_age_hours, age_hours


def clean_generated_artifacts(site_dir: Path) -> list[str]:
    """Delete only files generated by this pipeline, never arbitrary user files."""
    if site_dir.is_symlink():
        raise ValueError(f"Refusing to clean symlinked evidence directory: {site_dir}")
    removed: list[str] = []
    for filename in GENERATED_ARTIFACT_FILENAMES:
        artifact = site_dir / filename
        if artifact.exists() or artifact.is_symlink():
            artifact.unlink()
            removed.append(filename)
    return removed


def write_crawl_state(
    *,
    site: dict[str, str],
    site_dir: Path,
    scraper_config: dict[str, Any],
    completed_at: datetime | None = None,
) -> None:
    if (
        not (site_dir / "evidence.jsonl").exists()
        or not (site_dir / "manifest.json").exists()
    ):
        raise FileNotFoundError(
            f"Cannot mark crawl complete without evidence and manifest in {site_dir}"
        )
    manifest = json.loads((site_dir / "manifest.json").read_text(encoding="utf-8"))
    state = {
        "schema_version": CRAWL_STATE_VERSION,
        "status": "completed",
        "site_name": site["name"],
        "completed_at": (completed_at or datetime.now(timezone.utc)).isoformat(),
        "crawl_signature": crawl_signature(site, scraper_config),
        "crawl_limited": bool(manifest.get("crawl_limited")),
        "crawl_limit_reasons": list(manifest.get("crawl_limit_reasons", [])),
    }
    temporary = site_dir / ".crawl_state.json.tmp"
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(site_dir / CRAWL_STATE_FILENAME)


def recover_existing_crawl(
    *,
    site: dict[str, str],
    site_dir: Path,
    scraper_config: dict[str, Any],
    chunk_chars: int,
) -> CrawlRecoveryResult:
    """Validate a fully persisted crawl and recreate only its cache state."""
    evidence_file = site_dir / "evidence.jsonl"
    manifest_file = site_dir / "manifest.json"
    missing = [
        path.name for path in (evidence_file, manifest_file) if not path.is_file()
    ]
    if missing:
        return CrawlRecoveryResult("skipped", f"missing {', '.join(missing)}")

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return CrawlRecoveryResult("skipped", f"invalid manifest: {exc}")
    if not isinstance(manifest, dict):
        return CrawlRecoveryResult("skipped", "manifest must be a JSON object")
    if manifest.get("site_name") != site["name"]:
        return CrawlRecoveryResult("skipped", "manifest site name does not match")

    configured_root = str(site["url"])
    if not configured_root.lower().startswith(("http://", "https://")):
        configured_root = "https://" + configured_root
    expected_root = canonicalize(
        configured_root,
        keep_query=bool(scraper_config["include_query_urls"]),
    )
    manifest_root = canonicalize(
        str(manifest.get("root_url", "")),
        keep_query=bool(scraper_config["include_query_urls"]),
    )
    if not manifest_root or manifest_root != expected_root:
        return CrawlRecoveryResult("skipped", "manifest root URL does not match")

    try:
        completed_at = datetime.fromisoformat(str(manifest["created_at"]))
        pages_saved = int(manifest["pages_saved"])
        records_saved = sum(
            bool(line.strip())
            for line in evidence_file.read_text(encoding="utf-8").splitlines()
        )
        chunks = count_chunks(evidence_file, chunk_chars)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return CrawlRecoveryResult("skipped", f"invalid persisted evidence: {exc}")
    if completed_at.tzinfo is None:
        return CrawlRecoveryResult("skipped", "manifest timestamp has no timezone")
    if pages_saved <= 0 or records_saved != pages_saved:
        return CrawlRecoveryResult(
            "skipped",
            f"page count mismatch: manifest={pages_saved}, evidence={records_saved}",
        )
    if chunks <= 0:
        return CrawlRecoveryResult("skipped", "no textual evidence chunks")

    write_crawl_state(
        site=site,
        site_dir=site_dir,
        scraper_config=scraper_config,
        completed_at=completed_at,
    )
    return CrawlRecoveryResult(
        "recovered",
        f"pages={pages_saved}, chunks={chunks}, completed_at={completed_at.isoformat()}",
    )


def prompt_evidence_percentage(chunk_count: int, site_count: int = 1) -> float:
    while True:
        scope = "" if site_count == 1 else f" across {site_count} sites"
        raw = input(
            f"{chunk_count} chunks of evidence generated{scope}. "
            "How much do you want to include (%)? "
        )
        try:
            return percentage_value(raw)
        except argparse.ArgumentTypeError:
            print("Enter a number from 0 to 100.")


def prompt_confirmation(message: str) -> bool:
    answer = input(f"{message} Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def count_chunks(evidence_file: Path, chunk_chars: int) -> int:
    return sum(1 for _ in evidence_chunks(evidence_file, chunk_chars))


def translation_workload(chunks: list[dict[str, Any]], mode: str) -> tuple[int, int]:
    request_count = 0
    character_count = 0
    for chunk in chunks:
        candidates = [
            page
            for page in chunk["pages"]
            if mode == "deepl"
            or not is_english_hint(str(page.get("language_hint", "")))
        ]
        if candidates:
            request_count += 1
            character_count += sum(len(page["text"]) for page in candidates)
    return request_count, character_count


def terminal_progress_enabled(*, disabled: bool, stream: TextIO | None = None) -> bool:
    output = stream or sys.stdout
    is_terminal = getattr(output, "isatty", None)
    return not disabled and callable(is_terminal) and bool(is_terminal())


def run_site_jobs(
    sites: list[dict[str, str]],
    workers: int,
    operation: Callable[[dict[str, str]], T],
    *,
    progress: TerminalProgress | None = None,
    successes_are_terminal: bool = False,
    reused_names: set[str] | None = None,
) -> tuple[dict[str, T], dict[str, str]]:
    """Run isolated site jobs and collect every result before reporting failures."""
    results: dict[str, T] = {}
    errors: dict[str, str] = {}
    reused_names = reused_names or set()

    def record(site: dict[str, str], *, succeeded: bool) -> None:
        if progress is None:
            return
        if succeeded and successes_are_terminal:
            progress.record_terminal(
                site["name"],
                "completed",
                reused=site["name"] in reused_names,
            )
        elif not succeeded:
            progress.record_terminal(site["name"], "failed")

    def update_active(remaining: int) -> None:
        if progress is not None:
            progress.set_active(min(workers, remaining))

    update_active(len(sites))
    if workers == 1 or len(sites) <= 1:
        for index, site in enumerate(sites, start=1):
            try:
                results[site["name"]] = operation(site)
            except Exception as exc:  # noqa: BLE001 - isolate failures in a bulk run
                errors[site["name"]] = f"{type(exc).__name__}: {exc}"
                record(site, succeeded=False)
            else:
                record(site, succeeded=True)
            update_active(len(sites) - index)
        return results, errors

    with ThreadPoolExecutor(max_workers=min(workers, len(sites))) as executor:
        futures = {executor.submit(operation, site): site for site in sites}
        for index, future in enumerate(as_completed(futures), start=1):
            site = futures[future]
            try:
                results[site["name"]] = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate failures in a bulk run
                errors[site["name"]] = f"{type(exc).__name__}: {exc}"
                record(site, succeeded=False)
            else:
                record(site, succeeded=True)
            update_active(len(sites) - index)
    return results, errors


def prepare_classification_plan(
    site: dict[str, str],
    site_dir: Path,
    classifier_config: dict[str, Any],
    translation_mode: str,
) -> ClassificationPlan:
    evidence_file = site_dir / "evidence.jsonl"
    chunks, _, _, total_chunks = select_evidence_chunks(
        evidence_file,
        classifier_config["chunk_chars"],
        max_chunks=classifier_config["max_chunks"],
        evidence_percentage=classifier_config.get("evidence_percentage"),
    )
    translation_requests = 0
    translation_characters = 0
    if translation_mode != "off":
        translation_requests, translation_characters = translation_workload(
            chunks,
            translation_mode,
        )
    return ClassificationPlan(
        site=site,
        site_dir=site_dir,
        classifier_config=classifier_config,
        chunks_available=total_chunks,
        chunks_selected=len(chunks),
        translation_requests=translation_requests,
        translation_characters=translation_characters,
    )


def print_bulk_plan(plans: list[ClassificationPlan], translation_mode: str) -> None:
    total_available = sum(plan.chunks_available for plan in plans)
    total_selected = sum(plan.chunks_selected for plan in plans)
    translation_requests = sum(plan.translation_requests for plan in plans)
    translation_characters = sum(plan.translation_characters for plan in plans)
    print("BULK CLASSIFICATION PLAN")
    print(f"Sites ready: {len(plans)}")
    print(f"Evidence chunks available: {total_available}")
    print(f"Evidence chunks selected: {total_selected}")
    if translation_mode != "off":
        print(f"DeepL characters: {translation_characters}")
        print(f"DeepL requests: {translation_requests}")
    print(f"Jev requests: {total_selected}")
    for plan in plans:
        print(
            f"[{plan.site['name']}] {plan.chunks_selected} of "
            f"{plan.chunks_available} chunks selected"
        )


def print_bulk_summary(
    *,
    selected: int,
    completed: int,
    errors: dict[str, str],
    cancelled: int = 0,
    reused: int = 0,
) -> None:
    print(
        "BULK SUMMARY: "
        f"selected={selected}, completed={completed}, "
        f"failed={len(errors)}, cancelled={cancelled}, reused={reused}"
    )
    for name, error in errors.items():
        print(f"[{name}] failed: {error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = args.mode
    run_scraper = mode in {"smoke", "crawl"}
    run_classifier = mode in {"smoke", "classify"}
    smoke_test = mode == "smoke"

    configured_sites = load_sites(args.sites_file) if args.sites_file else SITES
    requested_names = args.site_names_group or args.site_names
    sites = select_sites(configured_sites, requested_names)
    if not sites:
        raise ValueError("no sites are configured")
    site_dirs = evidence_directories(sites, EVIDENCE_ROOT)

    scraper_config = {
        **SCRAPER_CONFIG,
        "max_pages": (SMOKE_MAX_PAGES if smoke_test else args.max_pages_per_site),
        "max_requests": args.max_requests_per_site,
        "max_queue_size": args.max_queue_size,
        "max_crawl_seconds": args.max_crawl_seconds,
        "max_sitemaps": args.max_sitemaps_per_site,
        "verbose": args.verbose,
    }
    base_classifier_config = {
        **CLASSIFIER_CONFIG,
        "max_chunks": SMOKE_MAX_CHUNKS if smoke_test else 0,
    }

    api_key = os.getenv(JEV_API_KEY_ENV, "")
    if run_classifier and not api_key:
        raise RuntimeError(
            f"{JEV_API_KEY_ENV} is not set. Set it in the environment before running."
        )
    deepl_api_key = os.getenv(DEEPL_API_KEY_ENV, "")
    if run_classifier and args.translation != "off" and not deepl_api_key:
        raise RuntimeError(
            f"{DEEPL_API_KEY_ENV} is not set. Set it in the environment before running."
        )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    if args.recover_existing_crawls:
        print("CRAWL RECOVERY: no website, Jev, or DeepL calls will be made.")
        recovery_progress = TerminalProgress(
            len(sites),
            enabled=terminal_progress_enabled(disabled=args.no_progress),
            label="Recovering",
            color="blue",
        )
        recovery_progress.start()

        def recover(site: dict[str, str]) -> CrawlRecoveryResult:
            return recover_existing_crawl(
                site=site,
                site_dir=site_dirs[site["name"]],
                scraper_config=scraper_config,
                chunk_chars=base_classifier_config["chunk_chars"],
            )

        recoveries, recovery_errors = run_site_jobs(
            sites,
            args.workers,
            recover,
            progress=recovery_progress,
            successes_are_terminal=True,
        )
        recovery_progress.stop()
        recovered_count = 0
        skipped_count = 0
        for site in sites:
            result = recoveries.get(site["name"])
            if result is None:
                continue
            if result.status == "recovered":
                recovered_count += 1
            else:
                skipped_count += 1
            print(f"[{site['name']}] {result.status}: {result.detail}")
        print(
            "RECOVERY SUMMARY: "
            f"selected={len(sites)}, recovered={recovered_count}, "
            f"skipped={skipped_count}, failed={len(recovery_errors)}"
        )
        for name, error in recovery_errors.items():
            print(f"[{name}] failed: {error}")
        if recovery_errors:
            raise BulkExecutionError(
                f"{len(recovery_errors)} site(s) failed during crawl recovery"
            )
        return

    if smoke_test:
        print(
            f"SMOKE TEST: at most {SMOKE_MAX_PAGES} pages and "
            f"{SMOKE_MAX_CHUNKS} Jev request per site."
        )
    elif mode == "crawl":
        print("FULL CRAWL: no Jev or DeepL API calls will be made.")
        print(
            "CRAWL LIMITS PER SITE: "
            f"pages={args.max_pages_per_site}, "
            f"requests={args.max_requests_per_site}, "
            f"queue={args.max_queue_size}, "
            f"seconds={args.max_crawl_seconds:g}, "
            f"sitemaps={args.max_sitemaps_per_site} (0 disables a limit)."
        )
        if args.force_crawl:
            print("CRAWL CACHE: bypassed by --force-crawl.")
        else:
            print(
                "CRAWL CACHE: completed crawls up to "
                f"{args.crawl_max_age_hours:g} hours old will be reused."
            )
    else:
        print("CLASSIFICATION: no API calls occur before bulk approval.")
    print(f"Selected {len(sites)} site(s); using up to {args.workers} worker(s).")

    progress: TerminalProgress | None = None
    if run_scraper:
        progress = TerminalProgress(
            len(sites),
            enabled=terminal_progress_enabled(disabled=args.no_progress),
            label="Scraping",
            color="cyan",
        )
        progress.start()

    def emit(message: str) -> None:
        if progress is None:
            print(message, flush=True)
        else:
            progress.log(message)

    errors: dict[str, str] = {}
    reused_names: set[str] = set()
    sites_to_scrape = list(sites)

    if mode == "crawl" and not args.force_crawl:
        sites_to_scrape = []
        for site in sites:
            is_recent, age_hours = recent_crawl(
                site=site,
                site_dir=site_dirs[site["name"]],
                scraper_config=scraper_config,
                max_age_hours=args.crawl_max_age_hours,
            )
            if is_recent:
                reused_names.add(site["name"])
                emit(
                    f"[{site['name']}] reusing crawl completed "
                    f"{age_hours:.1f} hour(s) ago"
                )
            else:
                sites_to_scrape.append(site)

    def scrape(site: dict[str, str]) -> Path:
        removed = clean_generated_artifacts(site_dirs[site["name"]])
        if removed:
            reason = "stale" if mode == "crawl" else "previous"
            emit(
                f"[{site['name']}] removed {reason} generated artifacts: "
                f"{', '.join(removed)}"
            )
        emit(f"[{site['name']}] scraping {site['url']}")
        return scrape_site(
            site_name=site["name"],
            root_url=site["url"],
            evidence_root=EVIDENCE_ROOT,
            log_callback=emit,
            **scraper_config,
        )

    if mode == "crawl":

        def complete_crawl(site: dict[str, str]) -> int:
            if site["name"] not in reused_names:
                scrape(site)
            evidence_file = site_dirs[site["name"]] / "evidence.jsonl"
            chunks = count_chunks(
                evidence_file,
                base_classifier_config["chunk_chars"],
            )
            if chunks == 0:
                raise ValueError(f"No textual evidence found in {evidence_file}")
            if site["name"] not in reused_names:
                write_crawl_state(
                    site=site,
                    site_dir=site_dirs[site["name"]],
                    scraper_config=scraper_config,
                )
            emit(
                f"[{site['name']}] crawl complete: {chunks} Jev request(s) "
                "would be required for full classification."
            )
            return chunks

        chunk_counts, errors = run_site_jobs(
            sites,
            args.workers,
            complete_crawl,
            progress=progress,
            successes_are_terminal=True,
            reused_names=reused_names,
        )
        if progress is not None:
            progress.stop()
        print_bulk_summary(
            selected=len(sites),
            completed=len(chunk_counts),
            errors=errors,
            reused=len(reused_names),
        )
        if errors:
            raise BulkExecutionError(f"{len(errors)} site(s) failed during bulk crawl")
        return

    if run_scraper:
        scraped, scrape_errors = run_site_jobs(
            sites_to_scrape,
            args.workers,
            scrape,
            progress=progress,
            successes_are_terminal=True,
        )
        site_dirs.update(scraped)
        errors.update(scrape_errors)
        if progress is not None:
            progress.stop()

    ready_sites = [site for site in sites if site["name"] not in errors]

    percentage = args.percentage
    if mode == "classify":

        def available_chunks(site: dict[str, str]) -> int:
            evidence_file = site_dirs[site["name"]] / "evidence.jsonl"
            if not evidence_file.exists():
                raise FileNotFoundError(f"Evidence not found: {evidence_file}")
            chunks = count_chunks(
                evidence_file,
                base_classifier_config["chunk_chars"],
            )
            if chunks == 0:
                raise ValueError(f"No textual evidence found in {evidence_file}")
            return chunks

        chunk_counts, count_errors = run_site_jobs(
            ready_sites,
            args.workers,
            available_chunks,
        )
        errors.update(count_errors)
        ready_sites = [site for site in ready_sites if site["name"] in chunk_counts]
        if not ready_sites:
            print_bulk_summary(selected=len(sites), completed=0, errors=errors)
            raise BulkExecutionError("no sites have classifiable evidence")
        total_chunks = sum(chunk_counts.values())
        if percentage is None:
            percentage = prompt_evidence_percentage(total_chunks, len(ready_sites))
        if percentage == 0:
            print("Bulk classification cancelled; no API calls were made.")
            print_bulk_summary(
                selected=len(sites),
                completed=0,
                errors=errors,
                cancelled=len(ready_sites),
            )
            return

    def make_plan(site: dict[str, str]) -> ClassificationPlan:
        classifier_config = dict(base_classifier_config)
        if mode == "classify":
            classifier_config["evidence_percentage"] = percentage
        if args.translation != "off":
            classifier_config.update(
                {
                    "translation_mode": args.translation,
                    "translation_target": args.translation_target,
                    "deepl_api_key": deepl_api_key,
                }
            )
        return prepare_classification_plan(
            site,
            site_dirs[site["name"]],
            classifier_config,
            args.translation,
        )

    prepared, plan_errors = run_site_jobs(ready_sites, args.workers, make_plan)
    errors.update(plan_errors)
    plans = [prepared[site["name"]] for site in ready_sites if site["name"] in prepared]
    if not plans:
        print_bulk_summary(selected=len(sites), completed=0, errors=errors)
        raise BulkExecutionError("no classification plans could be prepared")

    print_bulk_plan(plans, args.translation)
    needs_confirmation = mode == "classify" or any(
        plan.translation_requests for plan in plans
    )
    total_jev_requests = sum(plan.chunks_selected for plan in plans)
    total_deepl_requests = sum(plan.translation_requests for plan in plans)
    total_translation_characters = sum(plan.translation_characters for plan in plans)
    if needs_confirmation and not args.yes:
        approved = prompt_confirmation(
            f"Bulk run will send {total_translation_characters} characters in "
            f"{total_deepl_requests} DeepL request(s), followed by "
            f"{total_jev_requests} Jev request(s)."
        )
        if not approved:
            print("Bulk classification cancelled; no API calls were made.")
            print_bulk_summary(
                selected=len(sites),
                completed=0,
                errors=errors,
                cancelled=len(plans),
            )
            return

    plans_by_name = {plan.site["name"]: plan for plan in plans}

    progress = TerminalProgress(
        len(sites),
        enabled=terminal_progress_enabled(disabled=args.no_progress),
        label="Classifying",
        color="yellow",
    )
    if total_deepl_requests:
        progress.add_stage(
            "deepl",
            label="DeepL",
            total=total_deepl_requests,
            color="magenta",
        )
    progress.add_stage(
        "jev",
        label="Jev",
        total=total_jev_requests,
        color="green",
    )
    progress.start()
    for failed_name in errors:
        progress.record_terminal(failed_name, "failed")

    def classify(site: dict[str, str]) -> dict[str, Any]:
        plan = plans_by_name[site["name"]]
        emit(f"[{site['name']}] classifying saved evidence")
        return classify_site(
            site_name=site["name"],
            evidence_dir=plan.site_dir,
            api_key=api_key,
            progress_callback=progress.advance_stage,
            **plan.classifier_config,
        )

    classifiable_sites = [plan.site for plan in plans]
    results, classification_errors = run_site_jobs(
        classifiable_sites,
        args.workers,
        classify,
        progress=progress,
        successes_are_terminal=True,
    )
    errors.update(classification_errors)
    progress.stop()
    for site in sites:
        result = results.get(site["name"])
        if result is None:
            continue
        if result["evidence_is_partial"]:
            label = (
                "partial smoke-test result" if smoke_test else "partial classification"
            )
            print(
                f"[{site['name']}] {label}: "
                f"{result['provisional_classification']} "
                "(not a final classification)"
            )
        else:
            print(
                f"[{site['name']}] {result['classification']} "
                f"(is_digital_twin={result['is_digital_twin']})"
            )

    print_bulk_summary(
        selected=len(sites),
        completed=len(results),
        errors=errors,
    )
    if errors:
        raise BulkExecutionError(f"{len(errors)} site(s) failed during bulk execution")


if __name__ == "__main__":
    try:
        main()
    except BulkExecutionError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
