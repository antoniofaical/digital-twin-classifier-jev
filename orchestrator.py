"""Configuration and entry point for scraping and Jev classification."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from classifier import (
    classify_site,
    evidence_chunks,
    is_english_hint,
    select_evidence_chunks,
)
from scraper import evidence_directory_name, scrape_site

SITES = [
    {"name": "madidt", "url": "https://madidt.com/"},
]

EVIDENCE_ROOT = Path("evidence")
JEV_API_KEY_ENV = "TYPESAFE_PSN_DIG_TWIN_CLASS"
DEEPL_API_KEY_ENV = "DEEPL_API_KEY"

SMOKE_MAX_PAGES = 5
SMOKE_MAX_CHUNKS = 1

SCRAPER_CONFIG = {
    "include_subdomains": False,
    "include_query_urls": False,
    "respect_robots": True,
    "request_timeout": 20,
}

CLASSIFIER_CONFIG = {
    "model": "jev-latest",
    "chunk_chars": 20_000,
    "positive_threshold": 0.70,
    "negative_threshold": 0.30,
    "request_timeout": 60,
}

T = TypeVar("T")


class BulkExecutionError(RuntimeError):
    """Raised after all possible sites finish when one or more sites failed."""


@dataclass(frozen=True)
class ClassificationPlan:
    site: dict[str, str]
    site_dir: Path
    classifier_config: dict[str, Any]
    chunks_available: int
    chunks_selected: int
    translation_requests: int
    translation_characters: int


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


def run_site_jobs(
    sites: list[dict[str, str]],
    workers: int,
    operation: Callable[[dict[str, str]], T],
) -> tuple[dict[str, T], dict[str, str]]:
    """Run isolated site jobs and collect every result before reporting failures."""
    results: dict[str, T] = {}
    errors: dict[str, str] = {}
    if workers == 1 or len(sites) <= 1:
        for site in sites:
            try:
                results[site["name"]] = operation(site)
            except Exception as exc:  # noqa: BLE001 - isolate failures in a bulk run
                errors[site["name"]] = f"{type(exc).__name__}: {exc}"
        return results, errors

    with ThreadPoolExecutor(max_workers=min(workers, len(sites))) as executor:
        futures = {executor.submit(operation, site): site for site in sites}
        for future in as_completed(futures):
            site = futures[future]
            try:
                results[site["name"]] = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate failures in a bulk run
                errors[site["name"]] = f"{type(exc).__name__}: {exc}"
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
    *, selected: int, completed: int, errors: dict[str, str], cancelled: int = 0
) -> None:
    print(
        "BULK SUMMARY: "
        f"selected={selected}, completed={completed}, "
        f"failed={len(errors)}, cancelled={cancelled}"
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
        "max_pages": SMOKE_MAX_PAGES if smoke_test else 0,
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

    if smoke_test:
        print(
            f"SMOKE TEST: at most {SMOKE_MAX_PAGES} pages and "
            f"{SMOKE_MAX_CHUNKS} Jev request per site."
        )
    elif mode == "crawl":
        print("FULL CRAWL: no Jev or DeepL API calls will be made.")
    else:
        print("CLASSIFICATION: no API calls occur before bulk approval.")
    print(f"Selected {len(sites)} site(s); using up to {args.workers} worker(s).")

    errors: dict[str, str] = {}

    if run_scraper:

        def scrape(site: dict[str, str]) -> Path:
            print(f"[{site['name']}] scraping {site['url']}")
            return scrape_site(
                site_name=site["name"],
                root_url=site["url"],
                evidence_root=EVIDENCE_ROOT,
                **scraper_config,
            )

        scraped, scrape_errors = run_site_jobs(sites, args.workers, scrape)
        site_dirs.update(scraped)
        errors.update(scrape_errors)

    ready_sites = [site for site in sites if site["name"] not in errors]
    if mode == "crawl":

        def crawl_result(site: dict[str, str]) -> int:
            return count_chunks(
                site_dirs[site["name"]] / "evidence.jsonl",
                base_classifier_config["chunk_chars"],
            )

        chunk_counts, count_errors = run_site_jobs(
            ready_sites,
            args.workers,
            crawl_result,
        )
        errors.update(count_errors)
        for site in sites:
            if site["name"] in chunk_counts:
                print(
                    f"[{site['name']}] crawl complete: "
                    f"{chunk_counts[site['name']]} Jev request(s) would be "
                    "required for full classification."
                )
        print_bulk_summary(
            selected=len(sites),
            completed=len(chunk_counts),
            errors=errors,
        )
        if errors:
            raise BulkExecutionError(f"{len(errors)} site(s) failed during bulk crawl")
        return

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

    def classify(site: dict[str, str]) -> dict[str, Any]:
        plan = plans_by_name[site["name"]]
        print(f"[{site['name']}] classifying saved evidence")
        return classify_site(
            site_name=site["name"],
            evidence_dir=plan.site_dir,
            api_key=api_key,
            **plan.classifier_config,
        )

    classifiable_sites = [plan.site for plan in plans]
    results, classification_errors = run_site_jobs(
        classifiable_sites,
        args.workers,
        classify,
    )
    errors.update(classification_errors)
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
