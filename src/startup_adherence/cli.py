"""Batch command line interface for profile-aware website evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from .adapters.jev import JevClient
from .application.classification import (
    classify_saved,
    current_result,
    plan_classification,
    replay,
)
from .application.crawl import scrape_site
from .application.crawl_state import recent_crawl, recover, write_state
from .application.reporting import export_csv, print_overview, rows_for_results
from .domain.profile import load_profile, validate_profile
from .domain.urls import (
    DEFAULT_MAX_CRAWL_SECONDS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_SITEMAPS,
    canonicalize,
    evidence_directory_name,
    host,
)

DEFAULT_SITES = [{"name": "madidt", "url": "https://madidt.com/"}]
JEV_API_KEY_ENV = "TYPESAFE_PSN_DIG_TWIN_CLASS"
DEEPL_API_KEY_ENV = "DEEPL_API_KEY"
T = TypeVar("T")


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not 0 <= result < float("inf"):
        raise argparse.ArgumentTypeError("Must be finite and nonnegative")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile-based startup website evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "crawl", "classify", "score", "overview", "export"),
        default="smoke",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Versioned theme profile JSON; default digital twin",
    )
    parser.add_argument("--sites-file", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--site", action="append", dest="sites_one")
    selection.add_argument("--sites", nargs="+", dest="sites_many")
    parser.add_argument("--evidence-root", type=Path, default=Path("evidence"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-pages-per-site", type=nonnegative_int, default=DEFAULT_MAX_PAGES
    )
    parser.add_argument(
        "--max-requests-per-site", type=nonnegative_int, default=DEFAULT_MAX_REQUESTS
    )
    parser.add_argument(
        "--max-queue-size", type=nonnegative_int, default=DEFAULT_MAX_QUEUE_SIZE
    )
    parser.add_argument(
        "--max-crawl-seconds", type=nonnegative_float, default=DEFAULT_MAX_CRAWL_SECONDS
    )
    parser.add_argument(
        "--max-sitemaps-per-site", type=nonnegative_int, default=DEFAULT_MAX_SITEMAPS
    )
    parser.add_argument("--crawl-max-age-hours", type=nonnegative_float, default=24.0)
    parser.add_argument("--force-crawl", action="store_true")
    parser.add_argument("--recover-existing-crawls", action="store_true")
    parser.add_argument("--percentage", type=nonnegative_float)
    parser.add_argument(
        "--translation", choices=("off", "auto", "deepl"), default="off"
    )
    parser.add_argument("--translation-target", default="EN")
    parser.add_argument("--api-key-env", default=JEV_API_KEY_ENV)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.percentage is not None and (
        args.mode != "classify" or args.percentage > 100
    ):
        parser.error("--percentage requires classify mode and a value from 0 to 100")
    if args.force_crawl and args.mode != "crawl":
        parser.error("--force-crawl requires crawl mode")
    if args.recover_existing_crawls and (args.mode != "crawl" or args.force_crawl):
        parser.error(
            "--recover-existing-crawls requires crawl mode without --force-crawl"
        )
    if args.output is not None and args.mode != "export":
        parser.error("--output requires export mode")
    return args


def select_sites(
    configured: list[dict[str, str]], names: list[str] | None
) -> list[dict[str, str]]:
    if not isinstance(configured, list):
        raise TypeError("Sites file must contain a JSON list")
    unique: list[dict[str, str]] = []
    by_name: dict[str, dict[str, str]] = {}
    by_url: dict[tuple[str, int | None, str], dict[str, str]] = {}
    by_directory: dict[str, dict[str, str]] = {}
    skipped = 0
    for site in configured:
        if (
            not isinstance(site, dict)
            or not isinstance(site.get("name"), str)
            or not isinstance(site.get("url"), str)
            or not site["name"]
            or not site["url"]
        ):
            raise ValueError("Each site needs name and URL")
        url = canonicalize(site["url"], keep_query=False)
        if not url:
            raise ValueError(f"Invalid site URL for {site['name']}: {site['url']}")
        parsed = urlparse(url)
        # Crawls ignore query strings; www, scheme and trailing slash variants
        # refer to the same site input for this batch.
        url_key = (host(url), parsed.port, parsed.path.rstrip("/") or "/")
        directory = evidence_directory_name(site["name"]).casefold()
        original = by_name.get(site["name"]) or by_url.get(url_key)
        if original is not None:
            by_name[site["name"]] = original
            skipped += 1
            continue
        if directory in by_directory:
            raise ValueError("Site names collide on disk")
        by_name[site["name"]] = site
        by_url[url_key] = site
        by_directory[directory] = site
        unique.append(site)
    if skipped:
        print(
            f"Skipped {skipped} duplicate site entries (first occurrence kept).",
            file=sys.stderr,
        )
    if not names or names == ["all"]:
        return unique
    unknown = set(names) - by_name.keys()
    if unknown:
        raise ValueError(f"Unknown sites: {', '.join(sorted(unknown))}")
    selected = {by_name[name]["name"] for name in names}
    return [site for site in unique if site["name"] in selected]


def run_jobs(
    sites: list[dict[str, str]], workers: int, operation: Callable[[dict[str, str]], T]
) -> tuple[dict[str, T], dict[str, str]]:
    """Preserve unrelated site results if one site fails."""
    results: dict[str, T] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(len(sites), 1))) as executor:
        futures = {executor.submit(operation, site): site for site in sites}
        for future in as_completed(futures):
            site = futures[future]
            try:
                results[site["name"]] = future.result()
            except Exception as exc:  # noqa: BLE001 - each site must fail independently
                errors[site["name"]] = f"{type(exc).__name__}: {exc}"
    return results, errors


def approve(message: str) -> bool:
    print(f"{message} Continue? [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().casefold() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = (
        load_profile(args.profile)
        if args.profile
        else json.loads(
            files("startup_adherence.profiles")
            .joinpath("digital_twin.json")
            .read_text(encoding="utf-8")
        )
    )
    validate_profile(profile)
    configured = (
        json.loads(args.sites_file.read_text(encoding="utf-8"))
        if args.sites_file
        else DEFAULT_SITES
    )
    sites = select_sites(configured, args.sites_many or args.sites_one)
    if not sites:
        raise ValueError("No sites selected")
    directories = {
        site["name"]: args.evidence_root / evidence_directory_name(site["name"])
        for site in sites
    }
    crawl_config = {
        "max_pages": 5 if args.mode == "smoke" else args.max_pages_per_site,
        "max_requests": args.max_requests_per_site,
        "max_queue_size": args.max_queue_size,
        "max_crawl_seconds": args.max_crawl_seconds,
        "max_sitemaps": args.max_sitemaps_per_site,
        "include_subdomains": False,
        "include_query_urls": False,
        "respect_robots": True,
        "request_timeout": 20,
    }
    if args.mode in {"score", "overview", "export"}:
        print(
            f"{args.mode.upper()}: local saved responses only; no API calls will be made."
        )
        action = (
            (
                lambda site: replay(
                    site_name=site["name"],
                    site_dir=directories[site["name"]],
                    profile=profile,
                )
            )
            if args.mode == "score"
            else (lambda site: current_result(directories[site["name"]], profile))
        )
        results, errors = run_jobs(sites, args.workers, action)
        rows = rows_for_results(sites, results, profile)
        if rows:
            print_overview(rows, profile)
        if args.mode == "export" and not errors:
            destination = args.output or Path(f"{profile['id']}_scores.csv")
            count = export_csv(rows, profile, destination)
            print(f"CSV EXPORT COMPLETE: rows={count}, output={destination}")
        return finish_summary(len(sites), len(results), errors)

    if args.recover_existing_crawls:
        print("CRAWL RECOVERY: local artifacts only; no network or API calls.")
        results, errors = run_jobs(
            sites,
            args.workers,
            lambda site: recover(site, directories[site["name"]], crawl_config, 20_000),
        )
        for name, value in results.items():
            print(f"[{name}] {value}")
        return finish_summary(len(sites), len(results), errors)

    if args.mode in {"smoke", "crawl"}:
        print("CRAWL: website requests; no Jev or DeepL calls during this stage.")

        def do_crawl(site: dict[str, str]) -> Path:
            directory = directories[site["name"]]
            if (
                args.mode == "crawl"
                and not args.force_crawl
                and recent_crawl(
                    site, directory, crawl_config, args.crawl_max_age_hours
                )
            ):
                print(f"[{site['name']}] reusing verified crawl")
                return directory
            result = scrape_site(
                site_name=site["name"],
                root_url=site["url"],
                evidence_root=args.evidence_root,
                verbose=args.verbose,
                **crawl_config,
            )
            write_state(site, result, crawl_config)
            return result

        _, crawl_errors = run_jobs(sites, args.workers, do_crawl)
        if args.mode == "crawl":
            return finish_summary(
                len(sites), len(sites) - len(crawl_errors), crawl_errors
            )
        sites = [site for site in sites if site["name"] not in crawl_errors]
        if not sites:
            return finish_summary(len(crawl_errors), 0, crawl_errors)
    else:
        crawl_errors = {}

    if args.mode == "classify" and args.percentage == 0:
        print("Bulk classification cancelled; no API calls were made.")
        return finish_summary(len(sites), 0, crawl_errors)

    percentage = args.percentage
    if args.mode == "classify" and percentage is None:
        print("Percentage of saved evidence to classify [0-100]: ", end="", flush=True)
        percentage = nonnegative_float(sys.stdin.readline().strip())
        if percentage > 100:
            raise ValueError("Percentage must be at most 100")
        if percentage == 0:
            print("Bulk classification cancelled; no API calls were made.")
            return finish_summary(len(sites), 0, crawl_errors)

    plans, plan_errors = run_jobs(
        sites,
        args.workers,
        lambda site: plan_classification(
            directories[site["name"]] / "evidence.jsonl",
            percentage=percentage if args.mode == "classify" else None,
            max_chunks=1 if args.mode == "smoke" else 0,
            translation=args.translation,
        ),
    )
    total_jev = sum(plan.jev_requests for plan in plans.values())
    total_deepl = sum(plan.deepl_requests for plan in plans.values())
    total_chars = sum(plan.deepl_characters for plan in plans.values())
    print(
        f"BULK PLAN: sites={len(plans)}, Jev requests={total_jev}, DeepL requests={total_deepl}, DeepL characters={total_chars}"
    )
    if not plans:
        return finish_summary(len(sites), 0, {**crawl_errors, **plan_errors})
    if not args.yes and not approve("Paid API calls are planned."):
        print("Cancelled; no paid API calls were made.")
        return finish_summary(len(sites), 0, {**crawl_errors, **plan_errors})
    api_key = os.getenv(args.api_key_env, "")
    deepl_key = os.getenv(DEEPL_API_KEY_ENV, "")
    if not api_key or (total_deepl and not deepl_key):
        raise ValueError("Required Jev or DeepL API key is missing")

    def classify(site: dict[str, str]) -> dict[str, Any]:
        result = classify_saved(
            site_name=site["name"],
            site_dir=directories[site["name"]],
            profile=profile,
            plan=plans[site["name"]],
            jev_client=JevClient(api_key),
            model=args.model,
            translation=args.translation,
            target_language=args.translation_target,
            deepl_key=deepl_key,
            progress=(
                None
                if args.no_progress
                else lambda service: print(
                    f"[{site['name']}] {service} completed", flush=True
                )
            ),
        )
        print(
            f"[{site['name']}] fit={result['fit_score']:.2f}/100; gap={result['main_gap']}; partial={result['evidence_is_partial']}"
        )
        return result

    ready = [site for site in sites if site["name"] in plans]
    results, errors = run_jobs(ready, args.workers, classify)
    if results:
        print_overview(rows_for_results(ready, results, profile), profile)
    return finish_summary(
        len(sites) + len(crawl_errors),
        len(results),
        {**crawl_errors, **plan_errors, **errors},
    )


def finish_summary(selected: int, completed: int, errors: dict[str, str]) -> int:
    print(
        f"BULK SUMMARY: selected={selected}, completed={completed}, failed={len(errors)}"
    )
    for name, error in errors.items():
        print(f"[{name}] failed: {error}")
    return 1 if errors else 0
