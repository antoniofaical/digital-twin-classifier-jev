"""Configuration and entry point for scraping and Jev classification."""

from __future__ import annotations

import argparse
import os
from math import ceil
from pathlib import Path
from typing import Any

from classifier import (
    classify_site,
    evidence_chunks,
    is_english_hint,
    select_evidence_chunks,
)
from scraper import scrape_site

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
    return parser.parse_args(argv)


def prompt_evidence_percentage(chunk_count: int) -> float:
    while True:
        raw = input(
            f"{chunk_count} chunks of evidence generated. "
            "How much do you want to include (%)? "
        )
        normalized = raw.strip().removesuffix("%").strip().replace(",", ".")
        try:
            percentage = float(normalized)
        except ValueError:
            print("Enter a number from 0 to 100.")
            continue
        if 0 <= percentage <= 100:
            return percentage
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = args.mode
    run_scraper = mode in {"smoke", "crawl"}
    run_classifier = mode in {"smoke", "classify"}
    smoke_test = mode == "smoke"

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
        print("CLASSIFICATION: no API calls occur before evidence selection.")

    for site in SITES:
        name = site["name"]
        url = site["url"]
        site_dir = EVIDENCE_ROOT / name
        classifier_config = dict(base_classifier_config)

        if run_scraper:
            print(f"[{name}] scraping {url}")
            site_dir = scrape_site(
                site_name=name,
                root_url=url,
                evidence_root=EVIDENCE_ROOT,
                **scraper_config,
            )

        evidence_file = site_dir / "evidence.jsonl"

        if mode == "crawl":
            chunk_count = count_chunks(
                evidence_file,
                classifier_config["chunk_chars"],
            )
            print(
                f"[{name}] crawl complete: {chunk_count} Jev request(s) "
                "would be required for full classification."
            )

        if mode == "classify":
            chunk_count = count_chunks(
                evidence_file,
                classifier_config["chunk_chars"],
            )
            percentage = prompt_evidence_percentage(chunk_count)
            if percentage == 0:
                print(f"[{name}] classification cancelled; no API calls were made.")
                continue
            selected_count = min(
                chunk_count,
                ceil(chunk_count * percentage / 100),
            )
            print(
                f"[{name}] {selected_count} of {chunk_count} chunks selected; "
                f"{selected_count} Jev request(s) will be made."
            )
            classifier_config["evidence_percentage"] = percentage

        if run_classifier and args.translation != "off":
            chunks, _, _, _ = select_evidence_chunks(
                evidence_file,
                classifier_config["chunk_chars"],
                max_chunks=classifier_config["max_chunks"],
                evidence_percentage=classifier_config.get("evidence_percentage"),
            )
            translation_requests, translation_characters = translation_workload(
                chunks,
                args.translation,
            )
            if translation_requests:
                approved = prompt_confirmation(
                    f"[{name}] translation will send {translation_characters} "
                    f"characters in {translation_requests} DeepL request(s), "
                    f"followed by {len(chunks)} Jev request(s)."
                )
                if not approved:
                    print(f"[{name}] classification cancelled; no API calls were made.")
                    continue
            else:
                print(
                    f"[{name}] all selected fragments are marked as English; "
                    "no DeepL requests are needed."
                )
            classifier_config.update(
                {
                    "translation_mode": args.translation,
                    "translation_target": args.translation_target,
                    "deepl_api_key": deepl_api_key,
                }
            )

        if run_classifier:
            print(f"[{name}] classifying saved evidence")
            result = classify_site(
                site_name=name,
                evidence_dir=site_dir,
                api_key=api_key,
                **classifier_config,
            )
            if result["evidence_is_partial"]:
                label = (
                    "partial smoke-test result"
                    if smoke_test
                    else "partial classification"
                )
                print(
                    f"[{name}] {label}: "
                    f"{result['provisional_classification']} "
                    "(not a final classification)"
                )
            else:
                print(
                    f"[{name}] {result['classification']} "
                    f"(is_digital_twin={result['is_digital_twin']})"
                )


if __name__ == "__main__":
    main()
