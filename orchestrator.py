"""Configuration and entry point for scraping and Jev classification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from classifier import classify_site, evidence_chunks
from scraper import scrape_site

# Add or remove sites here. Each name becomes a dedicated evidence/<name>/ folder.
SITES = [
    {"name": "netabolics", "url": "https://netabolics.ai/"},
]

EVIDENCE_ROOT = Path("evidence")

# Keep secrets outside source control. classifier.py receives this value in memory.
JEV_API_KEY_ENV = "TYPESAFE_PSN_DIG_TWIN_CLASS"

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
            "crawl: full crawl with no Jev calls; "
            "classify: classify all previously saved evidence"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    mode = parse_args(argv).mode
    run_scraper = mode in {"smoke", "crawl"}
    run_classifier = mode in {"smoke", "classify"}
    smoke_test = mode == "smoke"

    scraper_config = {
        **SCRAPER_CONFIG,
        "max_pages": SMOKE_MAX_PAGES if smoke_test else 0,
    }
    classifier_config = {
        **CLASSIFIER_CONFIG,
        "max_chunks": SMOKE_MAX_CHUNKS if smoke_test else 0,
    }

    api_key = os.getenv(JEV_API_KEY_ENV, "")
    if run_classifier and not api_key:
        raise RuntimeError(
            f"{JEV_API_KEY_ENV} is not set. Set it in the environment before running."
        )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        print(
            f"SMOKE TEST: at most {SMOKE_MAX_PAGES} pages and "
            f"{SMOKE_MAX_CHUNKS} Jev request per site."
        )
    elif mode == "crawl":
        print("FULL CRAWL: no Jev API calls will be made.")
    else:
        print("FULL CLASSIFICATION: using all previously saved evidence.")

    for site in SITES:
        name = site["name"]
        url = site["url"]
        site_dir = EVIDENCE_ROOT / name

        if run_scraper:
            print(f"[{name}] scraping {url}")
            site_dir = scrape_site(
                site_name=name,
                root_url=url,
                evidence_root=EVIDENCE_ROOT,
                **scraper_config,
            )

        if mode == "crawl":
            evidence_file = site_dir / "evidence.jsonl"
            chunk_count = sum(
                1
                for _ in evidence_chunks(
                    evidence_file,
                    classifier_config["chunk_chars"],
                )
            )
            print(
                f"[{name}] crawl complete: {chunk_count} Jev request(s) "
                "would be required for full classification."
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
                print(
                    f"[{name}] partial smoke-test result: "
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
