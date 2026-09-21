"""Configuration and entry point for scraping and Jev classification."""

from __future__ import annotations

import os
from pathlib import Path

from classifier import classify_site
from scraper import scrape_site

# Add or remove sites here. Each name becomes a dedicated evidence/<name>/ folder.
SITES = [
    {"name": "netabolics", "url": "https://netabolics.ai/"},
]

EVIDENCE_ROOT = Path("evidence")

# Keep secrets outside source control. classifier.py receives this value in memory.
JEV_API_KEY = os.getenv("TYPESAFE_API_KEY", "")

RUN_SCRAPER = True
RUN_CLASSIFIER = True

SCRAPER_CONFIG = {
    "max_pages": 0,  # 0 = no page-count limit
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


def main() -> None:
    if RUN_CLASSIFIER and not JEV_API_KEY:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Set it in the environment before running."
        )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        name = site["name"]
        url = site["url"]
        site_dir = EVIDENCE_ROOT / name

        if RUN_SCRAPER:
            print(f"[{name}] scraping {url}")
            site_dir = scrape_site(
                site_name=name,
                root_url=url,
                evidence_root=EVIDENCE_ROOT,
                **SCRAPER_CONFIG,
            )

        if RUN_CLASSIFIER:
            print(f"[{name}] classifying saved evidence")
            result = classify_site(
                site_name=name,
                evidence_dir=site_dir,
                api_key=JEV_API_KEY,
                **CLASSIFIER_CONFIG,
            )
            print(
                f"[{name}] {result['classification']} "
                f"(is_digital_twin={result['is_digital_twin']})"
            )


if __name__ == "__main__":
    main()
