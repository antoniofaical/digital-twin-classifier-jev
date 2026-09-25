"""Callable site operations used by the batch CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .classification import ClassificationPlan, classify_saved
from .crawl import scrape_site
from .crawl_state import recent_crawl, write_state


@dataclass
class CrawlJob:
    evidence_root: Path
    directories: Mapping[str, Path]
    config: Mapping[str, Any]
    reuse: bool
    max_age_hours: float
    verbose: int
    scraper: Callable[..., Path] = scrape_site
    state_writer: Callable[..., None] = write_state
    message: Callable[[str], None] | None = None

    def __call__(self, site: dict[str, str]) -> Path:
        name = site["name"]
        directory = self.directories[name]
        if self.message:
            self.message(f"[{name}] crawl START {site['url']}")
        if self.reuse and recent_crawl(
            site, directory, dict(self.config), self.max_age_hours
        ):
            if self.message:
                self.message(f"[{name}] reusing verified crawl")
            return directory
        result = self.scraper(
            site_name=name,
            root_url=site["url"],
            evidence_root=self.evidence_root,
            verbose=self.verbose,
            log_callback=self.message,
            **self.config,
        )
        self.state_writer(site, result, dict(self.config))
        return result


@dataclass
class ClassificationJob:
    directories: Mapping[str, Path]
    plans: Mapping[str, ClassificationPlan]
    profile: dict[str, Any]
    client_factory: Callable[..., Any]
    api_key: str
    model: str
    translation: str
    target_language: str
    deepl_key: str
    progress: Callable[[str, str], None] | None = None
    message: Callable[[str], None] | None = None

    def __call__(self, site: dict[str, str]) -> dict[str, Any]:
        name = site["name"]
        result = classify_saved(
            site_name=name,
            site_dir=self.directories[name],
            profile=self.profile,
            plan=self.plans[name],
            jev_client=self.client_factory(self.api_key),
            model=self.model,
            translation=self.translation,
            target_language=self.target_language,
            deepl_key=self.deepl_key,
            progress=(partial(self.progress, name) if self.progress else None),
        )
        if self.message:
            self.message(
                f"[{name}] fit={result['fit_score']:.2f}/100; "
                f"gap={result['main_gap']}; partial={result['evidence_is_partial']}"
            )
        return result
