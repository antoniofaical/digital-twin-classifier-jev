"""Bounded website discovery and crawl persistence."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from uuid import uuid4

import requests
from bs4 import ParserRejectedMarkup

from ..adapters.extract import (
    PdfReadError,
    extract_response,
    response_language_hint,
    sitemap_locations,
)
from ..domain.urls import (
    DEFAULT_MAX_CRAWL_SECONDS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_QUEUE_SIZE,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_SITEMAPS,
    SKIPPED_EXTENSIONS,
    USER_AGENT,
    canonicalize,
    evidence_directory_name,
    in_scope,
    unsafe_url_reason,
)
from ..storage import append_jsonl, write_json


def scrape_site(
    *,
    site_name: str,
    root_url: str,
    evidence_root: Path,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    max_crawl_seconds: float = DEFAULT_MAX_CRAWL_SECONDS,
    max_sitemaps: int = DEFAULT_MAX_SITEMAPS,
    include_subdomains: bool = False,
    include_query_urls: bool = False,
    respect_robots: bool = True,
    request_timeout: int = 20,
    verbose: int = 0,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """Crawl one site and create evidence/<site_name>/evidence.jsonl."""
    limits = {
        "max_pages": max_pages,
        "max_requests": max_requests,
        "max_queue_size": max_queue_size,
        "max_crawl_seconds": max_crawl_seconds,
        "max_sitemaps": max_sitemaps,
    }
    if any(value < 0 for value in limits.values()):
        raise ValueError("crawl limits must be zero or positive")

    started_at = time.monotonic()
    root_url = root_url if re.match(r"^https?://", root_url) else "https://" + root_url
    root_url = canonicalize(root_url, keep_query=include_query_urls)
    if not root_url:
        raise ValueError("Site URL must use HTTP(S) and include a host")
    site_dir = evidence_root / evidence_directory_name(site_name)
    site_dir.mkdir(parents=True, exist_ok=True)
    stage = site_dir / "crawl_attempts" / uuid4().hex
    stage.mkdir(parents=True)
    staged_evidence = stage / "evidence.jsonl"
    staged_evidence.touch()

    def log(level: int, message: str) -> None:
        if verbose >= level:
            rendered = f"[{site_name}] {message}"
            if log_callback is None:
                print(rendered, flush=True)
            else:
                log_callback(rendered)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots = RobotFileParser()
    robots_url = urljoin(root_url, "/robots.txt")
    robots.set_url(robots_url)
    errors: list[dict[str, str]] = []
    page_errors: list[dict[str, str]] = []
    limit_reasons: set[str] = set()
    skipped_by_reason: Counter[str] = Counter()
    sitemap_seeds = {
        urljoin(root_url, "/sitemap.xml"),
        urljoin(root_url, "/sitemap_index.xml"),
        urljoin(root_url, "/wp-sitemap.xml"),
    }

    try:
        log(1, f"robots GET {robots_url}")
        response = session.get(robots_url, timeout=request_timeout)
        log(2, f"robots HTTP {response.status_code} {response.url}")
        if response.ok:
            robots.parse(response.text.splitlines())
            sitemap_seeds.update(
                line.split(":", 1)[1].strip()
                for line in response.text.splitlines()
                if line.lower().startswith("sitemap:")
            )
        else:
            robots.parse([])
    except requests.RequestException as exc:
        robots.parse([])
        errors.append({"url": robots_url, "error": str(exc)})
        log(1, f"robots ERROR {robots_url}: {exc}")

    queue: deque[str] = deque()
    queued: set[str] = set()
    visited: set[str] = set()
    sitemap_seen: set[str] = set()
    content_hashes: dict[str, str] = {}

    def time_limit_reached() -> bool:
        if not max_crawl_seconds:
            return False
        if time.monotonic() - started_at < max_crawl_seconds:
            return False
        limit_reasons.add("time_limit")
        return True

    def enqueue(url: str) -> None:
        value = canonicalize(url, keep_query=include_query_urls)
        if not value or not in_scope(value, root_url, include_subdomains):
            return
        if Path(urlparse(value).path.lower()).suffix in SKIPPED_EXTENSIONS:
            return
        unsafe_reason = unsafe_url_reason(value)
        if unsafe_reason:
            skipped_by_reason[unsafe_reason] += 1
            log(2, f"trap SKIP {value}; reason={unsafe_reason}")
            return
        if value in queued or value in visited:
            return
        if max_queue_size and len(queue) >= max_queue_size:
            limit_reasons.add("queue_limit")
            skipped_by_reason["queue_limit"] += 1
            log(2, f"queue SKIP {value}; limit={max_queue_size}")
            return
        queued.add(value)
        queue.append(value)

    def read_sitemap(url: str) -> None:
        value = canonicalize(url, keep_query=True)
        if (
            not value
            or value in sitemap_seen
            or not in_scope(value, root_url, include_subdomains)
        ):
            return
        if time_limit_reached():
            return
        if max_sitemaps and len(sitemap_seen) >= max_sitemaps:
            limit_reasons.add("sitemap_limit")
            skipped_by_reason["sitemap_limit"] += 1
            log(2, f"sitemap SKIP {value}; limit={max_sitemaps}")
            return
        sitemap_seen.add(value)
        try:
            log(1, f"sitemap GET {value}")
            response = session.get(value, timeout=request_timeout)
            response.raise_for_status()
            if not in_scope(response.url, root_url, include_subdomains):
                errors.append(
                    {"url": value, "error": "sitemap redirected outside scope"}
                )
                return
            kind, locations = sitemap_locations(response.content, response.url)
            log(
                2,
                f"sitemap HTTP {response.status_code} {response.url}; "
                f"kind={kind}, locations={len(locations)}",
            )
            if kind == "sitemapindex":
                for location in locations:
                    read_sitemap(location)
            else:
                for location in locations:
                    enqueue(location)
        except requests.RequestException as exc:
            errors.append({"url": value, "error": str(exc)})
            log(1, f"sitemap ERROR {value}: {exc}")

    for sitemap in sitemap_seeds:
        read_sitemap(sitemap)
    enqueue(root_url)

    pages_saved = 0
    page_requests = 0
    while queue:
        if max_pages and pages_saved >= max_pages:
            limit_reasons.add("page_limit")
            break
        if max_requests and page_requests >= max_requests:
            limit_reasons.add("request_limit")
            break
        if time_limit_reached():
            break
        url = queue.popleft()
        queued.discard(url)
        if url in visited:
            continue
        visited.add(url)
        if respect_robots and not robots.can_fetch(USER_AGENT, url):
            log(2, f"robots SKIP {url}")
            continue
        try:
            page_requests += 1
            log(1, f"page {page_requests} GET {url}")
            response = session.get(url, timeout=request_timeout)
            response.raise_for_status()
            if not in_scope(response.url, root_url, include_subdomains):
                issue = {"url": url, "error": "page redirected outside scope"}
                errors.append(issue)
                page_errors.append(issue)
                continue
            text, links = extract_response(response)
            final_url = canonicalize(
                response.url,
                keep_query=include_query_urls,
            )
            if final_url and in_scope(final_url, root_url, include_subdomains):
                visited.add(final_url)
            log(
                2,
                f"page HTTP {response.status_code} {response.url}; "
                f"type={response.headers.get('content-type', '')!r}, "
                f"characters={len(text)}, links={len(links)}",
            )
            content_hash = (
                hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
            )
            if content_hash and content_hash in content_hashes:
                skipped_by_reason["duplicate_content"] += 1
                log(
                    2,
                    f"content SKIP {response.url}; duplicate of "
                    f"{content_hashes[content_hash]}",
                )
                continue
            if content_hash:
                content_hashes[content_hash] = final_url or response.url
            append_jsonl(
                staged_evidence,
                {
                    "url": final_url or response.url,
                    "content_type": response.headers.get("content-type", ""),
                    "language_hint": response_language_hint(response),
                    "text": text,
                },
            )
            pages_saved += 1
            for link in links:
                enqueue(link)
            log(2, f"queue size={len(queue)} after {response.url}")
        except (
            requests.RequestException,
            OSError,
            PdfReadError,
            ParserRejectedMarkup,
        ) as exc:
            issue = {"url": url, "error": str(exc)}
            errors.append(issue)
            page_errors.append(issue)
            log(1, f"page ERROR {url}: {exc}")

    manifest = {
        "site_name": site_name,
        "root_url": root_url,
        "created_at": datetime.now(UTC).isoformat(),
        "pages_saved": pages_saved,
        "urls_visited": len(visited),
        "page_requests_attempted": page_requests,
        "sitemaps_checked": len(sitemap_seen),
        "remaining_queue": len(queue),
        "crawl_complete": not limit_reasons and not page_errors,
        "crawl_limited": bool(limit_reasons),
        "crawl_limit_reasons": sorted(limit_reasons),
        "stopped_by_page_limit": "page_limit" in limit_reasons,
        "stopped_by_request_limit": "request_limit" in limit_reasons,
        "stopped_by_time_limit": "time_limit" in limit_reasons,
        "limits": limits,
        "urls_skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "duration_seconds": round(time.monotonic() - started_at, 3),
        "errors": errors,
        "page_errors": page_errors,
    }
    write_json(stage / "manifest.json", manifest)
    if pages_saved == 0:
        raise ValueError(f"No pages were saved; crawl attempt retained at {stage}")
    staged_evidence.replace(site_dir / "evidence.jsonl")
    (stage / "manifest.json").replace(site_dir / "manifest.json")
    stage.rmdir()
    log(
        1,
        f"saved {pages_saved} page(s); visited={len(visited)}, "
        f"requests={page_requests}, errors={len(errors)}, remaining={len(queue)}, "
        f"limited={','.join(sorted(limit_reasons)) or 'no'}",
    )
    return site_dir
