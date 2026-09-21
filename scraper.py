"""Discover a public website and save its textual evidence."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    unquote,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, ParserRejectedMarkup

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:
    PdfReader = None
    PdfReadError = ValueError


USER_AGENT = "DigitalTwinClassifier/1.0"
DEFAULT_MAX_PAGES = 400
DEFAULT_MAX_REQUESTS = 500
DEFAULT_MAX_QUEUE_SIZE = 2_000
DEFAULT_MAX_CRAWL_SECONDS = 900.0
DEFAULT_MAX_SITEMAPS = 50
MAX_URL_LENGTH = 2_048
MAX_PATH_SEGMENTS = 30
REPEATED_PATH_SEQUENCE_LIMIT = 3
SKIPPED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".eot",
    ".exe",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".map",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".png",
    ".rar",
    ".svg",
    ".tar",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
}
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
PAGE_FILE_EXTENSIONS = {".asp", ".aspx", ".htm", ".html", ".jsp", ".php"}
DOCUMENT_FILE_EXTENSIONS = {".csv", ".json", ".md", ".pdf", ".txt", ".xml"}
SCHEMELESS_HOST_RE = re.compile(
    r"^(?P<host>(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})\.)+"
    r"[A-Za-z]{2,63}(?::\d+)?)(?P<suffix>(?:[/?#].*)?)$"
)


def evidence_directory_name(site_name: str) -> str:
    """Return a filesystem-safe directory name while preserving the display name."""
    ascii_name = (
        unicodedata.normalize("NFKD", site_name).encode("ascii", "ignore").decode()
    )
    directory_name = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_name).strip(".-_")
    if not directory_name:
        raise ValueError("site_name must contain at least one letter or number")
    return directory_name


def canonicalize(url: str, *, keep_query: bool) -> str:
    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    query = ""
    if keep_query and parsed.query:
        pairs = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS
        ]
        query = urlencode(sorted(pairs))
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    path = re.sub(
        r"%[0-9a-fA-F]{2}",
        lambda match: match.group(0).upper(),
        path,
    )
    return urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, "")
    )


def host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def in_scope(url: str, root_url: str, include_subdomains: bool) -> bool:
    candidate = host(url)
    root = host(root_url)
    return candidate == root or (include_subdomains and candidate.endswith("." + root))


def resolve_discovered_url(base_url: str, reference: str) -> str:
    """Resolve links while treating a bare hostname as an absolute URL."""
    value = reference.strip()
    match = SCHEMELESS_HOST_RE.match(value)
    if match:
        candidate_path = "/" + match.group("host").split(":", 1)[0]
        if Path(candidate_path).suffix.lower() not in (
            SKIPPED_EXTENSIONS | PAGE_FILE_EXTENSIONS | DOCUMENT_FILE_EXTENSIONS
        ):
            value = "//" + value
    return urljoin(base_url, value)


def unsafe_url_reason(url: str) -> str | None:
    """Identify URL shapes that are characteristic of crawler traps."""
    if len(url) > MAX_URL_LENGTH:
        return "url_too_long"
    segments = [
        unquote(segment).casefold()
        for segment in urlparse(url).path.split("/")
        if segment
    ]
    if len(segments) > MAX_PATH_SEGMENTS:
        return "path_too_deep"
    for block_size in range(
        1, min(4, len(segments) // REPEATED_PATH_SEQUENCE_LIMIT) + 1
    ):
        repeated_size = block_size * REPEATED_PATH_SEQUENCE_LIMIT
        for start in range(len(segments) - repeated_size + 1):
            block = segments[start : start + block_size]
            if all(
                segments[
                    start + repeat * block_size : start + (repeat + 1) * block_size
                ]
                == block
                for repeat in range(1, REPEATED_PATH_SEQUENCE_LIMIT)
            ):
                return "repeated_path_sequence"
    return None


def sitemap_locations(content: bytes, url: str) -> tuple[str, list[str]]:
    try:
        if content[:2] == b"\x1f\x8b" or urlparse(url).path.endswith(".gz"):
            content = gzip.decompress(content)
        root = ET.fromstring(content)
    except (ET.ParseError, OSError):
        return "unknown", []
    kind = root.tag.rsplit("}", 1)[-1].lower()
    locations = [
        (node.text or "").strip()
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1].lower() == "loc" and (node.text or "").strip()
    ]
    return kind, locations


def html_text_and_links(url: str, html: str) -> tuple[str, set[str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = {
        resolve_discovered_url(url, str(tag["href"]))
        for tag in soup.find_all("a", href=True)
    }
    links.update(
        resolve_discovered_url(url, str(tag["src"]))
        for tag in soup.find_all("iframe", src=True)
    )

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    descriptions = [
        tag.get("content", "").strip()
        for tag in soup.find_all("meta")
        if (tag.get("name") or tag.get("property") or "").lower()
        in {"description", "og:description", "twitter:description"}
    ]
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    body = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    text = "\n".join(part for part in [title, *descriptions, body] if part)
    return text, links


def response_language_hint(response: requests.Response) -> str:
    header = response.headers.get("content-language", "").split(",", 1)[0].strip()
    if header:
        return header
    if "html" not in response.headers.get("content-type", "").lower():
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    html_tag = soup.find("html")
    return str(html_tag.get("lang", "")).strip() if html_tag else ""


def extract_response(response: requests.Response) -> tuple[str, set[str]]:
    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type:
        return html_text_and_links(response.url, response.text)
    if "pdf" in content_type or urlparse(response.url).path.lower().endswith(".pdf"):
        if PdfReader is None:
            return "", set()
        reader = PdfReader(io.BytesIO(response.content))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip(), set()
    if (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
    ):
        return response.text.strip(), set(
            re.findall(r"https?://[^\s\"'<>]+", response.text)
        )
    return "", set()


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
    site_dir = evidence_root / evidence_directory_name(site_name)
    site_dir.mkdir(parents=True, exist_ok=True)

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
    limit_reasons: set[str] = set()
    skipped_by_reason: Counter[str] = Counter()
    sitemap_seeds = {
        urljoin(root_url, "/sitemap.xml"),
        urljoin(root_url, "/sitemap_index.xml"),
        urljoin(root_url, "/wp-sitemap.xml"),
    }

    try:
        log(2, f"robots GET {robots_url}")
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
        log(2, f"robots ERROR {robots_url}: {exc}")

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
            log(2, f"sitemap GET {value}")
            response = session.get(value, timeout=request_timeout)
            response.raise_for_status()
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
            log(2, f"sitemap ERROR {value}: {exc}")

    for sitemap in sitemap_seeds:
        read_sitemap(sitemap)
    enqueue(root_url)

    pages: list[dict[str, object]] = []
    page_requests = 0
    while queue:
        if max_pages and len(pages) >= max_pages:
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
            pages.append(
                {
                    "url": final_url or response.url,
                    "content_type": response.headers.get("content-type", ""),
                    "language_hint": response_language_hint(response),
                    "text": text,
                }
            )
            for link in links:
                enqueue(link)
            log(2, f"queue size={len(queue)} after {response.url}")
        except (
            requests.RequestException,
            OSError,
            PdfReadError,
            ParserRejectedMarkup,
        ) as exc:
            errors.append({"url": url, "error": str(exc)})
            log(2, f"page ERROR {url}: {exc}")

    with (site_dir / "evidence.jsonl").open("w", encoding="utf-8") as output:
        for page in pages:
            output.write(json.dumps(page, ensure_ascii=False) + "\n")

    manifest = {
        "site_name": site_name,
        "root_url": root_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages_saved": len(pages),
        "urls_visited": len(visited),
        "page_requests_attempted": page_requests,
        "sitemaps_checked": len(sitemap_seen),
        "remaining_queue": len(queue),
        "crawl_complete": not limit_reasons,
        "crawl_limited": bool(limit_reasons),
        "crawl_limit_reasons": sorted(limit_reasons),
        "stopped_by_page_limit": "page_limit" in limit_reasons,
        "stopped_by_request_limit": "request_limit" in limit_reasons,
        "stopped_by_time_limit": "time_limit" in limit_reasons,
        "limits": limits,
        "urls_skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "duration_seconds": round(time.monotonic() - started_at, 3),
        "errors": errors,
    }
    (site_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(
        1,
        f"saved {len(pages)} page(s); visited={len(visited)}, "
        f"requests={page_requests}, errors={len(errors)}, remaining={len(queue)}, "
        f"limited={','.join(sorted(limit_reasons)) or 'no'}",
    )
    return site_dir
