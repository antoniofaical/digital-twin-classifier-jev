"""Discover a public website and save its textual evidence."""

from __future__ import annotations

import gzip
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


USER_AGENT = "DigitalTwinClassifier/1.0"
SKIPPED_EXTENSIONS = {
    ".7z", ".avi", ".css", ".eot", ".exe", ".gif", ".ico", ".jpeg",
    ".jpg", ".js", ".map", ".mkv", ".mov", ".mp3", ".mp4", ".png",
    ".rar", ".svg", ".tar", ".ttf", ".wav", ".webm", ".webp",
    ".woff", ".woff2", ".zip",
}
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


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
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def in_scope(url: str, root_url: str, include_subdomains: bool) -> bool:
    candidate = host(url)
    root = host(root_url)
    return candidate == root or (include_subdomains and candidate.endswith("." + root))


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
    links = {urljoin(url, tag["href"]) for tag in soup.find_all("a", href=True)}
    links.update(urljoin(url, tag["src"]) for tag in soup.find_all("iframe", src=True))

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
    if content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
        return response.text.strip(), set(
            re.findall(r"https?://[^\s\"'<>]+", response.text)
        )
    return "", set()


def scrape_site(
    *,
    site_name: str,
    root_url: str,
    evidence_root: Path,
    max_pages: int = 0,
    include_subdomains: bool = False,
    include_query_urls: bool = False,
    respect_robots: bool = True,
    request_timeout: int = 20,
) -> Path:
    """Crawl one site and create evidence/<site_name>/evidence.jsonl."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", site_name):
        raise ValueError("site_name may contain only letters, numbers, dots, dashes, and underscores")

    root_url = root_url if re.match(r"^https?://", root_url) else "https://" + root_url
    root_url = canonicalize(root_url, keep_query=include_query_urls)
    site_dir = evidence_root / site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots = RobotFileParser()
    robots_url = urljoin(root_url, "/robots.txt")
    robots.set_url(robots_url)
    errors: list[dict[str, str]] = []
    sitemap_seeds = {
        urljoin(root_url, "/sitemap.xml"),
        urljoin(root_url, "/sitemap_index.xml"),
        urljoin(root_url, "/wp-sitemap.xml"),
    }

    try:
        response = session.get(robots_url, timeout=request_timeout)
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

    queue: deque[str] = deque()
    queued: set[str] = set()
    visited: set[str] = set()
    sitemap_seen: set[str] = set()

    def enqueue(url: str) -> None:
        value = canonicalize(url, keep_query=include_query_urls)
        if not value or not in_scope(value, root_url, include_subdomains):
            return
        if Path(urlparse(value).path.lower()).suffix in SKIPPED_EXTENSIONS:
            return
        if value not in queued and value not in visited:
            queued.add(value)
            queue.append(value)

    def read_sitemap(url: str) -> None:
        value = canonicalize(url, keep_query=True)
        if not value or value in sitemap_seen or not in_scope(value, root_url, include_subdomains):
            return
        sitemap_seen.add(value)
        try:
            response = session.get(value, timeout=request_timeout)
            response.raise_for_status()
            kind, locations = sitemap_locations(response.content, response.url)
            if kind == "sitemapindex":
                for location in locations:
                    read_sitemap(location)
            else:
                for location in locations:
                    enqueue(location)
        except requests.RequestException as exc:
            errors.append({"url": value, "error": str(exc)})

    for sitemap in sitemap_seeds:
        read_sitemap(sitemap)
    enqueue(root_url)

    pages: list[dict[str, object]] = []
    while queue and (max_pages == 0 or len(pages) < max_pages):
        url = queue.popleft()
        queued.discard(url)
        if url in visited:
            continue
        visited.add(url)
        if respect_robots and not robots.can_fetch(USER_AGENT, url):
            continue
        try:
            response = session.get(url, timeout=request_timeout)
            response.raise_for_status()
            text, links = extract_response(response)
            pages.append(
                {
                    "url": response.url,
                    "content_type": response.headers.get("content-type", ""),
                    "text": text,
                }
            )
            for link in links:
                enqueue(link)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    with (site_dir / "evidence.jsonl").open("w", encoding="utf-8") as output:
        for page in pages:
            output.write(json.dumps(page, ensure_ascii=False) + "\n")

    manifest = {
        "site_name": site_name,
        "root_url": root_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages_saved": len(pages),
        "urls_visited": len(visited),
        "sitemaps_checked": len(sitemap_seen),
        "remaining_queue": len(queue),
        "stopped_by_page_limit": bool(queue and max_pages and len(pages) >= max_pages),
        "errors": errors,
    }
    (site_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return site_dir
