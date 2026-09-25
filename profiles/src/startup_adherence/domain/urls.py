"""URL normalization, scope and crawl-trap detection."""

from __future__ import annotations

import re
import unicodedata
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

USER_AGENT = "StartupThemeAdherence/2.0"
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
