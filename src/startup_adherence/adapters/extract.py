"""Sitemap parsing and document text extraction."""

from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ..domain.urls import resolve_discovered_url

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:
    PdfReader = None
    PdfReadError = ValueError


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
