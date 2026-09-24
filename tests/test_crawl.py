from __future__ import annotations

import json

import pytest

from startup_adherence.application import crawl
from startup_adherence.application.crawl_state import recent_crawl, recover, write_state
from startup_adherence.domain.urls import canonicalize, in_scope, unsafe_url_reason


class FakeResponse:
    def __init__(self, url, text="hello", *, status=200, content_type="text/html"):
        self.url = url
        self.text = text
        self.status_code = status
        self.ok = status == 200
        self.headers = {"content-type": content_type}
        self.content = text.encode()

    def raise_for_status(self):
        if not self.ok:
            raise crawl.requests.HTTPError(f"status={self.status_code}")


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.headers = {}
        self.calls = []

    def get(self, url, *, timeout):
        self.calls.append(url)
        return self.pages.get(url, FakeResponse(url, "", status=404))


def test_url_scope_and_trap_checks():
    assert (
        canonicalize("https://example.com/a?utm_source=x&b=2", keep_query=True)
        == "https://example.com/a?b=2"
    )
    assert in_scope("https://sub.example.com/", "https://example.com/", True)
    assert not in_scope("https://example.com.evil.test/", "https://example.com/", True)
    assert unsafe_url_reason("https://example.com/a/a/a") == "repeated_path_sequence"


def test_invalid_root_is_rejected_without_persisting_attempt(tmp_path):
    with pytest.raises(ValueError, match="Site URL"):
        crawl.scrape_site(site_name="x", root_url="https://", evidence_root=tmp_path)
    assert not (tmp_path / "x").exists()


def test_redirected_page_is_never_saved_as_company_evidence(tmp_path, monkeypatch):
    pages = {
        "https://example.com/": FakeResponse(
            "https://unrelated.test/", "<p>Out of scope</p>"
        ),
    }
    fake = FakeSession(pages)
    monkeypatch.setattr(crawl.requests, "Session", lambda: fake)
    with pytest.raises(ValueError, match="No pages"):
        crawl.scrape_site(
            site_name="one",
            root_url="https://example.com/",
            evidence_root=tmp_path,
            max_sitemaps=1,
        )
    assert not (tmp_path / "one" / "evidence.jsonl").exists()
    attempt = next((tmp_path / "one" / "crawl_attempts").iterdir())
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert len(manifest["page_errors"]) == 1
    assert not manifest["crawl_complete"]


def test_crawl_cache_requires_matching_hash_and_recovery_is_offline(
    tmp_path, monkeypatch
):
    root = "https://example.com/"
    fake = FakeSession({root: FakeResponse(root, "<html><body>Company</body></html>")})
    monkeypatch.setattr(crawl.requests, "Session", lambda: fake)
    directory = crawl.scrape_site(
        site_name="one", root_url=root, evidence_root=tmp_path, max_sitemaps=1
    )
    site = {"name": "one", "url": root}
    config = {"max_pages": 400, "include_query_urls": False}
    write_state(site, directory, config)
    assert recent_crawl(site, directory, config, 24)
    (directory / "evidence.jsonl").write_text("changed")
    assert not recent_crawl(site, directory, config, 24)
    (directory / "evidence.jsonl").write_text(
        json.dumps({"url": root, "text": "Company"}) + "\n"
    )
    assert recover(site, directory, config, 20_000).startswith("recovered")
    assert recent_crawl(site, directory, config, 24)
