from __future__ import annotations

import json

import scraper


class FakeResponse:
    def __init__(
        self,
        url: str,
        *,
        text: str = "",
        content_type: str = "text/html",
        status_code: int = 200,
    ) -> None:
        self.url = url
        self.text = text
        self.content = text.encode()
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.ok = status_code < 400

    def raise_for_status(self) -> None:
        if not self.ok:
            raise scraper.requests.RequestException(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: int) -> FakeResponse:
        del timeout
        return self.responses.get(url, FakeResponse(url, status_code=404))


def test_canonicalize_removes_tracking_and_sorts_query() -> None:
    result = scraper.canonicalize(
        "https://Example.com/a//b?utm_source=x&z=2&a=1#section",
        keep_query=True,
    )
    assert result == "https://example.com/a/b?a=1&z=2"


def test_scope_rejects_lookalike_domains() -> None:
    root = "https://example.com/"
    assert scraper.in_scope("https://www.example.com/about", root, False)
    assert scraper.in_scope("https://docs.example.com/", root, True)
    assert not scraper.in_scope("https://notexample.com/", root, True)


def test_sitemap_index_and_urlset_are_parsed() -> None:
    index = (
        b"<sitemapindex><sitemap><loc>https://example.com/pages.xml</loc>"
        b"</sitemap></sitemapindex>"
    )
    urlset = b"<urlset><url><loc>https://example.com/about</loc></url></urlset>"
    assert scraper.sitemap_locations(index, "https://example.com/sitemap.xml") == (
        "sitemapindex",
        ["https://example.com/pages.xml"],
    )
    assert scraper.sitemap_locations(urlset, "https://example.com/pages.xml") == (
        "urlset",
        ["https://example.com/about"],
    )


def test_scrape_site_creates_isolated_evidence_folder(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    sitemap = (
        "<urlset>"
        "<url><loc>https://example.com/</loc></url>"
        "<url><loc>https://example.com/about</loc></url>"
        "</urlset>"
    )
    responses = {
        "https://example.com/robots.txt": FakeResponse(
            "https://example.com/robots.txt",
            text="User-agent: *\nAllow: /\nSitemap: https://example.com/sitemap.xml",
            content_type="text/plain",
        ),
        "https://example.com/sitemap.xml": FakeResponse(
            "https://example.com/sitemap.xml",
            text=sitemap,
            content_type="application/xml",
        ),
        root: FakeResponse(root, text="home"),
        "https://example.com/about": FakeResponse(
            "https://example.com/about", text="about"
        ),
    }
    fake_session = FakeSession(responses)

    def fake_extract(response: FakeResponse) -> tuple[str, set[str]]:
        links = {"https://example.com/about"} if response.url == root else set()
        return response.text, links

    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(scraper, "extract_response", fake_extract)
    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path / "evidence",
        respect_robots=True,
    )

    assert site_dir == tmp_path / "evidence" / "site1"
    rows = [
        json.loads(line)
        for line in (site_dir / "evidence.jsonl").read_text().splitlines()
    ]
    assert {row["url"] for row in rows} == {root, root + "about"}
    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["pages_saved"] == 2
    assert not manifest["stopped_by_page_limit"]


def test_language_hint_reads_html_lang_attribute() -> None:
    response = FakeResponse(
        "https://example.com/",
        text='<html lang="es"><body>Hola</body></html>',
    )
    assert scraper.response_language_hint(response) == "es"
