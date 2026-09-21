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
        self.requested: list[str] = []

    def get(self, url: str, timeout: int) -> FakeResponse:
        del timeout
        self.requested.append(url)
        return self.responses.get(url, FakeResponse(url, status_code=404))


def test_canonicalize_removes_tracking_and_sorts_query() -> None:
    result = scraper.canonicalize(
        "https://Example.com/a//b?utm_source=x&z=2&a=1#section",
        keep_query=True,
    )
    assert result == "https://example.com/a/b?a=1&z=2"
    assert (
        scraper.canonicalize("https://example.com/%d8%a8", keep_query=False)
        == "https://example.com/%D8%A8"
    )


def test_bare_hostname_is_resolved_as_absolute_not_as_a_relative_path() -> None:
    base = "https://homox.io/articles/"

    assert (
        scraper.resolve_discovered_url(base, "pdt.homox.io/") == "https://pdt.homox.io/"
    )
    assert (
        scraper.resolve_discovered_url(base, "report.pdf")
        == "https://homox.io/articles/report.pdf"
    )


def test_recursive_path_shape_is_rejected() -> None:
    assert (
        scraper.unsafe_url_reason(
            "https://homox.io/wishlist/pdt.homox.io/pdt.homox.io/pdt.homox.io/"
        )
        == "repeated_path_sequence"
    )
    assert scraper.unsafe_url_reason("https://homox.io/articles/product/") is None


def test_scope_rejects_lookalike_domains() -> None:
    root = "https://example.com/"
    assert scraper.in_scope("https://www.example.com/about", root, False)
    assert scraper.in_scope("https://docs.example.com/", root, True)
    assert not scraper.in_scope("https://notexample.com/", root, True)


def test_evidence_directory_name_slugifies_human_readable_names() -> None:
    assert (
        scraper.evidence_directory_name("Thoth BioSimulations")
        == "Thoth-BioSimulations"
    )
    assert scraper.evidence_directory_name("Médico & Saúde") == "Medico-Saude"


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


def test_scrape_site_creates_isolated_evidence_folder(
    tmp_path, monkeypatch, capsys
) -> None:
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
        verbose=2,
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
    output = capsys.readouterr().out
    assert "[site1] robots GET https://example.com/robots.txt" in output
    assert "[site1] sitemap HTTP 200" in output
    assert "[site1] page 1 GET" in output
    assert "characters=" in output


def test_verbose_logs_can_be_routed_to_progress_callback(
    tmp_path, monkeypatch, capsys
) -> None:
    root = "https://example.com/"
    fake_session = FakeSession(
        {
            root: FakeResponse(
                root,
                text="<html><body>home</body></html>",
            )
        }
    )
    messages: list[str] = []
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path / "evidence",
        respect_robots=False,
        verbose=1,
        log_callback=messages.append,
    )

    assert messages[0] == "[site1] page 1 GET https://example.com/"
    assert messages[-1].startswith("[site1] saved 1 page(s)")
    assert capsys.readouterr().out == ""


def test_homox_style_bare_hostname_does_not_create_recursive_paths(
    tmp_path, monkeypatch
) -> None:
    root = "https://homox.io/"
    wishlist = "https://homox.io/wishlist/"
    fake_session = FakeSession(
        {
            root: FakeResponse(
                root,
                text=(
                    '<html><body>Home<a href="/wishlist/">Wishlist</a>'
                    '<a href="pdt.homox.io/">Product</a></body></html>'
                ),
            ),
            wishlist: FakeResponse(
                wishlist,
                text=(
                    "<html><body>Wishlist details"
                    '<a href="pdt.homox.io/">Product</a></body></html>'
                ),
            ),
        }
    )
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    site_dir = scraper.scrape_site(
        site_name="Homox",
        root_url=root,
        evidence_root=tmp_path,
        respect_robots=False,
    )

    page_requests = [url for url in fake_session.requested if url in {root, wishlist}]
    assert page_requests == [root, wishlist]
    assert not any("/pdt.homox.io" in url for url in fake_session.requested)
    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["page_requests_attempted"] == 2
    assert not manifest["crawl_limited"]


def test_request_budget_stops_an_unbounded_link_chain(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    responses = {
        root: FakeResponse(root, text='<a href="/1">one</a>'),
        root + "1": FakeResponse(root + "1", text='<a href="/2">two</a>'),
        root + "2": FakeResponse(root + "2", text='<a href="/3">three</a>'),
    }
    fake_session = FakeSession(responses)
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path,
        max_pages=0,
        max_requests=2,
        respect_robots=False,
    )

    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["page_requests_attempted"] == 2
    assert manifest["remaining_queue"] == 1
    assert manifest["crawl_limited"]
    assert manifest["crawl_limit_reasons"] == ["request_limit"]
    assert manifest["stopped_by_request_limit"]


def test_queue_budget_records_dropped_urls_as_partial(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    links = "".join(f'<a href="/{number}">page</a>' for number in range(5))
    fake_session = FakeSession({root: FakeResponse(root, text=links)})
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path,
        max_queue_size=2,
        respect_robots=False,
    )

    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["crawl_limited"]
    assert manifest["crawl_limit_reasons"] == ["queue_limit"]
    assert manifest["urls_skipped_by_reason"]["queue_limit"] == 3


def test_duplicate_content_does_not_expand_more_links(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    copy = root + "copy/"
    trap = root + "trap/"
    fake_session = FakeSession(
        {
            root: FakeResponse(root, text='<body>Same<a href="/copy/"></a></body>'),
            copy: FakeResponse(copy, text='<body>Same<a href="/trap/"></a></body>'),
            trap: FakeResponse(trap, text="should not be requested"),
        }
    )
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path,
        respect_robots=False,
    )

    assert copy in fake_session.requested
    assert trap not in fake_session.requested
    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["pages_saved"] == 1
    assert manifest["urls_skipped_by_reason"]["duplicate_content"] == 1


def test_sitemap_budget_is_audited_as_partial(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    fake_session = FakeSession({root: FakeResponse(root, text="home")})
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path,
        max_sitemaps=1,
        respect_robots=False,
    )

    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["sitemaps_checked"] == 1
    assert manifest["crawl_limited"]
    assert manifest["crawl_limit_reasons"] == ["sitemap_limit"]


def test_time_budget_stops_before_next_page_request(tmp_path, monkeypatch) -> None:
    root = "https://example.com/"
    fake_session = FakeSession({root: FakeResponse(root, text="home")})
    ticks = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(scraper.time, "monotonic", lambda: next(ticks, 10.0))

    site_dir = scraper.scrape_site(
        site_name="site1",
        root_url=root,
        evidence_root=tmp_path,
        max_crawl_seconds=5,
        max_sitemaps=0,
        respect_robots=False,
    )

    manifest = json.loads((site_dir / "manifest.json").read_text())
    assert manifest["page_requests_attempted"] == 0
    assert manifest["crawl_limit_reasons"] == ["time_limit"]
    assert manifest["stopped_by_time_limit"]


def test_language_hint_reads_html_lang_attribute() -> None:
    response = FakeResponse(
        "https://example.com/",
        text='<html lang="es"><body>Hola</body></html>',
    )
    assert scraper.response_language_hint(response) == "es"
