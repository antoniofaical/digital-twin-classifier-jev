from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


try:
    import requests
except ImportError:
    requests_stub = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    requests_stub.RequestException = RequestException
    requests_stub.Response = object
    requests_stub.Session = object
    sys.modules["requests"] = requests_stub
else:
    if not hasattr(requests, "RequestException"):
        requests.RequestException = Exception
    if not hasattr(requests, "Response"):
        requests.Response = object
    if not hasattr(requests, "Session"):
        requests.Session = object

try:
    import bs4  # noqa: F401
except ImportError:
    bs4_stub = types.ModuleType("bs4")
    bs4_stub.BeautifulSoup = object
    sys.modules["bs4"] = bs4_stub


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
        return self.responses.get(url, FakeResponse(url, status_code=404))


class ScraperUnitTests(unittest.TestCase):
    def test_canonicalize_removes_tracking_and_sorts_query(self) -> None:
        result = scraper.canonicalize(
            "https://Example.com/a//b?utm_source=x&z=2&a=1#section",
            keep_query=True,
        )
        self.assertEqual(result, "https://example.com/a/b?a=1&z=2")

    def test_scope_rejects_lookalike_domains(self) -> None:
        root = "https://example.com/"
        self.assertTrue(scraper.in_scope("https://www.example.com/about", root, False))
        self.assertTrue(scraper.in_scope("https://docs.example.com/", root, True))
        self.assertFalse(scraper.in_scope("https://notexample.com/", root, True))

    def test_sitemap_index_and_urlset_are_parsed(self) -> None:
        index = b"<sitemapindex><sitemap><loc>https://example.com/pages.xml</loc></sitemap></sitemapindex>"
        urlset = b"<urlset><url><loc>https://example.com/about</loc></url></urlset>"
        self.assertEqual(
            scraper.sitemap_locations(index, "https://example.com/sitemap.xml"),
            ("sitemapindex", ["https://example.com/pages.xml"]),
        )
        self.assertEqual(
            scraper.sitemap_locations(urlset, "https://example.com/pages.xml"),
            ("urlset", ["https://example.com/about"]),
        )

    def test_scrape_site_creates_isolated_evidence_folder(self) -> None:
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

        with TemporaryDirectory() as tmp, patch.object(
            scraper.requests, "Session", return_value=fake_session
        ), patch.object(scraper, "extract_response", side_effect=fake_extract):
            site_dir = scraper.scrape_site(
                site_name="site1",
                root_url=root,
                evidence_root=Path(tmp) / "evidence",
                respect_robots=True,
            )

            self.assertEqual(site_dir, Path(tmp) / "evidence" / "site1")
            rows = [
                json.loads(line)
                for line in (site_dir / "evidence.jsonl").read_text().splitlines()
            ]
            self.assertEqual({row["url"] for row in rows}, {root, root + "about"})
            manifest = json.loads((site_dir / "manifest.json").read_text())
            self.assertEqual(manifest["pages_saved"], 2)
            self.assertFalse(manifest["stopped_by_page_limit"])


if __name__ == "__main__":
    unittest.main()
