from __future__ import annotations

import json
from typing import Any

import orchestrator


def _write_evidence(site_dir) -> None:
    site_dir.mkdir(parents=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
        encoding="utf-8",
    )


def test_crawl_mode_never_requires_key_or_calls_classifier(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        _write_evidence(site_dir)
        return site_dir

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("crawl mode must not call Jev")

    monkeypatch.delenv(orchestrator.JEV_API_KEY_ENV, raising=False)
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [{"name": "site1", "url": "https://example.com/"}],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "crawl"])

    assert calls["scraper"]["max_pages"] == 0
    output = capsys.readouterr().out
    assert "no Jev API calls will be made" in output
    assert "1 Jev request(s) would be required" in output


def test_smoke_mode_preserves_page_and_chunk_limits(tmp_path, monkeypatch) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        _write_evidence(site_dir)
        return site_dir

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "digital_twin_enabling",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [{"name": "site1", "url": "https://example.com/"}],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "smoke"])

    assert calls["scraper"]["max_pages"] == orchestrator.SMOKE_MAX_PAGES
    assert calls["classifier"]["max_chunks"] == orchestrator.SMOKE_MAX_CHUNKS
    assert calls["classifier"]["api_key"] == "test-secret"


def test_classify_mode_uses_saved_evidence_without_scraping(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fail_scrape_site(**kwargs):
        del kwargs
        raise AssertionError("classify mode must not scrape")

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": False,
            "classification": "not_digital_twin",
            "is_digital_twin": False,
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [{"name": "site1", "url": "https://example.com/"}],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fail_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify"])

    assert calls["classifier"]["evidence_dir"] == site_dir
    assert calls["classifier"]["max_chunks"] == 0
