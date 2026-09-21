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


def _configure_test_site(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [{"name": "site1", "url": "https://example.com/"}],
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
    _configure_test_site(tmp_path, monkeypatch)
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
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "smoke"])

    assert calls["scraper"]["max_pages"] == orchestrator.SMOKE_MAX_PAGES
    assert calls["classifier"]["max_chunks"] == orchestrator.SMOKE_MAX_CHUNKS
    assert calls["classifier"]["api_key"] == "test-secret"


def test_classify_mode_prompts_for_percentage_without_scraping(
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
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr("builtins.input", lambda _: "50")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fail_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify"])

    assert calls["classifier"]["evidence_dir"] == site_dir
    assert calls["classifier"]["max_chunks"] == 0
    assert calls["classifier"]["evidence_percentage"] == 50


def test_zero_percentage_cancels_without_jev_call(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("zero percent must not call Jev")

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr("builtins.input", lambda _: "0")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "classify"])

    assert "no Jev calls were made" in capsys.readouterr().out


def test_translation_requires_confirmation_before_api_calls(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        return site_dir

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "jev-secret")
    monkeypatch.setenv(orchestrator.DEEPL_API_KEY_ENV, "deepl-secret:fx")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "smoke", "--translation", "auto"])

    assert calls["classifier"]["translation_mode"] == "auto"
    assert calls["classifier"]["translation_target"] == "EN"
    assert calls["classifier"]["deepl_api_key"] == "deepl-secret:fx"


def test_rejected_translation_confirmation_cancels_all_api_calls(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)

    def fake_scrape_site(**kwargs):
        del kwargs
        return site_dir

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("rejected confirmation must not call APIs")

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "jev-secret")
    monkeypatch.setenv(orchestrator.DEEPL_API_KEY_ENV, "deepl-secret:fx")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)

    orchestrator.main(["--mode", "smoke", "--translation", "auto"])

    assert "no API calls were made" in capsys.readouterr().out
