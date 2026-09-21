from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import orchestrator


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonTTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_evidence(site_dir) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "evidence.jsonl").write_text(
        json.dumps({"url": "https://example.com/", "text": "evidence"}) + "\n",
        encoding="utf-8",
    )
    (site_dir / "manifest.json").write_text(
        json.dumps(
            {
                "site_name": site_dir.name,
                "pages_saved": 1,
                "stopped_by_page_limit": False,
                "errors": [],
            }
        )
        + "\n",
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

    orchestrator.main(["--mode", "crawl", "-vv"])

    assert calls["scraper"]["max_pages"] == orchestrator.DEFAULT_MAX_PAGES
    assert calls["scraper"]["max_requests"] == orchestrator.DEFAULT_MAX_REQUESTS
    assert calls["scraper"]["max_queue_size"] == orchestrator.DEFAULT_MAX_QUEUE_SIZE
    assert (
        calls["scraper"]["max_crawl_seconds"] == orchestrator.DEFAULT_MAX_CRAWL_SECONDS
    )
    assert calls["scraper"]["max_sitemaps"] == orchestrator.DEFAULT_MAX_SITEMAPS
    assert calls["scraper"]["verbose"] == 2
    output = capsys.readouterr().out
    assert "no Jev or DeepL API calls will be made" in output
    assert "1 Jev request(s) would be required" in output


def test_terminal_progress_renders_logs_and_terminal_states() -> None:
    stream = TTYBuffer()
    progress = orchestrator.TerminalProgress(3, enabled=True, stream=stream)

    progress.start()
    progress.set_active(2)
    progress.log("[site1] page 1 GET https://example.com")
    progress.record_terminal("site1", "completed", reused=True)
    progress.record_terminal("site2", "failed")
    progress.record_terminal("site3", "cancelled")
    progress.stop()

    output = stream.getvalue()
    assert "[site1] page 1 GET https://example.com\n" in output
    assert "Sites" in output
    assert "3/3 100%" in output
    assert "completed=1" in output
    assert "reused=1" in output
    assert "failed=1" in output
    assert progress.completed == 3
    assert progress.active == 0


def test_terminal_progress_rejects_duplicate_terminal_site() -> None:
    progress = orchestrator.TerminalProgress(1, enabled=False)
    progress.record_terminal("site1", "completed")

    with pytest.raises(ValueError, match="already recorded"):
        progress.record_terminal("site1", "failed")


def test_progress_auto_disables_for_non_tty_and_explicit_flag() -> None:
    assert orchestrator.terminal_progress_enabled(disabled=False, stream=TTYBuffer())
    assert not orchestrator.terminal_progress_enabled(
        disabled=False, stream=NonTTYBuffer()
    )
    assert not orchestrator.terminal_progress_enabled(disabled=True, stream=TTYBuffer())
    assert orchestrator.parse_args(["--mode", "crawl", "--no-progress"]).no_progress


def test_crawl_limits_have_safe_defaults_and_can_be_disabled() -> None:
    defaults = orchestrator.parse_args(["--mode", "crawl"])
    assert defaults.max_pages_per_site == orchestrator.DEFAULT_MAX_PAGES
    assert defaults.max_requests_per_site == orchestrator.DEFAULT_MAX_REQUESTS
    assert defaults.max_queue_size == orchestrator.DEFAULT_MAX_QUEUE_SIZE
    assert defaults.max_crawl_seconds == orchestrator.DEFAULT_MAX_CRAWL_SECONDS
    assert defaults.max_sitemaps_per_site == orchestrator.DEFAULT_MAX_SITEMAPS

    disabled = orchestrator.parse_args(
        [
            "--mode",
            "crawl",
            "--max-pages-per-site",
            "0",
            "--max-requests-per-site",
            "0",
            "--max-queue-size",
            "0",
            "--max-crawl-seconds",
            "0",
            "--max-sitemaps-per-site",
            "0",
        ]
    )
    assert disabled.max_pages_per_site == 0
    assert disabled.max_requests_per_site == 0
    assert disabled.max_queue_size == 0
    assert disabled.max_crawl_seconds == 0
    assert disabled.max_sitemaps_per_site == 0


def test_run_site_jobs_tracks_concurrent_success_failure_and_reuse() -> None:
    sites = [
        {"name": "site1", "url": "https://one.example/"},
        {"name": "site2", "url": "https://two.example/"},
        {"name": "site3", "url": "https://three.example/"},
    ]
    barrier = threading.Barrier(3)
    progress = orchestrator.TerminalProgress(3, enabled=False)

    def operation(site):
        barrier.wait(timeout=2)
        if site["name"] == "site3":
            raise RuntimeError("simulated failure")
        return site["name"]

    results, errors = orchestrator.run_site_jobs(
        sites,
        3,
        operation,
        progress=progress,
        successes_are_terminal=True,
        reused_names={"site2"},
    )

    assert set(results) == {"site1", "site2"}
    assert set(errors) == {"site3"}
    assert progress.completed == 3
    assert progress.successful == 2
    assert progress.reused == 1
    assert progress.failed == 1
    assert progress.active == 0


def test_smoke_mode_preserves_page_and_chunk_limits(tmp_path, monkeypatch) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    site = {"name": "site1", "url": "https://example.com/"}
    _write_evidence(site_dir)
    orchestrator.write_crawl_state(
        site=site,
        site_dir=site_dir,
        scraper_config={**orchestrator.SCRAPER_CONFIG, "max_pages": 0, "verbose": 0},
    )
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
    assert not (site_dir / orchestrator.CRAWL_STATE_FILENAME).exists()


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
    answers = iter(["50", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
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

    assert "no API calls were made" in capsys.readouterr().out


def test_site_intent_selects_only_requested_sites(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        calls.append(name)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    orchestrator.main(["--mode", "crawl", "--site", "site2"])

    assert calls == ["site2"]


def test_legacy_artifacts_are_cleaned_without_deleting_unknown_files(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    (site_dir / "classification.json").write_text("{}\n", encoding="utf-8")
    (site_dir / "user-notes.txt").write_text("keep me\n", encoding="utf-8")
    calls = 0

    def fake_scrape_site(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        assert not (site_dir / "evidence.jsonl").exists()
        assert not (site_dir / "manifest.json").exists()
        assert not (site_dir / "classification.json").exists()
        assert (site_dir / "user-notes.txt").read_text(encoding="utf-8") == "keep me\n"
        _write_evidence(site_dir)
        return site_dir

    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    orchestrator.main(["--mode", "crawl"])

    assert calls == 1
    assert (site_dir / "user-notes.txt").exists()
    state = json.loads((site_dir / orchestrator.CRAWL_STATE_FILENAME).read_text())
    assert state["schema_version"] == orchestrator.CRAWL_STATE_VERSION
    assert state["status"] == "completed"
    assert "removed stale generated artifacts" in capsys.readouterr().out


def test_recent_full_crawl_is_reused_and_force_crawl_bypasses_cache(
    tmp_path, monkeypatch, capsys
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    calls = 0
    progress_instances: list[orchestrator.TerminalProgress] = []
    progress_class = orchestrator.TerminalProgress

    class TrackingProgress(progress_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            progress_instances.append(self)

    def fake_scrape_site(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        _write_evidence(site_dir)
        return site_dir

    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "TerminalProgress", TrackingProgress)

    orchestrator.main(["--mode", "crawl"])
    orchestrator.main(["--mode", "crawl"])
    assert calls == 1
    assert progress_instances[1].completed == 1
    assert progress_instances[1].successful == 1
    assert progress_instances[1].reused == 1
    assert "reusing crawl completed" in capsys.readouterr().out

    orchestrator.main(["--mode", "crawl", "--force-crawl"])
    assert calls == 2


def test_crawl_freshness_uses_only_explicit_state(tmp_path) -> None:
    site = {"name": "site1", "url": "https://example.com/"}
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    config = {**orchestrator.SCRAPER_CONFIG, "max_pages": 0, "verbose": 0}
    now = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)

    is_recent, _ = orchestrator.recent_crawl(
        site=site,
        site_dir=site_dir,
        scraper_config=config,
        max_age_hours=24,
        now=now,
    )
    assert not is_recent

    orchestrator.write_crawl_state(
        site=site,
        site_dir=site_dir,
        scraper_config=config,
        completed_at=now - timedelta(hours=23),
    )
    is_recent, age_hours = orchestrator.recent_crawl(
        site=site,
        site_dir=site_dir,
        scraper_config=config,
        max_age_hours=24,
        now=now,
    )
    assert is_recent
    assert age_hours == 23

    is_recent, _ = orchestrator.recent_crawl(
        site=site,
        site_dir=site_dir,
        scraper_config=config,
        max_age_hours=22,
        now=now,
    )
    assert not is_recent


def test_naturally_completed_legacy_unlimited_crawl_remains_reusable(tmp_path) -> None:
    site = {"name": "site1", "url": "https://example.com/"}
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    legacy_config = {
        **orchestrator.SCRAPER_CONFIG,
        "max_pages": 0,
        "verbose": 0,
    }
    orchestrator.write_crawl_state(
        site=site,
        site_dir=site_dir,
        scraper_config=legacy_config,
    )
    state_file = site_dir / orchestrator.CRAWL_STATE_FILENAME
    state = json.loads(state_file.read_text())
    state["crawl_signature"] = {
        key: value
        for key, value in state["crawl_signature"].items()
        if key
        in {
            "root_url",
            "include_subdomains",
            "include_query_urls",
            "respect_robots",
            "max_pages",
        }
    }
    state_file.write_text(json.dumps(state) + "\n", encoding="utf-8")
    current_config = {
        **orchestrator.SCRAPER_CONFIG,
        "max_pages": orchestrator.DEFAULT_MAX_PAGES,
        "verbose": 0,
    }

    is_recent, _ = orchestrator.recent_crawl(
        site=site,
        site_dir=site_dir,
        scraper_config=current_config,
        max_age_hours=24,
    )

    assert is_recent


def test_crawl_state_preserves_limit_audit_data(tmp_path) -> None:
    site = {"name": "site1", "url": "https://example.com/"}
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    manifest_file = site_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest.update(
        {
            "crawl_limited": True,
            "crawl_limit_reasons": ["request_limit"],
        }
    )
    manifest_file.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    config = {
        **orchestrator.SCRAPER_CONFIG,
        "max_pages": orchestrator.DEFAULT_MAX_PAGES,
        "verbose": 0,
    }

    orchestrator.write_crawl_state(
        site=site,
        site_dir=site_dir,
        scraper_config=config,
    )

    state = json.loads((site_dir / orchestrator.CRAWL_STATE_FILENAME).read_text())
    assert state["crawl_limited"]
    assert state["crawl_limit_reasons"] == ["request_limit"]


def test_classify_uses_safe_directory_for_display_name(tmp_path, monkeypatch) -> None:
    site_dir = tmp_path / "evidence" / "Thoth-BioSimulations"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fake_classify_site(**kwargs):
        calls["classifier"] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {
                "name": "Thoth BioSimulations",
                "url": "https://www.thothbiosimulations.ca",
            }
        ],
    )
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify", "--percentage", "100", "--yes"])

    assert calls["classifier"]["site_name"] == "Thoth BioSimulations"
    assert calls["classifier"]["evidence_dir"] == site_dir


def test_bulk_rejects_evidence_directory_collisions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "Site One", "url": "https://one.example/"},
            {"name": "Site-One", "url": "https://two.example/"},
        ],
    )

    with pytest.raises(ValueError, match="same evidence directory"):
        orchestrator.main(["--mode", "crawl", "--workers", "2"])


def test_crawl_bulk_processes_sites_concurrently(tmp_path, monkeypatch) -> None:
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        calls.append(name)
        barrier.wait(timeout=2)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    orchestrator.main(["--mode", "crawl", "--sites", "all", "--workers", "2"])

    assert set(calls) == {"site1", "site2"}


def test_classify_bulk_prompts_once_and_applies_percentage_per_site(
    tmp_path, monkeypatch, capsys
) -> None:
    sites = [
        {"name": "site1", "url": "https://one.example/"},
        {"name": "site2", "url": "https://two.example/"},
    ]
    for site in sites:
        _write_evidence(tmp_path / "evidence" / site["name"])
    prompts: list[str] = []
    answers = iter(["50", "y"])
    calls: dict[str, dict[str, Any]] = {}

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    def fake_classify_site(**kwargs):
        calls[kwargs["site_name"]] = kwargs
        return {
            "evidence_is_partial": True,
            "provisional_classification": "not_digital_twin",
        }

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "test-secret")
    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(orchestrator, "SITES", sites)
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(orchestrator, "classify_site", fake_classify_site)

    orchestrator.main(["--mode", "classify", "--workers", "2"])

    assert len(prompts) == 2
    assert set(calls) == {"site1", "site2"}
    assert all(call["evidence_percentage"] == 50 for call in calls.values())
    output = capsys.readouterr().out
    assert "Sites ready: 2" in output
    assert "Jev requests: 2" in output


def test_bulk_failure_does_not_cancel_other_sites(
    tmp_path, monkeypatch, capsys
) -> None:
    completed: list[str] = []

    def fake_scrape_site(**kwargs):
        name = kwargs["site_name"]
        if name == "site1":
            raise RuntimeError("simulated failure")
        completed.append(name)
        site_dir = tmp_path / "evidence" / name
        _write_evidence(site_dir)
        return site_dir

    monkeypatch.setattr(orchestrator, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(
        orchestrator,
        "SITES",
        [
            {"name": "site1", "url": "https://one.example/"},
            {"name": "site2", "url": "https://two.example/"},
        ],
    )
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)

    with pytest.raises(orchestrator.BulkExecutionError):
        orchestrator.main(["--mode", "crawl", "--workers", "2"])

    assert completed == ["site2"]
    output = capsys.readouterr().out
    assert "selected=2, completed=1, failed=1" in output
    assert "[site1] failed: RuntimeError: simulated failure" in output


def test_translation_requires_confirmation_before_api_calls(
    tmp_path, monkeypatch
) -> None:
    site_dir = tmp_path / "evidence" / "site1"
    _write_evidence(site_dir)
    calls: dict[str, Any] = {}

    def fake_scrape_site(**kwargs):
        calls["scraper"] = kwargs
        _write_evidence(site_dir)
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
        _write_evidence(site_dir)
        return site_dir

    def fail_classify_site(**kwargs):
        del kwargs
        raise AssertionError("rejected confirmation must not call APIs")

    def fail_progress(*args, **kwargs):
        del args, kwargs
        raise AssertionError("rejected preflight must not start progress")

    monkeypatch.setenv(orchestrator.JEV_API_KEY_ENV, "jev-secret")
    monkeypatch.setenv(orchestrator.DEEPL_API_KEY_ENV, "deepl-secret:fx")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _configure_test_site(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "scrape_site", fake_scrape_site)
    monkeypatch.setattr(orchestrator, "classify_site", fail_classify_site)
    monkeypatch.setattr(orchestrator, "TerminalProgress", fail_progress)

    orchestrator.main(["--mode", "smoke", "--translation", "auto"])

    assert "no API calls were made" in capsys.readouterr().out
