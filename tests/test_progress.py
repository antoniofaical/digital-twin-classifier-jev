"""Terminal behavior with concurrent stage callbacks and plain redirected logs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import StringIO

from startup_adherence.progress import ProgressReporter


class Terminal(StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_stage_colors_and_concurrent_updates(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    stream = Terminal()
    reporter = ProgressReporter(stream=stream)
    reporter.begin("jev", 20)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: reporter.advance("jev"), range(20)))
    reporter.begin("deepl", 1)
    reporter.advance("deepl")
    reporter.begin("classification", 1)
    reporter.advance("classification")
    output = stream.getvalue()
    assert "JEV PROGRESS: 20/20" in output
    assert "\x1b[33m" in output
    assert "\x1b[35m" in output
    assert "\x1b[32m" in output


def test_redirected_output_is_plain_and_no_progress_is_silent():
    stream = StringIO()
    reporter = ProgressReporter(stream=stream)
    reporter.begin("crawl", 2)
    reporter.on_batch("crawl", "one", 1, 2, False)
    reporter.on_batch("crawl", "two", 2, 2, True)
    output = stream.getvalue()
    assert "CRAWL PROGRESS: 2/2" in output
    assert "failed=1" in output
    assert "\x1b[" not in output

    quiet = StringIO()
    reporter = ProgressReporter(enabled=False, stream=quiet)
    reporter.begin("jev", 1)
    reporter.on_service("one", "jev")
    reporter.finish()
    assert quiet.getvalue() == ""
