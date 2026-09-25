"""Thread-safe terminal progress for batch and text workflows."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from threading import RLock
from typing import TextIO


@dataclass
class _Stage:
    total: int
    done: int = 0
    failed: int = 0


_COLORS = {
    "saved_result": "36",  # cyan
    "crawl_recovery": "36",
    "crawl": "36",
    "evidence_plan": "34",  # blue
    "deepl": "35",  # magenta
    "jev": "33",  # yellow
    "classification": "32",  # green
    "scoring": "32",
    "overview": "32",
    "export": "32",
}


def _terminal_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None or not stream.isatty():
        return False
    if os.name != "nt":
        return os.environ.get("TERM") != "dumb"
    # Windows console needs Virtual Terminal processing for ANSI colors.
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.GetStdHandle.argtypes = [ctypes.c_int]
        kernel.GetStdHandle.restype = ctypes.c_void_p
        kernel.GetConsoleMode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        kernel.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        handle = kernel.GetStdHandle(-12 if stream is sys.stderr else -11)
        mode = ctypes.c_uint()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError, ValueError):
        return False


class ProgressReporter:
    """Keep stage counters and rendering away from application orchestration."""

    def __init__(self, *, enabled: bool = True, stream: TextIO | None = None):
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = enabled and self.stream.isatty()
        self.color = self.interactive and _terminal_color(self.stream)
        self._stages: dict[str, _Stage] = {}
        self._lock = RLock()
        self._live = False

    def begin(self, stage: str, total: int) -> None:
        if not self.enabled:
            return
        if total < 0:
            raise ValueError("Progress total cannot be negative")
        with self._lock:
            self._stages[stage] = _Stage(total)
            self._render(stage, "start")

    def advance(self, stage: str, *, site: str = "", failed: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            state = self._stages[stage]
            if state.done >= state.total:
                raise ValueError(f"Too many progress updates for {stage}")
            state.done += 1
            state.failed += int(failed)
            self._render(
                stage, f"[{site}] {'failed' if failed else 'done'}" if site else ""
            )

    def on_batch(
        self, stage: str, site: str, done: int, total: int, failed: bool
    ) -> None:
        self.advance(stage, site=site, failed=failed)

    def on_service(self, site: str, service: str) -> None:
        self.advance(service, site=site)

    def message(self, message: str) -> None:
        with self._lock:
            self._end_line()
            print(message, file=self.stream, flush=True)

    def finish(self) -> None:
        with self._lock:
            self._end_line()

    def _end_line(self) -> None:
        if self._live:
            print(file=self.stream, flush=True)
            self._live = False

    def _render(self, stage: str, detail: str) -> None:
        state = self._stages[stage]
        width = 20
        filled = width * state.done // max(state.total, 1)
        bar = "█" * filled + "░" * (width - filled)
        label = f"{stage.upper()} PROGRESS: {state.done}/{state.total}"
        suffix = f" {detail}" if detail else ""
        if state.failed:
            suffix += f" failed={state.failed}"
        rendered = f"{label} [{bar}]{suffix}"
        if self.interactive:
            columns = shutil.get_terminal_size((100, 24)).columns
            rendered = rendered[: max(1, columns - 1)]
            if self.color:
                rendered = f"\x1b[{_COLORS.get(stage, '37')}m{rendered}\x1b[0m"
                self.stream.write(f"\r\x1b[2K{rendered}")
            else:
                self.stream.write("\r" + rendered.ljust(max(1, columns - 1)))
            self._live = True
            if state.done == state.total:
                self._end_line()
        else:
            self.stream.write(rendered + "\n")
        self.stream.flush()
