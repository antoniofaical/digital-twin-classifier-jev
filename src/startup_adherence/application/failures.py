"""Failure context shared by the batch runner and classification pipeline."""

from __future__ import annotations

import traceback
from dataclasses import dataclass


class StageError(RuntimeError):
    """Mark which external service or computation failed, preserving its cause."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        super().__init__(f"{type(cause).__name__}: {cause}")


@dataclass(frozen=True)
class JobFailure:
    stage: str
    error_type: str
    message: str
    traceback_text: str
    missing_result: bool = False

    @classmethod
    def from_exception(cls, exc: Exception, stage: str) -> JobFailure:
        if isinstance(exc, StageError):
            stage = exc.stage
            cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
        else:
            cause = exc
        return cls(
            stage=stage,
            error_type=type(cause).__name__,
            message=str(cause),
            traceback_text="".join(traceback.format_exception(exc)),
            missing_result=(
                isinstance(cause, FileNotFoundError)
                and str(cause) == "No completed run with matching profile questions"
            ),
        )
