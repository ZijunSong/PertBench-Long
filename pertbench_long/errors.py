"""Typed error codes used by the public protocol. Messages never include private paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class PertBenchLongError(Exception):
    error_code = "INTERNAL"
    retryable = False

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_code": self.error_code,
            "retryable": self.retryable,
            "message": self.message,
        }


class SchemaError(PertBenchLongError):
    error_code = "SCHEMA_INVALID"


class SplitLeakError(PertBenchLongError):
    error_code = "SPLIT_LEAK"


class UnavailableExperiment(PertBenchLongError):
    error_code = "UNAVAILABLE_EXPERIMENT"


class IdempotencyConflict(PertBenchLongError):
    error_code = "IDEMPOTENCY_CONFLICT"


class BudgetExceeded(PertBenchLongError):
    error_code = "BUDGET_EXCEEDED"


class InvalidState(PertBenchLongError):
    error_code = "INVALID_STATE"


class IsolationUnavailable(PertBenchLongError):
    error_code = "ISOLATION_UNAVAILABLE"


class PathGuardError(PertBenchLongError):
    error_code = "PATH_REJECTED"


class SubmissionInvalid(PertBenchLongError):
    error_code = "SUBMISSION_INVALID"


class PredictorRejected(PertBenchLongError):
    error_code = "PREDICTOR_REJECTED"


class UnsupportedProfile(PertBenchLongError):
    error_code = "UNSUPPORTED_PROFILE"


class LabelBuildError(PertBenchLongError):
    error_code = "LABEL_BUILD_REJECTED"


class IsolationViolation(PertBenchLongError):
    error_code = "ISOLATION_VIOLATION"


@dataclass
class ErrorPayload:
    error_code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error_code": self.error_code,
            "retryable": self.retryable,
            "message": self.message,
        }
