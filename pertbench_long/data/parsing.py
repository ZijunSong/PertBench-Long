"""Strict metadata parsing. Generic bool() and implicit nM/h defaults are forbidden."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pertbench_long.errors import SchemaError, UnknownUnit
from pertbench_long.schemas.conditions import DOSE_TO_NM, MASS_DOSE_UNITS, TIME_TO_S

CONTROL_NAME_TOKENS = frozenset(
    {"control", "vehicle", "dmso", "untreated", "ntc", "non-targeting", "negctrl", "neg", "dummy"}
)
NTC_TOKENS = frozenset({"control", "ntc", "non-targeting", "negctrl", "neg", "dummy"})

_TRUE = frozenset({"true", "1"})
_FALSE = frozenset({"false", "0"})
_MISSING = frozenset({"", "nan", "none", "null", "<na>", "na"})


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in _MISSING:
        return True
    return False


def parse_bool(value: Any, *, field: str = "flag") -> bool | None:
    """Accept only declared bool / 0 / 1 / true / false. Missing returns None; unknown raises."""
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int,)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    try:
        import numpy as np

        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)) and int(value) in (0, 1):
            return bool(int(value))
    except Exception:
        pass
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SchemaError(f"{field} has unsupported boolean value {value!r}; only true/false/0/1 are accepted")


def parse_declared_unit(value: Any, *, kind: str, required: bool) -> str:
    """Return a unit token. Empty is 'none' only when the corresponding value is absent."""
    if is_missing(value):
        if required:
            raise UnknownUnit(f"{kind} unit is missing and cannot be assumed")
        return "none"
    unit = str(value).strip()
    if not unit or unit.lower() in _MISSING:
        if required:
            raise UnknownUnit(f"{kind} unit is missing and cannot be assumed")
        return "none"
    return unit


def require_dose_pair(dose: Any, unit: Any, *, require_exact: bool) -> tuple[Any, str]:
    if is_missing(dose):
        if require_exact:
            raise UnknownUnit("exact-dose protocols refuse a condition with no dose")
        return None, "none"
    unit_s = parse_declared_unit(unit, kind="dose", required=True)
    if unit_s in {"none", "unknown"} or unit_s in MASS_DOSE_UNITS:
        if require_exact:
            raise UnknownUnit(f"dose unit {unit_s!r} cannot enter an exact molar identity")
        raise UnknownUnit(f"dose unit {unit_s!r} is not a declared molar unit")
    if unit_s not in DOSE_TO_NM:
        raise UnknownUnit(f"unsupported dose unit {unit_s!r}")
    return dose, unit_s


def require_time_pair(time: Any, unit: Any, *, required: bool = False) -> tuple[Any, str]:
    if is_missing(time):
        if required:
            raise UnknownUnit("time value is required")
        return None, "none"
    unit_s = parse_declared_unit(unit, kind="time", required=True)
    if unit_s == "unknown":
        raise UnknownUnit("time unit is unknown")
    if unit_s not in TIME_TO_S:
        raise UnknownUnit(f"unsupported time unit {unit_s!r}")
    return time, unit_s


def require_matrix_kind(declared: str | None) -> str:
    if not declared or str(declared).strip() in {"", "none"}:
        raise SchemaError("matrix_kind must be declared; counts/nM/h are not assumed")
    kind = str(declared).strip()
    allowed = {"counts", "log1p", "normalized", "scaled", "unknown"}
    if kind not in allowed:
        raise SchemaError(f"unsupported matrix_kind {kind!r}")
    return kind


def looks_like_control_name(name: str | None) -> bool:
    if not name:
        return False
    return str(name).strip().lower() in CONTROL_NAME_TOKENS
