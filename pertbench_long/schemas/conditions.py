"""Versioned biological-condition identity. Float string concatenation is not an equivalence rule."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional, Sequence

from pertbench_long.errors import ConditionIdentityError, UnknownUnit
from pertbench_long.hashes import sha256_json

CONDITION_IDENTITY_VERSION = "cid_v1"

PERTURBATION_CHEMICAL = "chemical"
PERTURBATION_GENETIC_SINGLE = "genetic_single"
PERTURBATION_GENETIC_PAIR = "genetic_pair"
PERTURBATION_CONTROL = "control"
PERTURBATION_STIMULUS = "stimulus"
PERTURBATION_UNKNOWN = "unknown"

ALLOWED_PERTURBATION_KINDS = frozenset(
    {
        PERTURBATION_CHEMICAL,
        PERTURBATION_GENETIC_SINGLE,
        PERTURBATION_GENETIC_PAIR,
        PERTURBATION_CONTROL,
        PERTURBATION_STIMULUS,
        PERTURBATION_UNKNOWN,
    }
)

CONTEXT_CELL_LINE = "cell_line"
CONTEXT_PRIMARY = "primary_celltype"
CONTEXT_UNKNOWN = "unknown"
ALLOWED_CONTEXT_TYPES = frozenset({CONTEXT_CELL_LINE, CONTEXT_PRIMARY, CONTEXT_UNKNOWN})

# Exact molar conversions. Mass/volume units cannot join a molar identity without MW.
DOSE_TO_NM: dict[str, Fraction] = {
    "nM": Fraction(1),
    "uM": Fraction(1000),
    "mM": Fraction(1_000_000),
}
MASS_DOSE_UNITS = frozenset({"mg_ml", "ng_ml"})
TIME_TO_S: dict[str, Fraction] = {
    "s": Fraction(1),
    "min": Fraction(60),
    "h": Fraction(3600),
}


def _as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ConditionIdentityError("boolean is not a numeric dose/time")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ConditionIdentityError("dose/time must be finite")
        return Fraction(value).limit_denominator(1_000_000_000)
    text = str(value).strip()
    return Fraction(text)


def normalize_dose(value: Any, unit: str, *, require_exact: bool = False) -> dict[str, Any]:
    """Return a structured dose. Distinguishes missing, none, and unknown units."""
    unit = str(unit or "none")
    if value is None:
        if unit == "none":
            return {"status": "none", "canonical_nm": None, "original_value": None, "original_unit": "none"}
        if unit == "unknown":
            payload = {"status": "unknown_unit", "canonical_nm": None, "original_value": None, "original_unit": "unknown"}
            if require_exact:
                raise UnknownUnit("dose unit is unknown; exact-dose protocols refuse this condition")
            return payload
        if require_exact:
            raise ConditionIdentityError(f"dose value is missing but unit is {unit!r}")
        return {"status": "missing_value", "canonical_nm": None, "original_value": None, "original_unit": unit}
    number = _as_fraction(value)
    if unit == "none":
        raise ConditionIdentityError("a numeric dose cannot use unit=none")
    if unit == "unknown" or unit in MASS_DOSE_UNITS:
        if require_exact:
            raise UnknownUnit(f"dose unit {unit!r} cannot enter an exact molar identity")
        return {
            "status": "non_molar",
            "canonical_nm": None,
            "original_value": str(number),
            "original_unit": unit,
        }
    scale = DOSE_TO_NM.get(unit)
    if scale is None:
        raise UnknownUnit(f"unsupported dose unit {unit!r}")
    canonical = number * scale
    return {
        "status": "ok",
        "canonical_nm": str(canonical),
        "original_value": str(number),
        "original_unit": unit,
    }


def normalize_time(value: Any, unit: str, *, require_exact: bool = False) -> dict[str, Any]:
    unit = str(unit or "none")
    if value is None:
        if unit == "none":
            return {"status": "none", "canonical_s": None, "original_value": None, "original_unit": "none"}
        if unit == "unknown":
            if require_exact:
                raise UnknownUnit("time unit is unknown; exact-time protocols refuse this condition")
            return {"status": "unknown_unit", "canonical_s": None, "original_value": None, "original_unit": "unknown"}
        if require_exact:
            raise ConditionIdentityError(f"time value is missing but unit is {unit!r}")
        return {"status": "missing_value", "canonical_s": None, "original_value": None, "original_unit": unit}
    number = _as_fraction(value)
    if unit == "none":
        raise ConditionIdentityError("a numeric time cannot use unit=none")
    if unit == "unknown":
        if require_exact:
            raise UnknownUnit("time unit is unknown; exact-time protocols refuse this condition")
        return {"status": "unknown_unit", "canonical_s": None, "original_value": str(number), "original_unit": "unknown"}
    scale = TIME_TO_S.get(unit)
    if scale is None:
        raise UnknownUnit(f"unsupported time unit {unit!r}")
    return {
        "status": "ok",
        "canonical_s": str(number * scale),
        "original_value": str(number),
        "original_unit": unit,
    }


def parse_component_token(raw: str) -> tuple[str, ...]:
    text = str(raw).strip()
    if not text:
        raise ConditionIdentityError("empty perturbation component")
    for sep in ("+", "&", ",", ";", "|"):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                return tuple(parts)
    return (text,)


def normalize_components(
    components: Sequence[str] | str | None,
    *,
    kind: str,
    unordered_pairs: bool = True,
) -> tuple[str, ...]:
    if components is None:
        items: list[str] = []
    elif isinstance(components, str):
        items = list(parse_component_token(components))
    else:
        items = []
        for item in components:
            items.extend(parse_component_token(str(item)))
    cleaned = [part.strip() for part in items if str(part).strip()]
    if kind == PERTURBATION_GENETIC_PAIR:
        if len(cleaned) != 2:
            raise ConditionIdentityError(f"genetic_pair requires exactly two genes, got {cleaned}")
        if cleaned[0] == cleaned[1]:
            raise ConditionIdentityError("genetic_pair cannot be a self-pair")
        return tuple(sorted(cleaned)) if unordered_pairs else tuple(cleaned)
    if kind == PERTURBATION_GENETIC_SINGLE:
        if len(cleaned) != 1:
            raise ConditionIdentityError(f"genetic_single requires one gene, got {cleaned}")
        return (cleaned[0],)
    if kind == PERTURBATION_CONTROL:
        return tuple(cleaned) if cleaned else ("control",)
    if not cleaned:
        raise ConditionIdentityError("perturbation components are required for non-control conditions")
    return tuple(cleaned)


def condition_identity_payload(
    *,
    study: str,
    context_id: str,
    perturbation_kind: str,
    perturbation_components: Sequence[str],
    dose: Any = None,
    dose_unit: str = "none",
    time: Any = None,
    time_unit: str = "none",
    assay: str = "scrna",
    require_exact_dose: bool = False,
    require_exact_time: bool = False,
    identity_version: str = CONDITION_IDENTITY_VERSION,
) -> dict[str, Any]:
    if identity_version != CONDITION_IDENTITY_VERSION:
        raise ConditionIdentityError(f"unsupported condition identity version {identity_version!r}")
    if perturbation_kind not in ALLOWED_PERTURBATION_KINDS:
        raise ConditionIdentityError(f"illegal perturbation_kind {perturbation_kind!r}")
    dose_info = normalize_dose(dose, dose_unit, require_exact=require_exact_dose)
    time_info = normalize_time(time, time_unit, require_exact=require_exact_time)
    components = normalize_components(perturbation_components, kind=perturbation_kind)
    return {
        "identity_version": identity_version,
        "study": str(study),
        "context_id": str(context_id),
        "perturbation_kind": perturbation_kind,
        "perturbation_components": list(components),
        "dose_canonical_nm": dose_info.get("canonical_nm"),
        "time_canonical_s": time_info.get("canonical_s"),
        "assay": str(assay),
        "dose_status": dose_info["status"],
        "time_status": time_info["status"],
    }


def make_condition_id(
    *,
    study: str,
    context_id: str,
    perturbation_kind: str,
    perturbation_components: Sequence[str],
    dose: Any = None,
    dose_unit: str = "none",
    time: Any = None,
    time_unit: str = "none",
    assay: str = "scrna",
    require_exact_dose: bool = False,
    require_exact_time: bool = False,
    identity_version: str = CONDITION_IDENTITY_VERSION,
) -> str:
    payload = condition_identity_payload(
        study=study,
        context_id=context_id,
        perturbation_kind=perturbation_kind,
        perturbation_components=perturbation_components,
        dose=dose,
        dose_unit=dose_unit,
        time=time,
        time_unit=time_unit,
        assay=assay,
        require_exact_dose=require_exact_dose,
        require_exact_time=require_exact_time,
        identity_version=identity_version,
    )
    digest = sha256_json(
        {
            "identity_version": payload["identity_version"],
            "study": payload["study"],
            "context_id": payload["context_id"],
            "perturbation_kind": payload["perturbation_kind"],
            "perturbation_components": payload["perturbation_components"],
            "dose_canonical_nm": payload["dose_canonical_nm"],
            "time_canonical_s": payload["time_canonical_s"],
            "assay": payload["assay"],
        }
    )
    return f"{identity_version}:{digest}"


def doses_equivalent(value_a: Any, unit_a: str, value_b: Any, unit_b: str) -> bool:
    a = normalize_dose(value_a, unit_a, require_exact=True)
    b = normalize_dose(value_b, unit_b, require_exact=True)
    return a["canonical_nm"] == b["canonical_nm"]


def canonical_dose_nm(value: Any, unit: str) -> Fraction:
    info = normalize_dose(value, unit, require_exact=True)
    if info["canonical_nm"] is None:
        raise ConditionIdentityError("dose has no canonical molar value")
    return Fraction(info["canonical_nm"])


def canonical_time_s(value: Any, unit: str) -> Fraction:
    info = normalize_time(value, unit, require_exact=True)
    if info["canonical_s"] is None:
        raise ConditionIdentityError("time has no canonical second value")
    return Fraction(info["canonical_s"])


def display_perturbation(kind: str, components: Sequence[str]) -> str:
    if kind == PERTURBATION_GENETIC_PAIR:
        return "+".join(normalize_components(components, kind=kind))
    if not components:
        return kind
    return str(components[0]) if len(components) == 1 else "+".join(components)


def infer_genetic_kind(components: Sequence[str] | str) -> str:
    parts = normalize_components(components, kind=PERTURBATION_UNKNOWN) if not isinstance(components, str) else parse_component_token(components)
    if len(parts) == 2:
        return PERTURBATION_GENETIC_PAIR
    if len(parts) == 1:
        return PERTURBATION_GENETIC_SINGLE
    raise ConditionIdentityError(f"cannot infer genetic kind from {parts}")


def records_by_condition(records: Sequence[Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        key = rec.condition_id or rec.condition_key()
        out.setdefault(key, []).append(i)
    return out


def unique_condition_or_raise(matches: Mapping[str, Any] | Iterable[str], *, where: str) -> str:
    from pertbench_long.errors import AmbiguousCondition

    items = list(matches) if not isinstance(matches, Mapping) else list(matches)
    ids = sorted({str(x) for x in items})
    if not ids:
        raise AmbiguousCondition(f"{where}: no matching condition")
    if len(ids) > 1:
        raise AmbiguousCondition(f"{where}: selector matched {len(ids)} conditions; first-match is forbidden")
    return ids[0]
