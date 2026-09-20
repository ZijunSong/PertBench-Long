"""Schema validation with field-localized errors. No I/O side effects."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from pertbench_long.errors import SchemaError
from pertbench_long.schemas.types import (
    ALLOWED_ASSAYS,
    ALLOWED_COST_UNITS,
    ALLOWED_DOSE_UNITS,
    ALLOWED_LABEL_PROFILES,
    ALLOWED_MATRIX_KINDS,
    ALLOWED_PROTOCOLS,
    ALLOWED_SPECIES,
    ALLOWED_TIME_UNITS,
    PRIVATE_PUBLIC_FORBIDDEN_KEYS,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    CandidateExperiment,
    ExperimentRecord,
    PrivateEpisodeSpec,
    PublicEpisodeSpec,
    ResourceProfile,
    TargetDescription,
    dataclass_field_names,
)


class FieldError(SchemaError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}", details={"field": field})
        self.field = field


def _require_mapping(payload: Any, where: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FieldError(where, "expected object")
    return dict(payload)


def _require_str(payload: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    if key not in payload:
        raise FieldError(key, "missing required field")
    value = payload[key]
    if not isinstance(value, str):
        raise FieldError(key, "expected string")
    if not allow_empty and not value:
        raise FieldError(key, "must be non-empty")
    return value


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    if key not in payload:
        raise FieldError(key, "missing required field")
    value = payload[key]
    if not isinstance(value, bool):
        raise FieldError(key, "expected boolean")
    return value


def _require_int(payload: Mapping[str, Any], key: str, *, min_value: int | None = None) -> int:
    if key not in payload:
        raise FieldError(key, "missing required field")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldError(key, "expected integer")
    if min_value is not None and value < min_value:
        raise FieldError(key, f"must be >= {min_value}")
    return value


def _require_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    if key not in payload:
        raise FieldError(key, "missing required field")
    value = payload[key]
    if not isinstance(value, list):
        raise FieldError(key, "expected array")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise FieldError(f"{where}.{extra[0]}", f"unknown field {extra[0]!r}")


def _reject_nan(value: Any, field: str) -> None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise FieldError(field, "NaN/Inf not allowed")


def _unique(ids: Sequence[str], field: str) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise FieldError(field, f"duplicate id {item!r}")
        seen.add(item)


def validate_schema_version(version: str, field: str = "schema_version") -> None:
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise FieldError(field, f"unsupported schema_version {version!r}; supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}")


def parse_resource_profile(payload: Any) -> ResourceProfile:
    if isinstance(payload, str):
        if payload != "cpu_pilot_v1":
            raise FieldError("resource_profile", f"unknown named profile {payload!r}")
        return ResourceProfile()
    data = _require_mapping(payload, "resource_profile")
    _reject_unknown(data, dataclass_field_names(ResourceProfile), "resource_profile")
    return ResourceProfile(**{k: data[k] for k in dataclass_field_names(ResourceProfile) if k in data})


def parse_candidate(payload: Any, index: int) -> CandidateExperiment:
    data = _require_mapping(payload, f"candidate_experiments[{index}]")
    allowed = dataclass_field_names(CandidateExperiment)
    _reject_unknown(data, allowed, f"candidate_experiments[{index}]")
    cost = int(data.get("cost", 1))
    if cost != 1:
        raise FieldError(f"candidate_experiments[{index}].cost", "v1 requires unit cost 1")
    dose_unit = str(data.get("dose_unit", "none"))
    time_unit = str(data.get("time_unit", "none"))
    if dose_unit not in ALLOWED_DOSE_UNITS:
        raise FieldError(f"candidate_experiments[{index}].dose_unit", f"illegal unit {dose_unit!r}")
    if time_unit not in ALLOWED_TIME_UNITS:
        raise FieldError(f"candidate_experiments[{index}].time_unit", f"illegal unit {time_unit!r}")
    for key in ("dose", "time"):
        if key in data:
            _reject_nan(data[key], f"candidate_experiments[{index}].{key}")
    return CandidateExperiment(
        experiment_id=_require_str(data, "experiment_id"),
        cell_type=_require_str(data, "cell_type"),
        perturbation=_require_str(data, "perturbation"),
        cost=cost,
        dose=data.get("dose"),
        dose_unit=dose_unit,
        time=data.get("time"),
        time_unit=time_unit,
        description=str(data.get("description", "")),
    )


def parse_target(payload: Any, index: int) -> TargetDescription:
    data = _require_mapping(payload, f"targets[{index}]")
    _reject_unknown(data, dataclass_field_names(TargetDescription), f"targets[{index}]")
    dose_unit = str(data.get("dose_unit", "none"))
    time_unit = str(data.get("time_unit", "none"))
    if dose_unit not in ALLOWED_DOSE_UNITS:
        raise FieldError(f"targets[{index}].dose_unit", f"illegal unit {dose_unit!r}")
    if time_unit not in ALLOWED_TIME_UNITS:
        raise FieldError(f"targets[{index}].time_unit", f"illegal unit {time_unit!r}")
    return TargetDescription(
        target_id=_require_str(data, "target_id"),
        cell_type=_require_str(data, "cell_type"),
        perturbation=_require_str(data, "perturbation"),
        dose=data.get("dose"),
        dose_unit=dose_unit,
        time=data.get("time"),
        time_unit=time_unit,
    )


def validate_public_payload(payload: Mapping[str, Any]) -> PublicEpisodeSpec:
    data = _require_mapping(payload, "$")
    _reject_unknown(data, dataclass_field_names(PublicEpisodeSpec), "public")
    for forbidden in PRIVATE_PUBLIC_FORBIDDEN_KEYS:
        if forbidden in data:
            raise FieldError(forbidden, "private field is not allowed in public spec")
    version = _require_str(data, "schema_version")
    validate_schema_version(version)
    protocol = _require_str(data, "protocol")
    if protocol not in ALLOWED_PROTOCOLS:
        raise FieldError("protocol", f"unknown protocol {protocol!r}")
    label_profile = _require_str(data, "label_profile")
    if label_profile not in ALLOWED_LABEL_PROFILES:
        raise FieldError("label_profile", f"unknown label_profile {label_profile!r}")
    cost_unit = _require_str(data, "cost_unit")
    if cost_unit not in ALLOWED_COST_UNITS:
        raise FieldError("cost_unit", "v1 cost_unit must be 'credit'")
    candidates = [parse_candidate(item, i) for i, item in enumerate(_require_list(data, "candidate_experiments"))]
    targets = [parse_target(item, i) for i, item in enumerate(_require_list(data, "targets"))]
    if not targets:
        raise FieldError("targets", "at least one target is required")
    if not _require_list(data, "initial_evidence"):
        raise FieldError("initial_evidence", "at least one initial evidence artifact is required")
    if not _require_list(data, "reference_evidence"):
        raise FieldError("reference_evidence", "control/reference evidence must be listed explicitly")
    _unique([c.experiment_id for c in candidates], "candidate_experiments")
    _unique([t.target_id for t in targets], "targets")
    budget = _require_int(data, "experimental_budget", min_value=0)
    if budget > len(candidates):
        raise FieldError("experimental_budget", "budget cannot exceed number of candidate experiments")
    resource = data.get("resource_profile", "cpu_pilot_v1")
    parse_resource_profile(resource)
    public = PublicEpisodeSpec(
        schema_version=version,
        episode_id=_require_str(data, "episode_id"),
        protocol=protocol,
        protocol_version=_require_str(data, "protocol_version"),
        label_profile=label_profile,
        objective=_require_str(data, "objective"),
        initial_evidence=list(data["initial_evidence"]),
        reference_evidence=list(data["reference_evidence"]),
        candidate_experiments=candidates,
        targets=targets,
        target_control_available=_require_bool(data, "target_control_available"),
        gene_universe_artifact=_require_str(data, "gene_universe_artifact"),
        experimental_budget=budget,
        cost_unit=cost_unit,
        resource_profile=resource if isinstance(resource, str) else parse_resource_profile(resource),
        data_release_id=_require_str(data, "data_release_id"),
        public_split_fingerprint=_require_str(data, "public_split_fingerprint"),
        public_scoring_fingerprint=_require_str(data, "public_scoring_fingerprint"),
        reference_policy=str(data.get("reference_policy", "target_control_available_v1")),
        gene_panel_policy=str(data.get("gene_panel_policy", "episode_specific_from_O_plus_C")),
        canonical_units=dict(data.get("canonical_units") or {"effect": "log1p_mean_diff"}),
        artifact_metadata=dict(data.get("artifact_metadata") or {}),
        related_family=data.get("related_family"),
        synthetic=bool(data.get("synthetic", False)),
        notes=list(data.get("notes") or []),
    )
    scan_public_leak(public.to_dict())
    return public


def scan_public_leak(payload: Any, where: str = "public") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if key in PRIVATE_PUBLIC_FORBIDDEN_KEYS or lowered in PRIVATE_PUBLIC_FORBIDDEN_KEYS:
                raise FieldError(f"{where}.{key}", "private key leaked into public serialization")
            if "private" in lowered and key not in {"notes"}:
                text = str(value).lower()
                if "label" in text or "held-out outcome" in text:
                    raise FieldError(f"{where}.{key}", "private outcome information leaked")
            scan_public_leak(value, f"{where}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            scan_public_leak(item, f"{where}[{i}]")


def validate_experiment_record(payload: Mapping[str, Any], where: str = "record") -> ExperimentRecord:
    data = _require_mapping(payload, where)
    _reject_unknown(data, dataclass_field_names(ExperimentRecord), where)
    species = _require_str(data, "species")
    if species not in ALLOWED_SPECIES:
        raise FieldError(f"{where}.species", f"illegal species {species!r}")
    assay = _require_str(data, "assay")
    if assay not in ALLOWED_ASSAYS:
        raise FieldError(f"{where}.assay", f"illegal assay {assay!r}")
    dose_unit = str(data.get("dose_unit", "none"))
    time_unit = str(data.get("time_unit", "none"))
    matrix_kind = str(data.get("matrix_kind", "unknown"))
    if dose_unit not in ALLOWED_DOSE_UNITS:
        raise FieldError(f"{where}.dose_unit", f"illegal unit {dose_unit!r}")
    if time_unit not in ALLOWED_TIME_UNITS:
        raise FieldError(f"{where}.time_unit", f"illegal unit {time_unit!r}")
    if matrix_kind not in ALLOWED_MATRIX_KINDS:
        raise FieldError(f"{where}.matrix_kind", f"illegal matrix_kind {matrix_kind!r}")
    donor = data.get("donor")
    if donor is not None and not isinstance(donor, str):
        raise FieldError(f"{where}.donor", "donor must be string or null")
    rec = ExperimentRecord(
        observation_id=_require_str(data, "observation_id"),
        study=_require_str(data, "study"),
        species=species,
        cell_type=_require_str(data, "cell_type"),
        perturbation_id=_require_str(data, "perturbation_id"),
        dose=data.get("dose"),
        dose_unit=dose_unit,
        time=data.get("time"),
        time_unit=time_unit,
        assay=assay,
        observation_unit=str(data.get("observation_unit", "cell")),
        replicate_policy=str(data.get("replicate_policy", "fixed_bundle_v1")),
        original_obs_id=str(data.get("original_obs_id", "")),
        sample_id=data.get("sample_id"),
        donor=donor,
        condition_id=str(data.get("condition_id", "")),
        source_file=str(data.get("source_file", "")),
        matrix_kind=matrix_kind,
    )
    return rec


def validate_disjoint_roles(
    observed: Sequence[str],
    queryable: Sequence[str],
    targets: Sequence[str],
    *,
    layer: str,
) -> None:
    o, q, t = set(observed), set(queryable), set(targets)
    if o & q:
        raise FieldError(f"split.{layer}", f"O ∩ Q nonempty: {sorted(o & q)[:8]}")
    if o & t:
        raise FieldError(f"split.{layer}", f"O ∩ T nonempty: {sorted(o & t)[:8]}")
    if q & t:
        raise FieldError(f"split.{layer}", f"Q ∩ T nonempty: {sorted(q & t)[:8]}")


def validate_private_payload(payload: Mapping[str, Any]) -> PrivateEpisodeSpec:
    data = _require_mapping(payload, "private")
    allowed = dataclass_field_names(PrivateEpisodeSpec)
    _reject_unknown(data, allowed, "private")
    version = _require_str(data, "schema_version")
    validate_schema_version(version)
    public = validate_public_payload(data["public"] if "public" in data else data.get("public_spec"))
    o = list(_require_list(data, "observed_condition_ids"))
    q = list(_require_list(data, "queryable_condition_ids"))
    t = list(_require_list(data, "target_condition_ids"))
    c = list(_require_list(data, "control_condition_ids"))
    if not c:
        raise FieldError("control_condition_ids", "control/reference conditions must be listed")
    validate_disjoint_roles(o, q, t, layer="condition")
    roles = data.get("observation_ids_by_role") or {}
    if not isinstance(roles, Mapping):
        raise FieldError("observation_ids_by_role", "expected object")
    validate_disjoint_roles(
        list(roles.get("O", [])),
        list(roles.get("Q", [])),
        list(roles.get("T", [])),
        layer="observation",
    )
    matrix_kind = _require_str(data, "matrix_kind")
    if matrix_kind not in ALLOWED_MATRIX_KINDS:
        raise FieldError("matrix_kind", f"illegal matrix_kind {matrix_kind!r}")
    if matrix_kind == "unknown":
        raise FieldError("matrix_kind", "unknown matrices cannot enter official scoring")
    genes = list(_require_list(data, "gene_ids"))
    if not genes:
        raise FieldError("gene_ids", "gene universe G cannot be empty")
    _unique(genes, "gene_ids")
    return PrivateEpisodeSpec(
        schema_version=version,
        episode_id=_require_str(data, "episode_id"),
        public=public,
        observed_condition_ids=o,
        queryable_condition_ids=q,
        target_condition_ids=t,
        control_condition_ids=c,
        observation_ids_by_role={k: list(v) for k, v in roles.items()},
        experiment_id_to_condition=dict(data.get("experiment_id_to_condition") or {}),
        target_id_to_condition=dict(data.get("target_id_to_condition") or {}),
        label_artifact=_require_str(data, "label_artifact"),
        source_inventory_id=_require_str(data, "source_inventory_id"),
        split_config=dict(data.get("split_config") or {}),
        scoring_config=dict(data.get("scoring_config") or {}),
        private_data_hash=_require_str(data, "private_data_hash"),
        development_group=_require_str(data, "development_group"),
        evaluation_group=_require_str(data, "evaluation_group"),
        gene_ids=genes,
        gene_order_hash=_require_str(data, "gene_order_hash"),
        matrix_kind=matrix_kind,
        donor_metadata_present=bool(data.get("donor_metadata_present", False)),
        provenance=dict(data.get("provenance") or {}),
    )
