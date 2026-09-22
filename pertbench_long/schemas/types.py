"""Public and private episode schemas. Validator creates no directories and does not read test data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Optional

SCHEMA_VERSION = "1.0"
SCHEMA_VERSION_V2 = "2.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_V2})

PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD = "within_study_celltype_ood_v1"
PROTOCOL_DONOR_HELDOUT = "donor_heldout_v1"
PROTOCOL_STUDY_HELDOUT = "study_heldout_v1"
PROTOCOL_MOA_HELDOUT = "moa_heldout_v1"
PROTOCOL_TEMPORAL_INTERP = "temporal_interpolation_v1"
PROTOCOL_CHEMICAL_DOSE = "chemical_dose_acquisition_v1"
PROTOCOL_GENETIC_PAIR = "genetic_pair_acquisition_v1"
PROTOCOL_CONTEXT_CAMPAIGN = "context_campaign_acquisition_v1"
PROTOCOL_PROGRAM_RANK = "program_rank_acquisition_v1"
PROTOCOL_TEMPORAL_FORECAST = "temporal_forecasting_v1"

ALLOWED_PROTOCOLS = frozenset(
    {
        PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
        PROTOCOL_DONOR_HELDOUT,
        PROTOCOL_STUDY_HELDOUT,
        PROTOCOL_MOA_HELDOUT,
        PROTOCOL_TEMPORAL_INTERP,
        PROTOCOL_CHEMICAL_DOSE,
        PROTOCOL_GENETIC_PAIR,
        PROTOCOL_CONTEXT_CAMPAIGN,
        PROTOCOL_PROGRAM_RANK,
        PROTOCOL_TEMPORAL_FORECAST,
    }
)
# A protocol is implemented only when registry builder/split/label/scorer/baseline all exist.
IMPLEMENTED_PROTOCOLS = frozenset(
    {
        PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
        PROTOCOL_CHEMICAL_DOSE,
        PROTOCOL_GENETIC_PAIR,
        PROTOCOL_CONTEXT_CAMPAIGN,
    }
)
UNSUPPORTED_PROTOCOLS = ALLOWED_PROTOCOLS - IMPLEMENTED_PROTOCOLS

LABEL_EFFECT_PROXY_V1 = "effect_proxy_v1"
LABEL_REPLICATE_DE_V1 = "replicate_de_v1"
LABEL_LOGNORM_CELLMEAN_DELTA_V1 = "lognorm_cellmean_delta_v1"
ALLOWED_LABEL_PROFILES = frozenset({LABEL_EFFECT_PROXY_V1, LABEL_REPLICATE_DE_V1, LABEL_LOGNORM_CELLMEAN_DELTA_V1})

SCORING_DIRECTION_V1 = "direction_score_v1"
SCORING_CONTINUOUS_AUBC_V1 = "continuous_effect_aubc_v1"
ALLOWED_SCORING_PROFILES = frozenset({SCORING_DIRECTION_V1, SCORING_CONTINUOUS_AUBC_V1})

TASK_FAMILY_CELLTYPE_OOD = "celltype_ood"
TASK_FAMILY_CHEMICAL_DOSE = "chemical_dose"
TASK_FAMILY_GENETIC_PAIR = "genetic_pair"
TASK_FAMILY_CONTEXT_CAMPAIGN = "context_campaign"

ALLOWED_SCORING_TRACKS = frozenset({"official", "diagnostic", "pilot"})
ALLOWED_READINESS = frozenset({"official", "diagnostic", "pilot"})
COST_POLICY_UNIT_V1 = "unit_cost_v1"
NAMED_RESOURCE_PROFILES = frozenset({"cpu_pilot_v1", "long_cpu_v1"})

ALLOWED_COST_UNITS = frozenset({"credit"})
ALLOWED_MATRIX_KINDS = frozenset({"counts", "normalized", "log1p", "scaled", "unknown"})
ALLOWED_DOSE_UNITS = frozenset({"nM", "uM", "mM", "mg_ml", "ng_ml", "none", "unknown"})
ALLOWED_TIME_UNITS = frozenset({"h", "min", "s", "none", "unknown"})
ALLOWED_ASSAYS = frozenset({"scrna", "bulk_rna", "unknown"})
ALLOWED_SPECIES = frozenset({"human", "mouse", "rat", "rabbit", "pig", "synthetic", "unknown"})

PRIVATE_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "label_path",
        "private_root",
        "target_effects",
        "hidden_mapping",
        "t_observation_ids",
        "private_data_hash",
        "label_artifact",
        "source_test_path",
    }
)


def _as_plain(obj: Any) -> Any:
    if dataclasses_is_dataclass(obj):
        return {k: _as_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v) for v in obj]
    return obj


def dataclasses_is_dataclass(obj: Any) -> bool:
    try:
        from dataclasses import is_dataclass

        return is_dataclass(obj) and not isinstance(obj, type)
    except Exception:
        return False


@dataclass(frozen=True)
class ResourceProfile:
    name: str = "cpu_pilot_v1"
    cpu: int = 2
    memory_gib: int = 8
    workspace_gib: int = 1
    tool_timeout_s: int = 60
    episode_timeout_s: int = 600
    max_external_tool_calls: int = 80
    max_generation_tokens: int = 32000
    max_output_bytes: int = 50 * 1024 * 1024


def named_resource_profile(name: str) -> ResourceProfile:
    if name == "cpu_pilot_v1":
        return ResourceProfile()
    if name == "long_cpu_v1":
        return ResourceProfile(
            name="long_cpu_v1",
            cpu=4,
            memory_gib=16,
            workspace_gib=2,
            tool_timeout_s=120,
            episode_timeout_s=1800,
            max_external_tool_calls=240,
            max_generation_tokens=64000,
        )
    raise ValueError(f"unknown named profile {name!r}")


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    relative_path: str
    sha256: Optional[str] = None
    media_type: str = "application/x-hdf5"
    bytes: Optional[int] = None


@dataclass(frozen=True)
class CandidateExperiment:
    experiment_id: str
    cell_type: str
    perturbation: str
    cost: int = 1
    dose: Optional[float] = None
    dose_unit: str = "none"
    time: Optional[float] = None
    time_unit: str = "none"
    description: str = ""
    context_id: str = ""
    context_type: str = "unknown"
    condition_id: str = ""
    perturbation_kind: str = "unknown"
    perturbation_components: tuple[str, ...] = ()
    control_group_id: Optional[str] = None


@dataclass(frozen=True)
class TargetDescription:
    target_id: str
    cell_type: str
    perturbation: str
    dose: Optional[float] = None
    dose_unit: str = "none"
    time: Optional[float] = None
    time_unit: str = "none"
    context_id: str = ""
    context_type: str = "unknown"
    condition_id: str = ""
    perturbation_kind: str = "unknown"
    perturbation_components: tuple[str, ...] = ()
    control_group_id: Optional[str] = None


@dataclass(frozen=True)
class ExperimentRecord:
    observation_id: str
    study: str
    species: str
    cell_type: str
    perturbation_id: str
    dose: Optional[float]
    dose_unit: str
    time: Optional[float]
    time_unit: str
    assay: str
    observation_unit: str = "cell"
    replicate_policy: str = "fixed_bundle_v1"
    original_obs_id: str = ""
    sample_id: Optional[str] = None
    donor: Optional[str] = None
    condition_id: str = ""
    source_file: str = ""
    matrix_kind: str = "unknown"
    context_id: str = ""
    context_type: str = "unknown"
    perturbation_kind: str = "unknown"
    perturbation_components: tuple[str, ...] = ()
    replicate_id: Optional[str] = None
    batch_id: Optional[str] = None
    donor_id: Optional[str] = None
    control_group_id: Optional[str] = None
    identity_version: str = ""
    missing_identity_reason: Optional[str] = None

    def condition_key(self) -> str:
        if self.condition_id:
            return self.condition_id
        # Legacy PBMC identity. New tasks must set condition_id via cid_v1.
        dose = "NA" if self.dose is None else f"{self.dose}{self.dose_unit}"
        time = "NA" if self.time is None else f"{self.time}{self.time_unit}"
        return "|".join(
            [
                self.study,
                self.species,
                self.cell_type,
                self.perturbation_id,
                dose,
                time,
                self.assay,
            ]
        )

    def resolved_context_id(self) -> str:
        return self.context_id or self.cell_type

    def resolved_components(self) -> tuple[str, ...]:
        if self.perturbation_components:
            return tuple(self.perturbation_components)
        if self.perturbation_id:
            return (self.perturbation_id,)
        return ()


@dataclass
class PublicEpisodeSpec:
    schema_version: str
    episode_id: str
    protocol: str
    protocol_version: str
    label_profile: str
    objective: str
    initial_evidence: list[str]
    reference_evidence: list[str]
    candidate_experiments: list[CandidateExperiment]
    targets: list[TargetDescription]
    target_control_available: bool
    gene_universe_artifact: str
    experimental_budget: int
    cost_unit: str
    resource_profile: str | ResourceProfile
    data_release_id: str
    public_split_fingerprint: str
    public_scoring_fingerprint: str
    reference_policy: str = "target_control_available_v1"
    gene_panel_policy: str = "episode_specific_from_O_plus_C"
    canonical_units: dict[str, str] = field(default_factory=lambda: {"effect": "log1p_mean_diff"})
    artifact_metadata: dict[str, Any] = field(default_factory=dict)
    related_family: Optional[str] = None
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)
    task_family: str = ""
    split_variant: str = ""
    scoring_profile: str = SCORING_DIRECTION_V1
    cost_policy: str = COST_POLICY_UNIT_V1
    readiness: str = "diagnostic"
    data_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _as_plain(self)
        return payload


@dataclass
class PrivateEpisodeSpec:
    schema_version: str
    episode_id: str
    public: PublicEpisodeSpec
    observed_condition_ids: list[str]
    queryable_condition_ids: list[str]
    target_condition_ids: list[str]
    control_condition_ids: list[str]
    observation_ids_by_role: dict[str, list[str]]
    experiment_id_to_condition: dict[str, str]
    target_id_to_condition: dict[str, str]
    label_artifact: str
    source_inventory_id: str
    split_config: dict[str, Any]
    scoring_config: dict[str, Any]
    private_data_hash: str
    development_group: str
    evaluation_group: str
    gene_ids: list[str]
    gene_order_hash: str
    matrix_kind: str
    donor_metadata_present: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    control_mapping: dict[str, str] = field(default_factory=dict)
    readiness: str = "diagnostic"
    label_validity: str = "unknown"
    protocol_implementation: str = "implemented"

    def to_dict(self) -> dict[str, Any]:
        return _as_plain(self)


@dataclass
class Claim:
    claim_id: str
    scope: dict[str, Any]
    statement: str
    evidence_ids: list[str]
    prediction_entries: list[dict[str, str]]
    limitations: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)


@dataclass
class FinalSubmission:
    schema_version: str
    episode_id: str
    evidence_version: int
    prediction_artifact: str
    claim_artifact: str
    limitations: list[str] = field(default_factory=list)
    stop_reason: str = "submitted"


def dataclass_field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}
