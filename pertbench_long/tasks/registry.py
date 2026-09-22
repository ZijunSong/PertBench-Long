"""Task family registry. A protocol is implemented only when all required hooks exist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pertbench_long.errors import UnsupportedProtocol
from pertbench_long.schemas.types import (
    IMPLEMENTED_PROTOCOLS,
    LABEL_EFFECT_PROXY_V1,
    LABEL_LOGNORM_CELLMEAN_DELTA_V1,
    PROTOCOL_CHEMICAL_DOSE,
    PROTOCOL_CONTEXT_CAMPAIGN,
    PROTOCOL_GENETIC_PAIR,
    PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    SCORING_CONTINUOUS_AUBC_V1,
    SCORING_DIRECTION_V1,
    TASK_FAMILY_CELLTYPE_OOD,
    TASK_FAMILY_CHEMICAL_DOSE,
    TASK_FAMILY_CONTEXT_CAMPAIGN,
    TASK_FAMILY_GENETIC_PAIR,
    UNSUPPORTED_PROTOCOLS,
)


@dataclass(frozen=True)
class TaskRegistration:
    task_family: str
    protocol: str
    schema_version: str
    builder: Callable[..., Any]
    split_validator: Callable[..., Any]
    label_profile: str
    submission_validator: Callable[..., Any]
    scorer: Callable[..., Any]
    baseline_factory: Callable[..., Any]
    scoring_profile: str
    implemented: bool = True


def _pbmc_builder(**kwargs):
    from pertbench_long.episodes.builder import build_episode_from_store

    return build_episode_from_store(**kwargs)


def _chemical_builder(**kwargs):
    from pertbench_long.episodes.builders.chemical_dose import build_chemical_dose_episode

    return build_chemical_dose_episode(**kwargs)


def _pair_builder(**kwargs):
    from pertbench_long.episodes.builders.genetic_pair import build_genetic_pair_episode

    return build_genetic_pair_episode(**kwargs)


def _context_builder(**kwargs):
    from pertbench_long.episodes.builders.context_campaign import build_context_campaign_episode

    return build_context_campaign_episode(**kwargs)


def _pbmc_split(*args, **kwargs):
    from pertbench_long.episodes.split import audit_split

    return audit_split(*args, **kwargs)


def _dose_split(*args, **kwargs):
    from pertbench_long.episodes.split import audit_chemical_dose_split

    return audit_chemical_dose_split(*args, **kwargs)


def _pair_split(*args, **kwargs):
    from pertbench_long.episodes.split import audit_genetic_pair_split

    return audit_genetic_pair_split(*args, **kwargs)


def _context_split(*args, **kwargs):
    from pertbench_long.episodes.split import audit_context_campaign_split

    return audit_context_campaign_split(*args, **kwargs)


def _proxy_submit(*args, **kwargs):
    from pertbench_long.evaluation.submission import validate_predictions

    return validate_predictions(*args, require_direction=True, **kwargs)


def _continuous_submit(*args, **kwargs):
    from pertbench_long.evaluation.submission import validate_predictions

    return validate_predictions(*args, require_direction=False, **kwargs)


def _proxy_score(*args, **kwargs):
    from pertbench_long.evaluation.scoring import score_predictions

    return score_predictions(*args, **kwargs)


def _continuous_score(*args, **kwargs):
    from pertbench_long.evaluation.scoring import score_continuous_predictions

    return score_continuous_predictions(*args, **kwargs)


def _pbmc_baselines():
    from pertbench_long.baselines.engine import pbmc_baselines

    return pbmc_baselines()


def _dose_baselines():
    from pertbench_long.baselines.engine import chemical_dose_baselines

    return chemical_dose_baselines()


def _pair_baselines():
    from pertbench_long.baselines.engine import genetic_pair_baselines

    return genetic_pair_baselines()


def _context_baselines():
    from pertbench_long.baselines.engine import context_campaign_baselines

    return context_campaign_baselines()


_REGISTRY: dict[str, TaskRegistration] = {
    PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD: TaskRegistration(
        task_family=TASK_FAMILY_CELLTYPE_OOD,
        protocol=PROTOCOL_WITHIN_STUDY_CELLTYPE_OOD,
        schema_version=SCHEMA_VERSION,
        builder=_pbmc_builder,
        split_validator=_pbmc_split,
        label_profile=LABEL_EFFECT_PROXY_V1,
        submission_validator=_proxy_submit,
        scorer=_proxy_score,
        baseline_factory=_pbmc_baselines,
        scoring_profile=SCORING_DIRECTION_V1,
    ),
    PROTOCOL_CHEMICAL_DOSE: TaskRegistration(
        task_family=TASK_FAMILY_CHEMICAL_DOSE,
        protocol=PROTOCOL_CHEMICAL_DOSE,
        schema_version=SCHEMA_VERSION_V2,
        builder=_chemical_builder,
        split_validator=_dose_split,
        label_profile=LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        submission_validator=_continuous_submit,
        scorer=_continuous_score,
        baseline_factory=_dose_baselines,
        scoring_profile=SCORING_CONTINUOUS_AUBC_V1,
    ),
    PROTOCOL_GENETIC_PAIR: TaskRegistration(
        task_family=TASK_FAMILY_GENETIC_PAIR,
        protocol=PROTOCOL_GENETIC_PAIR,
        schema_version=SCHEMA_VERSION_V2,
        builder=_pair_builder,
        split_validator=_pair_split,
        label_profile=LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        submission_validator=_continuous_submit,
        scorer=_continuous_score,
        baseline_factory=_pair_baselines,
        scoring_profile=SCORING_CONTINUOUS_AUBC_V1,
    ),
    PROTOCOL_CONTEXT_CAMPAIGN: TaskRegistration(
        task_family=TASK_FAMILY_CONTEXT_CAMPAIGN,
        protocol=PROTOCOL_CONTEXT_CAMPAIGN,
        schema_version=SCHEMA_VERSION_V2,
        builder=_context_builder,
        split_validator=_context_split,
        label_profile=LABEL_LOGNORM_CELLMEAN_DELTA_V1,
        submission_validator=_continuous_submit,
        scorer=_continuous_score,
        baseline_factory=_context_baselines,
        scoring_profile=SCORING_CONTINUOUS_AUBC_V1,
    ),
}


def list_tasks() -> dict[str, TaskRegistration]:
    return dict(_REGISTRY)


def implemented_protocols() -> frozenset[str]:
    return frozenset(name for name, spec in _REGISTRY.items() if spec.implemented)


def get_task(protocol: str) -> TaskRegistration:
    if protocol in UNSUPPORTED_PROTOCOLS:
        raise UnsupportedProtocol(f"protocol {protocol!r} is named but not implemented")
    spec = _REGISTRY.get(protocol)
    if spec is None:
        raise UnsupportedProtocol(f"unknown protocol {protocol!r}")
    if not spec.implemented:
        raise UnsupportedProtocol(f"protocol {protocol!r} is registered but not implemented")
    if protocol not in IMPLEMENTED_PROTOCOLS:
        raise UnsupportedProtocol(f"protocol {protocol!r} is missing from IMPLEMENTED_PROTOCOLS")
    return spec


def require_implemented(protocol: str) -> TaskRegistration:
    return get_task(protocol)
