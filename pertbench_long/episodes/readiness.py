"""Split implementation status, data status, and official scoring eligibility."""

from __future__ import annotations

from typing import Any, Mapping

from pertbench_long.errors import UnsupportedProfile

UNRELEASED_RELEASE_IDS = frozenset({"local_unreleased", "unreleased", ""})
FIXTURE_A_FAMILY = 0.5


def resolve_scoring_eligibility(
    *,
    synthetic: bool,
    matrix_kind: str,
    data_release_id: str,
    requested_track: str = "pilot",
    provenance_verified: bool = False,
    source_verified: bool = False,
    split_audited: bool = False,
    labels_validated: bool = False,
    scoring_scale_hash: str | None = None,
) -> dict[str, Any]:
    """Return track/readiness. not-synthetic + counts is never enough for official."""
    if matrix_kind in {"unknown", "scaled"}:
        track = "diagnostic"
    elif synthetic:
        track = "pilot"
    elif requested_track == "official":
        missing = []
        if str(data_release_id or "") in UNRELEASED_RELEASE_IDS:
            missing.append("frozen_data_release_id")
        if not provenance_verified:
            missing.append("verified_provenance")
        if not source_verified:
            missing.append("source_verified")
        if not split_audited:
            missing.append("split_audited")
        if not labels_validated:
            missing.append("labels_validated")
        if missing:
            raise UnsupportedProfile(
                "official track requires frozen release evidence; "
                f"missing={missing}; local_unreleased cannot be official"
            )
        track = "official"
    else:
        track = "pilot"

    readiness = track
    if not synthetic and str(data_release_id or "") in UNRELEASED_RELEASE_IDS:
        readiness = "pilot"

    scale_source = "declared_fixed_scale" if scoring_scale_hash else "fixture_default_not_calibrated"
    if track == "official" and not scoring_scale_hash:
        raise UnsupportedProfile("official continuous scoring requires a frozen scoring_scale_hash")

    return {
        "scoring_track": track,
        "readiness": readiness,
        "label_validity": "ok" if track != "diagnostic" else "diagnostic",
        "protocol_implemented": True,
        "source_verified": bool(source_verified and not synthetic),
        "labels_validated": bool(labels_validated),
        "split_audited": bool(split_audited),
        "episode_runnable": True,
        "isolation_qualified": False,
        "live_model_tested": False,
        "scale_source": scale_source,
        "scoring_scale_hash": scoring_scale_hash,
        "a_family_note": "a_family=0.5 is a fixture/pilot default, not a completed scientific calibration"
        if scoring_scale_hash is None
        else "a_family taken from a frozen calibration artifact",
    }


def eligibility_notes(info: Mapping[str, Any]) -> list[str]:
    notes = [info["a_family_note"], f"scale_source={info['scale_source']}"]
    if not info.get("source_verified"):
        notes.append("source_unverified")
    return notes
