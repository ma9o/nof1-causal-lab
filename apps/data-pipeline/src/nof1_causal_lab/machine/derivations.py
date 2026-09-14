"""Deterministic derivation cascade for machine-maintained artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.graph import Derivation, topological_derivation_order, transition_spec
from nof1_causal_lab.machine.moves import (
    RetractedArtifact,
    TransitionEffects,
    apply_transition,
    is_stale,
    run_retractions,
)

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, OperationId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore


def complete_computed_transition(
    store: ArtifactStore,
    state: EpisodeState,
    transition_id: OperationId,
    produced: list[ArtifactVersionInfo],
) -> TransitionEffects:
    """Apply optional-output retractions and the derivation cascade for one run."""
    retracted = run_retractions(state, transition_spec(transition_id), produced)
    return complete_derivation_cascade(store, state, produced, retracted)


def complete_derivation_cascade(
    store: ArtifactStore,
    state: EpisodeState,
    produced: list[ArtifactVersionInfo],
    retracted: list[RetractedArtifact] | None = None,
) -> TransitionEffects:
    """Apply initial effects to a temporary state, then derive reachable nodes."""
    all_produced = list(produced)
    all_retracted = list(retracted or [])
    next_state = apply_transition(state, all_produced, all_retracted)
    affected: set[ArtifactId] = {info.artifact_id for info in produced} | {
        item.artifact_id for item in all_retracted
    }

    try:
        for spec in topological_derivation_order():
            parents = _current_parent_versions(next_state, spec)
            if parents is None:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=f"{spec.produces}.parents_absent",
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            stale_parents = [parent for parent in spec.from_ if is_stale(next_state, parent)]
            if stale_parents:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=f"{spec.produces}.parents_stale.{stale_parents[0]}",
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            if next_state.matches_inputs(spec.produces, *spec.from_):
                continue

            if not affected.intersection(spec.from_):
                continue

            info = _derive_one(store, spec, parents)
            if info is None:
                retraction = _retract_current(
                    next_state,
                    spec.produces,
                    reason_ref=_empty_finding_reason(spec.produces),
                )
                if retraction is not None:
                    all_retracted.append(retraction)
                    next_state = next_state.without([spec.produces])
                    affected.add(spec.produces)
                continue

            all_produced.append(info)
            next_state = next_state.with_versions([info])
            affected.add(spec.produces)
    except Exception:
        for info in reversed(all_produced):
            store.delete_version(info.artifact_id, info.version)
        raise

    return TransitionEffects(produced=all_produced, retracted=all_retracted)


def _retract_current(
    state: EpisodeState,
    artifact_id: ArtifactId,
    *,
    reason_ref: str,
) -> RetractedArtifact | None:
    if not state.has(artifact_id):
        return None
    return RetractedArtifact(artifact_id=artifact_id, reason_ref=reason_ref)


def _current_parent_versions(
    state: EpisodeState,
    spec: Derivation,
) -> dict[ArtifactId, int] | None:
    pins: dict[ArtifactId, int] = {}
    for parent in spec.from_:
        info = state.get(parent)
        if info is None:
            return None
        pins[parent] = info.version
    return pins


def _derive_one(
    store: ArtifactStore,
    spec: Derivation,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo | None:
    if spec.produces == "identification_report":
        return _derive_identification_report(store, pins)
    if spec.produces == "validation_report":
        return _derive_validation_report(store, pins)
    raise AssertionError(f"No derivation body for {spec.produces}")


def read_model(store: ArtifactStore, version: int) -> ModelSpec:
    from functools import cache

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    return ModelSpec.model_validate(
        store.read_json_file("model", version, json_filename("model", "model")),
        context={"distribution_array_loader": cache(store.read_array)},
    )


def _read_panel(store: ArtifactStore, version: int) -> pl.DataFrame:
    return store.read_parquet_file("panel", version, parquet_filename("panel", "panel"))


def _derive_identification_report(
    store: ArtifactStore, pins: dict[ArtifactId, int]
) -> ArtifactVersionInfo:
    from nof1_causal_lab.models.identification import identify_model

    report = identify_model(read_model(store, pins["model"]))
    return store.write_version(
        "identification_report",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:identification_report",
        json_files={
            json_filename("identification_report", "identification_report"): report.model_dump(
                mode="json"
            )
        },
    )


def _derive_validation_report(
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
) -> ArtifactVersionInfo:
    from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
    from nof1_causal_lab.flows.transitions.validation.flow import (
        derive_validation_status,
        validate_extraction,
    )

    model = read_model(store, pins["model"])
    panel = _read_panel(store, pins["panel"])
    audit_result = validate_extraction(model, [panel])
    if not audit_result:
        raise RuntimeError(
            "validation_report derivation returned an empty audit result; "
            "refusing to fabricate an is_valid=False report with empty indicators."
        )

    indicator_issues = [
        issue for audit in audit_result.get("indicators", {}).values() for issue in audit["issues"]
    ]
    dataset_issues = audit_result.get("dataset_issues", [])
    status = derive_validation_status([*indicator_issues, *dataset_issues])
    payload = ValidationReportArtifact.model_validate(
        {**audit_result, "is_valid": status["is_valid"]}
    ).model_dump(mode="json")
    return store.write_version(
        "validation_report",
        provenance="computed",
        derived_from=pins,
        produced_by="derive:validation_report",
        json_files={json_filename("validation_report", "validation_report"): payload},
    )


def _empty_finding_reason(artifact_id: ArtifactId) -> str:
    return f"{artifact_id}.model_requirements_missing"
