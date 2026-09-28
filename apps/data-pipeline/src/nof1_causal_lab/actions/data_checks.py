"""Data-only checks and explicit model/observation schema binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.execution import RetractedArtifact, TransitionEffects, apply_transition
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.machine.artifacts import EpisodeState


def read_data_metadata(store: ArtifactStore, revision: GitOid) -> PreparedDataMetadata:
    return PreparedDataMetadata.model_validate(
        store.read_json_file("panel", revision, json_filename("panel", "metadata"))
    )


def data_binding_issues(model: ModelSpec, metadata: PreparedDataMetadata) -> list[str]:
    """Compare scientific observation semantics, independently of generating-model ancestry."""
    variables = {item.id: item for item in metadata.variables}
    issues = []
    for indicator in model.indicators:
        variable = variables.get(indicator.id)
        if variable is None:
            issues.append(f"No prepared variable for model indicator {indicator.id}")
            continue
        for field, expected, actual in (
            ("measurement_dtype", indicator.measurement_dtype, variable.measurement_dtype),
            ("aggregation", indicator.aggregation, variable.aggregation),
            ("ordinal_levels", indicator.ordinal_levels, variable.ordinal_levels),
            ("categorical_levels", indicator.categorical_levels, variable.categorical_levels),
        ):
            if expected != actual:
                issues.append(f"Observation {indicator.id} has incompatible {field}")
        window = indicator.observation_window or model.measurement_clock
        assert (
            variable.observation_window is not None
        )  # PreparedDataMetadata resolves every window.
        if window is None or parse_duration_to_hours(window) != parse_duration_to_hours(
            variable.observation_window
        ):
            issues.append(f"Observation {indicator.id} has an incompatible observation window")
    return issues


def require_data_binding(store: ArtifactStore, model: ModelSpec, revision: GitOid) -> None:
    issues = data_binding_issues(model, read_data_metadata(store, revision))
    if issues:
        raise ValueError("Selected model and observations are incompatible: " + "; ".join(issues))


def evaluate_data_checks(
    workspace_id: str, state: EpisodeState, effects: TransitionEffects
) -> TransitionEffects:
    """Profile newly prepared observations without reading or evaluating any model."""
    from nof1_causal_lab.flows.transitions.validation.flow import profile_data

    store = ArtifactStore(workspace_id)
    selected = apply_transition(state, effects.produced, effects.retracted)
    panel = selected.get("panel")
    if panel is None:
        raise ValueError("Data preparation produced no usable observations")
    metadata = read_data_metadata(store, panel.revision)
    data = store.read_parquet_file("panel", panel.revision, parquet_filename("panel", "panel"))
    profile = profile_data(data, metadata=metadata)
    report = store.write_artifact(
        "data_profile",
        derived_from={"panel": panel.revision},
        produced_by="check:data_profile",
        json_files={json_filename("data_profile", "data_profile"): profile.model_dump(mode="json")},
    )
    retracted = list(effects.retracted)
    if selected.has("validation_report"):
        retracted.append(
            RetractedArtifact(artifact_id="validation_report", reason_ref="panel.changed")
        )
    return effects.model_copy(
        update={"produced": [*effects.produced, report], "retracted": retracted}
    )
