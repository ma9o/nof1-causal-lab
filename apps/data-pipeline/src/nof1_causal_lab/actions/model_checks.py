"""Explicit action-owned checks, selected by the scientific inputs they consume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.actions.checks import check_specification
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.execution import RetractedArtifact, TransitionEffects, apply_transition
from nof1_causal_lab.machine.store import ArtifactStore, read_model

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.model_checks import CheckGroup
    from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState

# Bump when a check's interpretation or implementation changes.
CHECK_POLICY_VERSION = "model-checks-v3"


def evaluate_model_checks(
    workspace_id: str,
    state: EpisodeState,
    effects: TransitionEffects,
    *,
    action: Literal["edit_model", "fit"],
) -> TransitionEffects:
    """Finish one edit or fit before its model and findings commit together.

    This is a fixed sequence, not an artifact scheduler. Reuse is scoped to the
    selected snapshot and keyed independently for each family of scientific checks.
    """
    selected = apply_transition(state, effects.produced, effects.retracted)
    if not selected.has("model"):
        return effects
    store = ArtifactStore(workspace_id)
    model_record = selected.current["model"]
    model = read_model(store, model_record.revision)
    inputs = model_record.model_inputs
    previous = state.checks
    keys: dict[CheckGroup, str] = {}
    reused: list[CheckGroup | Literal["predictive"]] = []
    produced = list(effects.produced)
    retracted = list(effects.retracted)

    def unchanged(group: CheckGroup, values: object) -> bool:
        keys[group] = scientific_id("check", [CHECK_POLICY_VERSION, group, values])
        same = previous is not None and previous.input_keys.get(group) == keys[group]
        if same:
            reused.append(group)
        return same

    specification = (
        previous.specification
        if unchanged("specification", [inputs["compilation"], inputs["belief"]])
        and previous is not None
        else check_specification(model)
    )
    if not unchanged("identification", inputs["identification"]):
        produced.append(_write_identification(store, {"model": model_record.revision}))
    else:
        produced.append(selected.current["identification_report"])

    panel = selected.get("panel") if action == "edit_model" else None
    if panel is not None:
        pins: dict[ArtifactId, GitOid] = {
            "model": model_record.revision,
            "panel": panel.revision,
            "data_profile": selected.current["data_profile"].revision,
        }
        if not unchanged(
            "compatibility",
            [
                inputs["observations"],
                inputs["belief"],
                panel.revision,
                pins["data_profile"],
            ],
        ):
            produced.append(_write_validation(store, pins))
        else:
            produced.append(selected.current["validation_report"])
    elif action == "edit_model":
        retracted.extend(
            RetractedArtifact(artifact_id=identity, reason_ref=f"{identity}.panel_absent")
            for identity in ("data_profile", "validation_report")
            if selected.has(identity)
        )

    predictive = None
    if action == "edit_model":
        from nof1_causal_lab.actions.predictive_checks import check_model_predictive

        predictive, was_reused = check_model_predictive(
            store,
            selected,
            model,
            specification,
            previous=previous.predictive if previous is not None else None,
        )
        if was_reused:
            reused.append("predictive")
    return effects.model_copy(
        update={
            "produced": produced,
            "retracted": retracted,
            "checks": ModelCheckReport(
                input_keys=keys,
                specification=specification,
                predictive=predictive,
                reused=tuple(reused),
            ),
        }
    )


def _read_panel(store: ArtifactStore, revision: GitOid) -> pl.DataFrame:
    return store.read_parquet_file("panel", revision, parquet_filename("panel", "panel"))


def _write_identification(store: ArtifactStore, pins: dict[ArtifactId, GitOid]) -> ArtifactRecord:
    from nof1_causal_lab.models.identification import identify_model

    report = identify_model(read_model(store, pins["model"]))
    return store.write_artifact(
        "identification_report",
        derived_from=pins,
        produced_by="check:identification_report",
        json_files={
            json_filename("identification_report", "identification_report"): report.model_dump(
                mode="json"
            )
        },
    )


def _write_validation(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
) -> ArtifactRecord:
    from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
    from nof1_causal_lab.flows.transitions.validation.flow import (
        validate_extraction,
    )

    model = read_model(store, pins["model"])
    panel = _read_panel(store, pins["panel"])
    from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact

    profile = DataProfileArtifact.model_validate(
        store.read_json_file(
            "data_profile", pins["data_profile"], json_filename("data_profile", "data_profile")
        )
    )
    audit_result = validate_extraction(model, [panel], data_profile=profile)
    if not audit_result:
        raise RuntimeError(
            "validation_report derivation returned an empty audit result; "
            "refusing to fabricate an is_valid=False report with empty indicators."
        )

    from nof1_causal_lab.actions.data_checks import data_binding_issues, read_data_metadata

    for issue in data_binding_issues(model, read_data_metadata(store, pins["panel"])):
        audit_result["dataset_issues"].append(
            {
                "indicator_id": None,
                "issue_type": "measurement_definitions",
                "severity": "error",
                "message": issue,
            }
        )
    from nof1_causal_lab.actions.checks import check_model_data

    preflight = check_model_data(
        model, panel, time_origin=read_data_metadata(store, pins["panel"]).time_origin
    )
    payload = ValidationReportArtifact.model_validate(
        {
            "indicators": audit_result["indicators"],
            "dataset_issues": audit_result["dataset_issues"],
            "preflight": preflight,
        }
    ).model_dump(mode="json")
    return store.write_artifact(
        "validation_report",
        derived_from=pins,
        produced_by="check:validation_report",
        json_files={json_filename("validation_report", "validation_report"): payload},
    )
