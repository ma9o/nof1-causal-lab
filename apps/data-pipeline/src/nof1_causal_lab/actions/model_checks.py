"""Explicit action-owned checks, selected by the scientific inputs they consume."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.actions.checks import check_specification
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport, QuestionCheckReport
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study.records import Applied, ModelFitResult
from nof1_causal_lab.study.state import RetractedArtifact, apply_effects
from nof1_causal_lab.study.store import ArtifactStore, read_model, read_question

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.model_checks import CheckGroup
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.models.ssm.compile.inputs import (
        FitCompilationResult,
        ModelCompilationResult,
    )
    from nof1_causal_lab.study.state import ArtifactRecord, StudyState

# Bump when a check's interpretation or implementation changes.
CHECK_POLICY_VERSION = "model-checks-v5"


def evaluate_model_checks[ResultT: ModelFitResult | None](
    workspace_id: str,
    state: StudyState,
    applied: Applied[ResultT],
    *,
    action: Literal["edit_model", "fit"],
) -> Applied[ResultT]:
    """Finish one edit or fit before its model and findings commit together.

    This is a fixed sequence, not an artifact scheduler. Reuse is scoped to the
    selected snapshot and keyed independently for each family of scientific checks.
    """
    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    if not selected.has("model"):
        return applied
    store = ArtifactStore(workspace_id)
    model_record = selected.current["model"]
    model = read_model(store, model_record.revision)
    question_record = selected.current["question"]
    question = read_question(store, question_record.revision)
    # The question's outcome scopes what compiles, fits and simulates.
    selection = StructuralSelection.for_question(model, question)
    inputs = model_record.model_inputs
    previous = state.checks
    keys: dict[CheckGroup, str] = {}
    reused: list[CheckGroup | Literal["predictive"]] = []
    produced = list(applied.effects.produced)
    retracted = list(applied.effects.retracted)

    @cache
    def compilation() -> ModelCompilationResult:
        return compile_model(selection)

    @cache
    def fit_inputs() -> FitCompilationResult:
        return compile_fit_inputs(compilation(), selection)

    def unchanged(group: CheckGroup, values: object) -> bool:
        keys[group] = scientific_id("check", [CHECK_POLICY_VERSION, group, values])
        same = previous is not None and previous.input_keys.get(group) == keys[group]
        if same:
            reused.append(group)
        return same

    specification = (
        previous.specification
        if unchanged("specification", [inputs["compilation"], inputs["belief"], selection.outcome])
        and previous is not None
        else check_specification(compilation(), fit_inputs())
    )
    if not unchanged("identification", [inputs["identification"], selection.outcome]):
        identification_record = _write_identification(
            store,
            {"question": question_record.revision, "model": model_record.revision},
            selection,
        )
    else:
        identification_record = selected.current["identification_report"]
    produced.append(identification_record)

    current_panel = selected.get("panel")
    question_checks = (
        previous.question
        if unchanged(
            "question",
            [
                question_record.revision,
                inputs["identification"],
                identification_record.revision,
                current_panel.revision if current_panel is not None else None,
            ],
        )
        and previous is not None
        else _check_question(store, question_record, question, selection, current_panel)
    )

    panel = current_panel if action == "edit_model" else None
    if panel is not None:
        pins: dict[ArtifactId, GitOid] = {
            "question": question_record.revision,
            "model": model_record.revision,
            "panel": panel.revision,
            "data_profile": selected.current["data_profile"].revision,
        }
        if not unchanged(
            "compatibility",
            [
                inputs["observations"],
                inputs["belief"],
                selection.outcome,
                panel.revision,
                pins["data_profile"],
            ],
        ):
            produced.append(_write_validation(store, pins, selection, fit_inputs()))
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
            selection,
            compilation,
            previous=previous.predictive if previous is not None else None,
        )
        if was_reused:
            reused.append("predictive")
    return Applied(
        result=applied.result,
        effects=applied.effects.with_checks(
            produced=tuple(produced),
            retracted=tuple(retracted),
            checks=ModelCheckReport(
                input_keys=keys,
                specification=specification,
                question=question_checks,
                predictive=predictive,
                reused=tuple(reused),
            ),
        ),
    )


def _read_panel(store: ArtifactStore, revision: GitOid) -> pl.DataFrame:
    return store.read_parquet_file("panel", revision, parquet_filename("panel", "panel"))


def _write_identification(
    store: ArtifactStore, pins: dict[ArtifactId, GitOid], selection: StructuralSelection
) -> ArtifactRecord:
    report = selection.identification
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


def _check_question(
    store: ArtifactStore,
    record: ArtifactRecord,
    question: QuestionSpec,
    selection: StructuralSelection,
    panel: ArtifactRecord | None,
) -> QuestionCheckReport:
    from nof1_causal_lab.models.question_checks import question_findings
    from nof1_causal_lab.study.lineage import read_data_metadata

    return QuestionCheckReport(
        question_revision=record.revision,
        panel_revision=panel.revision if panel is not None else None,
        findings=question_findings(
            question,
            selection,
            panel=_read_panel(store, panel.revision) if panel is not None else None,
            time_origin=read_data_metadata(store, panel.revision).time_origin
            if panel is not None
            else None,
        ),
    )


def _write_validation(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    selection: StructuralSelection,
    inputs: FitCompilationResult,
) -> ArtifactRecord:
    from nof1_causal_lab.actions.validation.flow import (
        validate_extraction,
    )
    from nof1_causal_lab.artifacts.validation_report import (
        ValidationIssue,
        ValidationReportArtifact,
    )

    model = selection.model
    panel = _read_panel(store, pins["panel"])
    from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact

    profile = store.read_value(
        "data_profile", pins["data_profile"], "data_profile.json", DataProfileArtifact
    )
    audit_result = validate_extraction(model, [panel], data_profile=profile)
    from nof1_causal_lab.actions.checks import check_model_data
    from nof1_causal_lab.models.model_inputs import data_binding_issues
    from nof1_causal_lab.study.lineage import read_data_metadata

    preflight = check_model_data(
        inputs, panel, time_origin=read_data_metadata(store, pins["panel"]).time_origin
    )
    binding_issues = tuple(
        ValidationIssue(
            indicator_id=None, issue_type="measurement_definitions", severity="error", message=issue
        )
        for issue in data_binding_issues(model, read_data_metadata(store, pins["panel"]))
    )
    payload = ValidationReportArtifact(
        data=audit_result.revised(dataset_issues=(*audit_result.dataset_issues, *binding_issues)),
        preflight=preflight,
    )
    return store.write_artifact(
        "validation_report",
        derived_from=pins,
        produced_by="check:validation_report",
        json_files={
            json_filename("validation_report", "validation_report"): payload.model_dump(mode="json")
        },
    )
