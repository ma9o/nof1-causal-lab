"""Current-code findings cached by the scientific inputs their existing owners consume."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.actions.data_checks import read_data_profile
from nof1_causal_lab.artifacts.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.model_checks import CheckGroup, ModelCheckReport, QuestionCheckReport
from nof1_causal_lab.artifacts.validation_report import ValidationIssue, ValidationReportArtifact
from nof1_causal_lab.models.model_inputs import data_binding_issues, input_fingerprints
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
from nof1_causal_lab.study.lineage import read_data_metadata
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore, cached_value, read_model, read_question

if TYPE_CHECKING:
    from collections.abc import Callable
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.models.ssm.compile.inputs import (
        FitCompilationResult,
        ModelCompilationResult,
    )
    from nof1_causal_lab.study.records import Applied, ModelFitResult
    from nof1_causal_lab.study.state import StudyState


def read_identification(
    store: ArtifactStore, selection: StructuralSelection
) -> IdentificationReport:
    inputs = input_fingerprints(selection.model)
    value, _ = cached_value(
        store.workspace_id,
        ("identification", inputs["identification"], selection.outcome or ""),
        TypeAdapter(IdentificationReport),
        lambda: selection.identification,
    )
    return value


def read_validation(
    store: ArtifactStore,
    selection: StructuralSelection,
    panel_revision: GitOid,
    inputs: Callable[[], FitCompilationResult],
) -> tuple[ValidationReportArtifact, bool]:
    from nof1_causal_lab.actions.validation.flow import validate_extraction

    def render() -> ValidationReportArtifact:
        model = selection.model
        panel = store.read_parquet_file("panel", panel_revision, "panel.parquet")
        metadata = read_data_metadata(store, panel_revision)
        audit = validate_extraction(
            model, [panel], data_profile=read_data_profile(store, panel_revision)
        )
        issues = tuple(
            ValidationIssue(
                indicator_id=None,
                issue_type="measurement_definitions",
                severity="error",
                message=issue,
            )
            for issue in data_binding_issues(model, metadata)
        )
        return ValidationReportArtifact(
            data=audit.revised(dataset_issues=(*audit.dataset_issues, *issues)),
            preflight=check_model_data(inputs(), panel, time_origin=metadata.time_origin),
        )

    fingerprints = input_fingerprints(selection.model)
    return cached_value(
        store.workspace_id,
        (
            "compatibility",
            fingerprints["observations"],
            fingerprints["belief"],
            selection.outcome or "",
            panel_revision,
        ),
        TypeAdapter(ValidationReportArtifact),
        render,
    )


def read_model_checks(
    workspace_id: str,
    state: StudyState,
    *,
    action: Literal["edit_model", "fit"],
) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None]:
    """Reuse across Git revisions with equal consumed inputs; never persist check identities."""
    store = ArtifactStore(workspace_id)
    model_record, question_record = state.current["model"], state.current["question"]
    model, question = (
        read_model(store, model_record.revision),
        read_question(store, question_record.revision),
    )
    selection = StructuralSelection.for_question(model, question)
    fingerprints = input_fingerprints(model)

    @cache
    def compilation() -> ModelCompilationResult:
        return compile_model(selection)

    @cache
    def fit_inputs() -> FitCompilationResult:
        return compile_fit_inputs(compilation(), selection)

    specification, specification_reused = cached_value(
        workspace_id,
        (
            "specification",
            fingerprints["compilation"],
            fingerprints["belief"],
            selection.outcome or "",
        ),
        TypeAdapter(tuple[SpecificationAssessment, ...]),
        lambda: check_specification(compilation(), fit_inputs()),
    )
    identification, identification_reused = cached_value(
        workspace_id,
        ("identification", fingerprints["identification"], selection.outcome or ""),
        TypeAdapter(IdentificationReport),
        lambda: selection.identification,
    )
    panel = state.get("panel")
    from nof1_causal_lab.models.question_checks import question_findings

    def question_report() -> QuestionCheckReport:
        return QuestionCheckReport(
            question_revision=question_record.revision,
            panel_revision=panel.revision if panel is not None else None,
            findings=question_findings(
                question,
                selection,
                panel=store.read_parquet_file("panel", panel.revision, "panel.parquet")
                if panel is not None
                else None,
                time_origin=read_data_metadata(store, panel.revision).time_origin
                if panel is not None
                else None,
            ),
        )

    question_checks, question_reused = cached_value(
        workspace_id,
        (
            "question",
            question_record.revision,
            fingerprints["identification"],
            selection.outcome or "",
            panel.revision if panel is not None else "",
        ),
        TypeAdapter(QuestionCheckReport),
        question_report,
    )
    groups: tuple[tuple[CheckGroup, bool], ...] = (
        ("specification", specification_reused),
        ("identification", identification_reused),
        ("question", question_reused),
    )
    reused: list[CheckGroup | Literal["predictive"]] = [group for group, hit in groups if hit]
    validation = None
    predictive = None
    if panel is not None:
        validation, hit = read_validation(store, selection, panel.revision, fit_inputs)
        if hit:
            reused.append("compatibility")
    if action == "edit_model":
        from nof1_causal_lab.actions.predictive_checks import check_model_predictive

        predictive, hit = check_model_predictive(store, state, selection, compilation)
        if hit:
            reused.append("predictive")
    return (
        ModelCheckReport(
            specification=specification,
            question=question_checks,
            predictive=predictive,
            reused=tuple(reused),
        ),
        identification,
        validation,
    )


def evaluate_model_checks(
    workspace_id: str,
    state: StudyState,
    applied: Applied[ModelFitResult | None],
    *,
    action: Literal["edit_model", "fit"],
) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None]:
    """The existing check activity warms the same reads used by saved calls."""
    return read_model_checks(
        workspace_id,
        apply_effects(state, applied.effects.produced, applied.effects.retracted),
        action=action,
    )
