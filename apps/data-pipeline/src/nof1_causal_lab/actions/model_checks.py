"""Evaluate action-owned findings, reusing equal scientific inputs during execution."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.actions.data_checks import read_data_profile
from nof1_causal_lab.artifacts.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.model_checks import CheckGroup, ModelCheckReport, QuestionCheckReport
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
from nof1_causal_lab.study.data import read_data_history
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore, cached_value, read_model, read_question

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.data_ref import DataRef
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.models.ssm.compile.inputs import (
        FitCompilationResult,
        ModelCompilationResult,
    )
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.state import StudyState


def read_validation(
    store: ArtifactStore,
    selection: StructuralSelection,
    source: DataRef[GitOid, int],
    inputs: Callable[[], FitCompilationResult],
) -> tuple[ValidationReportArtifact, bool]:
    """Read or compute model-data compatibility findings.

    Args:
        store: Workspace artifact store supplying the selected observations.
        selection: Model and outcome being checked against those observations.
        source: Exact data revision and replicate index to assess.
        inputs: Deferred fit compilation, evaluated only when the report is uncached.

    Returns:
        The validation report and a flag indicating whether it was read from cache.
        The cache key includes observation and belief fingerprints as well as the
        selected outcome and data history.
    """
    from nof1_causal_lab.actions.validation.flow import validate_extraction

    def render() -> ValidationReportArtifact:
        model = selection.model
        history = read_data_history(store, source)
        panel = history.observations.recorded.frame
        audit = validate_extraction(model, [panel], data_profile=read_data_profile(store, source))
        return ValidationReportArtifact(
            data=audit,
            preflight=check_model_data(
                inputs(), history.observations, time_origin=history.time_origin
            ),
        )

    fingerprints = input_fingerprints(selection.model)
    return cached_value(
        store.workspace_id,
        (
            "compatibility",
            fingerprints["observations"],
            fingerprints["belief"],
            selection.outcome or "",
            source.revision,
            str(source.replicate_index),
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
    """Evaluate a check bundle for the action to retain with its outcome."""
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
    source = state.data if action == "fit" else None
    history = read_data_history(store, source) if source is not None else None
    from nof1_causal_lab.models.question_checks import question_findings

    def question_report() -> QuestionCheckReport:
        return QuestionCheckReport(
            question_revision=question_record.revision,
            data=source,
            findings=question_findings(
                question,
                selection,
                panel=history.observations.recorded.frame if history is not None else None,
                time_origin=history.time_origin if history is not None else None,
            ),
        )

    question_checks, question_reused = cached_value(
        workspace_id,
        (
            "question",
            question_record.revision,
            fingerprints["identification"],
            selection.outcome or "",
            source.revision if source is not None else "",
            str(source.replicate_index) if source is not None else "",
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
    if source is not None:
        validation, hit = read_validation(store, selection, source, fit_inputs)
        if hit:
            reused.append("compatibility")
    return (
        ModelCheckReport(
            specification=specification,
            question=question_checks,
            reused=tuple(reused),
        ),
        identification,
        validation,
    )


def evaluate_model_checks[ResultT](
    workspace_id: str,
    state: StudyState,
    applied: Applied[ResultT],
    *,
    action: Literal["edit_model", "fit"],
) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None]:
    """Return the reports the workflow publishes with the action's outcome."""
    return read_model_checks(
        workspace_id,
        apply_effects(state, applied.effects.produced, applied.effects.retracted),
        action=action,
    )
