"""Evaluate the scientific findings owned by the executing action."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport, QuestionCheckReport
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_fit_inputs, compile_model
from nof1_causal_lab.study.data import read_data_history
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore, read_model, read_question

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.data_ref import DataRef
    from nof1_causal_lab.artifacts.identification import IdentificationReport
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
) -> ValidationReportArtifact:
    """Compute model-dependent compatibility findings for the selected history."""
    from nof1_causal_lab.actions.validation.flow import validate_extraction

    def render() -> ValidationReportArtifact:
        model = selection.model
        history = read_data_history(store, source)
        panel = history.observations.recorded.frame
        audit = validate_extraction(model, [panel])
        return ValidationReportArtifact(
            data=audit,
            preflight=check_model_data(
                inputs(), history.observations, time_origin=history.time_origin
            ),
        )

    return render()


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

    @cache
    def compilation() -> ModelCompilationResult:
        return compile_model(selection)

    @cache
    def fit_inputs() -> FitCompilationResult:
        return compile_fit_inputs(compilation(), selection)

    specification = check_specification(compilation(), fit_inputs())
    identification = selection.identification
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

    question_checks = question_report()
    validation = (
        read_validation(store, selection, source, fit_inputs) if source is not None else None
    )
    return (
        ModelCheckReport(
            specification=specification,
            question=question_checks,
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
