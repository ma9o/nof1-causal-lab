"""Complete action-owned scientific results before atomic publication."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.actions.io import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.study.artifact_files import parquet_filename
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import ActionAttempt, Applied, StagedActionAttempt
from nof1_causal_lab.study.store import ArtifactStore, read_model, read_question

if TYPE_CHECKING:
    from nof1_causal_lab.actions.effects import ActionEffects
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.study.results import ActionOutput


from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.study.state import StudyState, apply_effects


class OutputBuilder:
    """Read one execution's inputs and computed findings to construct its final result."""

    def __init__(self, workspace_id: str, attempt: StagedActionAttempt, state: StudyState) -> None:
        """Bind only the current action's staged artifacts and reports."""
        self.store = ArtifactStore(workspace_id)
        assert isinstance(attempt.outcome, Applied)
        self.applied = attempt.outcome
        self.state = state

    @cached_property
    def checks(
        self,
    ) -> tuple[ModelCheckReport, IdentificationReport] | None:
        """Checks computed by this action, without searching another call's reports."""
        from nof1_causal_lab.artifacts.identification import IdentificationReport
        from nof1_causal_lab.artifacts.model_checks import ModelCheckReport

        reports = self.applied.effects.reports
        if "checks" not in reports:
            return None
        return (
            self.store.read_report(reports["checks"], ModelCheckReport),
            self.store.read_report(reports["identification"], IdentificationReport),
        )

    def simulation(self) -> SimulationReport:
        """The findings computed for this simulation's retained draws."""
        from nof1_causal_lab.artifacts.simulation import SimulationReport

        return self.store.read_report(self.applied.effects.reports["simulation"], SimulationReport)

    @cached_property
    def question(self) -> QuestionSpec | None:
        """Read the question selected by this execution's inputs."""
        return (
            read_question(self.store, self.state.current["question"].revision)
            if self.state.has("question")
            else None
        )

    @cached_property
    def model(self) -> ModelSpec | None:
        """Read the scientific model produced by this execution."""
        return (
            read_model(self.store, self.state.current["model"].revision)
            if self.state.has("model")
            else None
        )

    def identification(self) -> IdentificationReport | None:
        """Return model identification findings, or ``None`` without retained checks."""
        if self.checks is None:
            return None
        return self.checks[1]

    def model_output(self) -> EditModelOutput:
        """Retain the model and scientific findings without computing viewer projections."""
        model = self.model
        assert model is not None, "A model producer must retain its model"
        return EditModelOutput(
            model=model,
            checks=self.checks[0] if self.checks is not None else None,
            identification=self.identification(),
        )


def build_output(
    workspace_id: str, attempt: StagedActionAttempt, state: StudyState
) -> ActionOutput:
    """Complete one action's result using only its inputs and staged execution evidence."""
    from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
    from nof1_causal_lab.study.lineage import read_data_metadata
    from nof1_causal_lab.study.visuals import observation_history, simulation_observation_histories

    assert isinstance(attempt.outcome, Applied)
    store = ArtifactStore(workspace_id)
    effects = attempt.outcome.effects
    if attempt.action == "data_diff":
        return store.read_report(effects.reports["data-diff"], DataDiffOutput)
    if attempt.action == "model_diff":
        return store.read_result(effects.reports["model-diff"], ModelDiffOutput)
    reader = OutputBuilder(workspace_id, attempt, state)
    match attempt.action:
        case "edit_question":
            assert reader.question is not None
            return EditQuestionOutput(question=reader.question)
        case "edit_model":
            return reader.model_output()
        case "prepare_data":
            panel_revision = next(
                artifact.revision
                for artifact in effects.produced
                if artifact.artifact_id == "panel"
            )
            metadata = read_data_metadata(store, panel_revision)
            panel = store.read_parquet_file(
                "panel", panel_revision, parquet_filename("panel", "panel")
            )
            return PrepareDataOutput(
                metadata=metadata,
                profile=store.read_report(effects.reports["data-profile"], DataProfileArtifact),
                data={
                    variable.id: observation_history(
                        metadata.time_origin,
                        variable,
                        panel.filter(pl.col("indicator_id") == variable.id).sort("anchor_time"),
                    )
                    for variable in metadata.variables
                },
            )
        case "fit":
            from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
            from nof1_causal_lab.artifacts.posterior import FitCheckReport, InferenceReport
            from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact

            model = read_model(store, state.current["model"].revision)
            inference = store.read_report(effects.reports["inference"], InferenceReport)
            checks = store.read_report(effects.reports["checks"], ModelCheckReport)
            assert checks.question is not None, "A completed fit evaluates its pinned question"
            return FitOutput(
                model=model,
                checks=FitCheckReport(
                    validation=store.read_report(
                        effects.reports["validation"], ValidationReportArtifact
                    ),
                    question=checks.question,
                ),
                inference=inference,

            )
        case "simulate":
            report = reader.simulation()
            evidence = report.evidence
            return SimulateOutput(
                report=report,
                data=simulation_observation_histories(
                    evidence,
                    evidence.arms.action.observations.values,
                    evidence.observation_layout.mask.values,
                ),
            )


def complete_attempt(workspace_id: str, attempt: StagedActionAttempt) -> ActionAttempt:
    """Finish a staged execution and retain the one result its publication will own."""
    from nof1_causal_lab.study.records import failed_attempt, retained_attempt

    assert attempt.request is not None
    if not isinstance(attempt.outcome, Applied):
        return failed_attempt(attempt.request, attempt.outcome)
    inputs = StudyRepository(workspace_id).input_state(attempt.request)
    state = staged_state(inputs, attempt.outcome.effects)
    result = build_output(workspace_id, attempt, state)
    store = ArtifactStore(workspace_id)
    identity = store.write_result(result)
    produced = tuple(
        store.result_artifact(info, identity) for info in attempt.outcome.effects.produced
    )
    return retained_attempt(
        attempt, identity, attempt.outcome.effects.revised(produced=produced, reports={})
    )


def staged_state(inputs: StudyState, effects: ActionEffects) -> StudyState:
    """Resolve the execution's selected artifacts and its newly prepared history."""
    state = apply_effects(inputs, effects.produced, effects.retracted)
    panel = next((item for item in effects.produced if item.artifact_id == "panel"), None)
    return (
        state.revised(data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0))
        if panel is not None
        else state
    )
