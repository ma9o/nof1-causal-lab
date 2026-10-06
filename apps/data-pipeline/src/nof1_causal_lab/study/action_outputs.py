"""Load only the scientific outputs owned by a completed action."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from nof1_causal_lab.actions.call_logs import read_call_log
from nof1_causal_lab.actions.contracts import call_identity
from nof1_causal_lab.actions.io import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.actions.results import ActionSuccess, FailedPoll, SuccessfulPoll
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.visuals import simulation_observation_histories

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from nof1_causal_lab.json_types import JsonValue


def _arrays(store: ArtifactStore, references: Iterable[str]) -> Mapping[str, JsonValue]:
    arrays: dict[str, JsonValue] = {}
    for reference in sorted(set(references)):
        values = store.read_array(reference)
        arrays[reference] = np.where(
            np.isfinite(values), values, np.asarray(None, dtype=object)
        ).tolist()
    return arrays


def completed_call(workspace_id: str, revision: StudyRevision) -> ActionSuccess | FailedPoll:
    """Each branch reads its own results; inputs stay references to other calls."""
    attempt = revision.record.attempt
    if attempt.request is None:
        raise StudyLookupError("This historical attempt has no retained call arguments")
    repository = StudyRepository(workspace_id)
    identity = call_identity(attempt.request)
    messages = read_call_log(repository, revision.commit_id).messages
    if not isinstance(attempt.outcome, Applied):
        return FailedPoll(
            call_id=identity, action=attempt.action, commit_id=revision.commit_id, messages=messages
        )

    def success[ActionT: str, BodyT](
        response_type: type[SuccessfulPoll[ActionT, BodyT]], action: ActionT, body: BodyT
    ) -> SuccessfulPoll[ActionT, BodyT]:
        return response_type(
            call_id=identity,
            action=action,
            commit_id=revision.commit_id,
            body=body,
            messages=messages,
        )

    # Comparisons already own complete reports and need no model or data reader.
    store = ArtifactStore(workspace_id)
    if attempt.action == "data_diff":
        return success(
            SuccessfulPoll[Literal["data_diff"], DataDiffOutput],
            "data_diff",
            store.read_report(attempt.outcome.effects.reports["data-diff"], DataDiffOutput),
        )
    if attempt.action == "model_diff":
        return success(
            SuccessfulPoll[Literal["model_diff"], ModelDiffOutput],
            "model_diff",
            store.read_report(attempt.outcome.effects.reports["model-diff"], ModelDiffOutput),
        )

    inputs = repository.input_state(attempt.request)
    effects = attempt.outcome.effects
    state = apply_effects(inputs, effects.produced, effects.retracted)
    panel = next((item for item in effects.produced if item.artifact_id == "panel"), None)
    if panel is not None:
        state = state.revised(data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0))
    reader = ModelReader(workspace_id, at=revision.commit_id, state=state)
    match attempt.action:
        case "edit_question":
            question = reader.question
            assert question is not None, "A successful question action publishes its question"
            return success(
                SuccessfulPoll[Literal["edit_question"], EditQuestionOutput],
                "edit_question",
                EditQuestionOutput(question=question),
            )
        case "edit_model":
            return success(
                SuccessfulPoll[Literal["edit_model"], EditModelOutput],
                "edit_model",
                reader.model_output(),
            )
        case "prepare_data":
            result = attempt.outcome.result
            history = reader.data_history
            return success(
                SuccessfulPoll[Literal["prepare_data"], PrepareDataOutput],
                "prepare_data",
                PrepareDataOutput(
                    workers=result.workers,
                    extraction_reused=result.extraction_reused,
                    raw_data=reader.raw_data,
                    measurements=reader.measurements,
                    metadata=reader.data_metadata,
                    profile=reader.data_profile,
                    data={
                        variable.id: value
                        for variable in history.variables
                        if (value := reader.observation_history(variable.id)) is not None
                    }
                    if history is not None
                    else {},
                ),
            )
        case "fit":
            fit = reader.fit()
            result = attempt.outcome.result
            return success(
                SuccessfulPoll[Literal["fit"], FitOutput],
                "fit",
                FitOutput(
                    model=reader.model_output(fit),
                    inference=result,
                    summary=fit,
                    inference_report=reader.inference_report,
                    parameter_draws=reader.parameter_draws(),
                    arrays=_arrays(store, result.evidence.array_references)
                    if result is not None
                    else {},
                ),
            )
        case "simulate":
            result = attempt.outcome.result
            evidence = result.evidence
            references = (
                *evidence.parameter_draws.values(),
                evidence.latent_paths,
                evidence.observations,
                evidence.observation_layout.mask,
                evidence.observation_layout.support_start_times,
                evidence.observation_layout.support_end_times,
                *(
                    (evidence.reference_latent_paths,)
                    if evidence.reference_latent_paths is not None
                    else ()
                ),
                *(
                    (evidence.reference_observations,)
                    if evidence.reference_observations is not None
                    else ()
                ),
            )
            return success(
                SuccessfulPoll[Literal["simulate"], SimulateOutput],
                "simulate",
                SimulateOutput(
                    simulation=result,
                    report=reader.simulation(),
                    data=simulation_observation_histories(
                        evidence,
                        store.read_array(evidence.observations),
                        store.read_array(evidence.observation_layout.mask),
                        store.read_array(evidence.observation_layout.support_start_times),
                        store.read_array(evidence.observation_layout.support_end_times),
                    ),
                    paths=reader.simulation_paths(start=0, count=evidence.draws),
                    arrays=_arrays(store, references),
                ),
            )
