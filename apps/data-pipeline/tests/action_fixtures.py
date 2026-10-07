"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.io import EditQuestionInput
from nof1_causal_lab.actions.model_checks import evaluate_model_checks
from tests.helpers import fixture_entity_id


def edit_and_check(workspace_id, request, state):
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.store import ArtifactStore

    staged = edit_model(workspace_id, request)
    assert isinstance(staged, Applied)
    checks, identification, validation = evaluate_model_checks(
        workspace_id, state, staged, action="edit_model"
    )
    store = ArtifactStore(workspace_id)
    reports = {"checks": checks, "identification": identification}
    if validation is not None:
        reports["validation"] = validation
    return staged.revised(
        effects=staged.effects.revised(
            reports={name: store.write_report(report) for name, report in reports.items()}
        )
    )


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as edit_question does."""
    from nof1_causal_lab.actions.contracts import EditQuestionRequest
    from nof1_causal_lab.actions.edit_question import edit_question
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.study.history import StudyRepository

    request = EditQuestionRequest(
        input=EditQuestionInput(
            question=question
            if question is not None
            else QuestionSpec(
                text="What does it answer?",
                outcome=fixture_entity_id("construct", "unmeasured_outcome"),
            )
        )
    )
    journal = StudyRepository(workspace_id)
    return journal.append(
        applied_record(
            workspace_id,
            edit_question(workspace_id, request),
            request=request,
            seq=journal.latest_seq() + 1,
        )
    )


def applied_record(
    workspace_id, result, *, seq, request=None, ts="2026-01-01T00:00:00Z", **metadata
):
    """A successful test publication composes the actual owned action payload."""
    from pydantic import TypeAdapter

    from nof1_causal_lab.actions.output_builder import build_output, staged_state
    from nof1_causal_lab.artifacts.model_spec import ModelEditResult
    from nof1_causal_lab.artifacts.posterior import ModelFitResult
    from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import (
        ActionAttempt,
        Applied,
        AttemptRecord,
        DataPreparationResult,
        StagedEditAttempt,
        StagedEditQuestionAttempt,
        StagedFitAttempt,
        StagedPrepareAttempt,
        StagedSimulateAttempt,
        applied_attempt,
    )
    from nof1_causal_lab.study.store import ArtifactStore

    if request is None and isinstance(result.result, ModelFitResult):
        from nof1_causal_lab.actions.contracts import FitRequest
        from nof1_causal_lab.actions.io import FitInput
        from nof1_causal_lab.artifacts.identity import GitOid

        request = FitRequest[GitOid](
            input=FitInput[GitOid](
                model_ref=result.result.model.revision,
                data_ref=result.result.data.revision,
                replicate_index=result.result.data.replicate_index,
            )
        )
    if result.result is None and (
        getattr(request, "action", None) == "edit_model"
        or any(artifact.artifact_id == "model" for artifact in result.effects.produced)
    ):
        result = result.revised(result=ModelEditResult())
    if request is not None:
        attempt = applied_attempt(request, result)
    else:
        match result.result:
            case None:
                attempt = StagedEditQuestionAttempt(
                    action="edit_question", request=None, outcome=result
                )
            case ModelEditResult():
                attempt = StagedEditAttempt(action="edit_model", request=None, outcome=result)
            case DataPreparationResult():
                attempt = StagedPrepareAttempt(action="prepare_data", request=None, outcome=result)
            case ModelFitResult():
                attempt = StagedFitAttempt(action="fit", request=None, outcome=result)
            case ModelSimulationResult():
                attempt = StagedSimulateAttempt(action="simulate", request=None, outcome=result)
            case _:
                raise TypeError("A publication fixture needs an owned action result")
    repository, store = StudyRepository(workspace_id), ArtifactStore(workspace_id)
    state = staged_state(repository.state(repository.head()), result.effects)
    if not state.has("question"):
        from tests.helpers import write_question

        state = state.with_artifacts((write_question(store),))
    if isinstance(result.result, ModelFitResult):
        state = state.revised(data=result.result.data)
    if isinstance(result.result, DataPreparationResult):
        from nof1_causal_lab.actions.data_checks import evaluate_data_checks

        profile = evaluate_data_checks(workspace_id, state, result)
        attempt = attempt.revised(
            outcome=result.revised(
                effects=result.effects.revised(
                    reports={**result.effects.reports, "data-profile": store.write_report(profile)}
                )
            )
        )
    if (
        isinstance(result.result, ModelSimulationResult)
        and "simulation" not in result.effects.reports
    ):
        from nof1_causal_lab.actions.simulate import read_simulation_report

        report = read_simulation_report(
            store, result.result.evidence, state.current["question"].revision
        )
        attempt = attempt.revised(
            outcome=result.revised(
                effects=result.effects.revised(
                    reports={**result.effects.reports, "simulation": store.write_report(report)}
                )
            )
        )
    if isinstance(result.result, ModelFitResult):
        from nof1_causal_lab.artifacts.model_checks import ModelCheckReport, QuestionCheckReport
        from nof1_causal_lab.artifacts.validation_report import (
            DataProfileArtifact,
            ValidationReportArtifact,
        )
        from nof1_causal_lab.study.store import read_model
        from tests.inference_fixtures import _report

        assert isinstance(attempt.outcome, Applied)
        reports = dict(attempt.outcome.effects.reports)
        if "inference" not in reports:
            report = _report(read_model(store, state.current["model"].revision)).revised(
                run=result.result,
            )
            reports["inference"] = store.write_report(report)
        if "checks" not in reports:
            reports["checks"] = store.write_report(
                ModelCheckReport(
                    specification=(),
                    question=QuestionCheckReport(
                        question_revision=state.current["question"].revision,
                        data=result.result.data,
                        findings=(),
                    ),
                )
            )
        if "validation" not in reports:
            reports["validation"] = store.write_report(
                ValidationReportArtifact(
                    data=DataProfileArtifact(indicators={}, dataset_issues=()),
                )
            )
        attempt = attempt.revised(
            outcome=attempt.outcome.revised(
                effects=attempt.outcome.effects.revised(reports=reports),
            )
        )
    output = build_output(workspace_id, attempt, state)
    # Storage-focused fixtures may retain physical input artifacts. The result itself
    # is always the same complete body used by production publication.
    saved = TypeAdapter(ActionAttempt).validate_python(
        {
            "action": attempt.action,
            "request": request,
            "outcome": Applied(
                result=store.write_result(output), effects=result.effects.revised(reports={})
            ),
        }
    )
    return AttemptRecord(seq=seq, ts=ts, attempt=saved, **metadata)


def empty_simulation_summary():
    """Reports used only for routing have no plotted reductions."""
    from nof1_causal_lab.artifacts.simulation import SimulationSummary

    return SimulationSummary(
        state_frames={},
        indicator_frames={},
        action_category_probabilities={},
        reference_category_probabilities={},
    )
