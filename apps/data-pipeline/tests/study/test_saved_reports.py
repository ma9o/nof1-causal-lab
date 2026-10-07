"""Action findings survive cache eviction and are never evaluated by result readers."""

import asyncio
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pygit2
import pytest

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.io import EditModelInput, FitInput, PrepareDataInput, SimulateInput
from nof1_causal_lab.actions.temporal.activities import (
    evaluate_data_checks_activity,
    evaluate_model_checks_activity,
    journal_activity,
)
from nof1_causal_lab.actions.temporal.messages import AttemptPublication, EvaluateChecksInput
from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated
from nof1_causal_lab.artifacts.data_preparation import FileSourceRef
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import DistributionId, GitOid, GitRef
from nof1_causal_lab.artifacts.model_checks import (
    ModelCheckReport,
    QuestionCheckReport,
)
from nof1_causal_lab.artifacts.posterior import (
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
    ModelFitResult,
)
from nof1_causal_lab.artifacts.posterior_diagnostics import ParameterConvergenceReport
from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.simulation import (
    ModelSimulationResult,
    SimulationEvidence,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.lineage import fitted_law_report
from nof1_causal_lab.study.records import Applied, DataPreparationResult
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data
from tests.action_fixtures import applied_record, question_root
from tests.integration.runner_fixtures import panel_frame, panel_metadata, seed_model
from tests.model_fixtures import x_y_model

pytestmark = pytest.mark.contract


def test_action_reports_are_published_once_and_loaded_without_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    workspace = "SAVED"
    question_root(workspace)
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={
            "panel.parquet": panel_frame(n_days=2).with_columns(
                pl.col("anchor_time", "support_start", "support_end").str.to_datetime(
                    time_zone="UTC"
                )
            )
        },
    )
    profile = DataProfileArtifact(indicators={}, dataset_issues=())

    def publish(applied, *, request=None, model_checks=None, data_profile=None):
        assert request is not None
        evaluated = None
        if model_checks is not None:
            monkeypatch.setattr(
                "nof1_causal_lab.actions.model_checks.evaluate_model_checks",
                lambda *_args, **_kwargs: model_checks,
            )
            evaluated = asyncio.run(
                evaluate_model_checks_activity(
                    EvaluateChecksInput(
                        workspace_id=workspace,
                        state=repository.state(repository.head()),
                        applied=applied,
                        request=request,
                    )
                )
            )
        if data_profile is not None:
            monkeypatch.setattr(
                "nof1_causal_lab.actions.data_checks.evaluate_data_checks",
                lambda *_args: data_profile,
            )
            evaluated = asyncio.run(
                evaluate_data_checks_activity(
                    EvaluateChecksInput(
                        workspace_id=workspace,
                        state=repository.state(repository.head()),
                        applied=applied,
                        request=request,
                    )
                )
            )
        if evaluated is not None:
            # Even a multi-megabyte report crosses Temporal only by immutable reference.
            assert len(evaluated.model_dump_json().encode()) < 10_000
            applied = applied.revised(
                effects=applied.effects.revised(
                    reports={**applied.effects.reports, **evaluated.reports}
                )
            )
        record = applied_record(
            workspace,
            applied,
            seq=repository.latest_seq() + 1,
            request=request,
            messages=evaluated.messages if evaluated is not None else (),
        )
        publication = AttemptPublication(
            workspace_id=workspace,
            parent_id=repository.head(),
            record=record,
        )
        assert len(publication.model_dump_json().encode()) < 20_000
        return publication, asyncio.run(journal_activity(publication))

    _, prepared = publish(
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=(panel,))),
        request=PrepareDataRequest[GitOid, FileSourceRef](
            input=PrepareDataInput[GitOid, FileSourceRef](
                model_ref=seed_model(store).revision,
                source=panel_metadata().source,
                extraction={
                    variable.observation.id: variable.extraction
                    for variable in panel_metadata().preparation.variables
                },
            )
        ),
        data_profile=profile,
    )
    model = x_y_model()
    authored = store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={"question": repository.question().revision},
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)},
    )
    checks = ModelCheckReport(
        specification=(
            Evaluated(subject="model_execution", outcome="passed", evidence="Saved" * 500_000),
        ),
        question=QuestionCheckReport(
            question_revision=repository.question().revision,
            data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
            findings=(),
        ),
    )
    assert checks.question is not None
    edit_checks = checks.revised(question=checks.question.revised(data=None))

    identification = IdentificationReport(outcome=None)
    validation = ValidationReportArtifact(
        data=profile,
        preflight=(Evaluated(subject="fit_preflight", outcome="failed", evidence="Saved failure"),),
    )
    _, edited = publish(
        Applied(result=None, effects=ActionEffects(produced=(authored,))),
        request=EditModelRequest[GitOid](
            input=EditModelInput[GitOid](
                parent_ref=repository.question().revision,
                model=model,
            )
        ),
        model_checks=(edit_checks, identification, None),
    )
    fitted = store.write_artifact(
        "model",
        produced_by="fit",
        derived_from={"model": authored.revision, "panel": panel.revision},
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)},
    )
    inference = InferenceReport(
        core=InferenceReportCore(
            time_origin=None,
            inference_metadata=InferenceMetadata(n_samples=2, duration_seconds=3),
            engine=NotEvaluated(subject="production_engine", reason="ARCHIVED_ENGINE_NOT_RETAINED"),
            inference_diagnostics=None,
            sampler_diagnostics=None,
            convergence=ParameterConvergenceReport(assessments=()),
        ),
        detail=InferenceReportDetail(initial_latent_delta=((1.0, 2.0),), divergent=(False, True)),
    )

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("Saved result reads and publication must not run checks")

    monkeypatch.setattr("nof1_causal_lab.actions.fit.read_inference_report", unexpected_evaluation)
    initial_delta = store.write_array(np.asarray([[1.0, 2.0]]))
    divergent = store.write_array(np.asarray([False, True]))
    fit_publication, fit_revision = publish(
        Applied(
            result=ModelFitResult(
                model=GitRef(workspace_id=workspace, revision=authored.revision, path="model.json"),
                data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
                evidence=InferenceEvidence(
                    distribution=DistributionId("distribution:retained"),
                    engine=None,
                    time_origin=None,
                    duration_seconds=3,
                    initial_latent_delta=initial_delta,
                    chain_extra_fields={"diverging": divergent},
                ),
            ),
            effects=ActionEffects(
                produced=(fitted,), reports={"inference": store.write_report(inference)}
            ),
        ),
        request=FitRequest[GitOid](
            input=FitInput[GitOid](
                replicate_index=0, model_ref=authored.revision, data_ref=panel.revision
            )
        ),
        model_checks=(checks, identification, validation),
    )
    zeroes = store.write_array(np.zeros((2, 2, 0)))
    assert model.measurement_clock is not None
    variables = tuple(
        item.observation.resolved(model.measurement_clock) for item in model.indicators
    )
    support_ends = np.broadcast_to(np.array([[0.0], [1.0]]), (2, len(variables)))
    observations = np.zeros((2, 2, len(variables)))
    observations[1] = 7
    observations[1, 0, 0] = np.inf
    mask = np.ones_like(observations, dtype=bool)
    mask[1, 1, 0] = False
    evidence = SimulationEvidence(
        model=GitRef(workspace_id=workspace, revision=fitted.revision, path="model.json"),
        design=SimulationSpec(start="2026-01-01", horizon="1d"),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 1),
        draws=2,
        seed=0,
        state_ids=(),
        parameter_draws={},
        latent_paths=zeroes,
        observations=store.write_array(observations),
        observation_layout={
            "variables": variables,
            "support_start_times": store.write_array(support_ends - 1),
            "support_end_times": store.write_array(support_ends),
            "mask": store.write_array(mask),
        },
    )
    simulation = SimulationReport(
        evidence=evidence,
        law=AuthoredLawProvenance(),
        fit_reliability="not_fitted",
        causal=NotApplicable(reason="No intervention was requested."),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.actions.simulate.read_simulation_report", unexpected_evaluation
    )
    simulation_publication, simulated = publish(
        Applied(
            result=ModelSimulationResult(evidence=evidence),
            effects=ActionEffects(reports={"simulation": store.write_report(simulation)}),
        ),
        request=SimulateRequest[GitOid](
            input=SimulateInput[GitOid](
                simulation=SimulationSpec(start="2026-01-01", horizon="1d"),
                model_ref=fitted.revision,
            )
        ),
    )
    cache = Path(data.cache_dir(workspace))
    cache.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cache)
    monkeypatch.setattr(
        "nof1_causal_lab.actions.output_builder.build_output", unexpected_evaluation
    )

    for owner in (
        "model_checks.read_model_checks",
        "data_checks.read_data_profile",
        "fit.read_inference_report",
        "simulate.read_simulation_report",
    ):
        monkeypatch.setattr(f"nof1_causal_lab.actions.{owner}", unexpected_evaluation)
    assert asyncio.run(journal_activity(fit_publication)) == fit_revision
    assert asyncio.run(journal_activity(simulation_publication)) == simulated
    prepared_reader = ModelReader(workspace, at=prepared.commit_id)
    saved_profile = prepared_reader.data_profile
    assert saved_profile is not None
    assert saved_profile == profile
    edited_reader = ModelReader(workspace, at=edited.commit_id)
    edited_output = edited_reader.model_output()
    assert edited_output is not None
    assert edited_output.checks == edit_checks
    saved_identification = edited_reader.identification()
    assert saved_identification is not None
    assert saved_identification == identification
    saved_validation = edited_reader.validation_report
    assert saved_validation is None
    reader = ModelReader(workspace, at=simulated.commit_id)
    assert reader.fit_result is not None
    assert reader.fit_result.question_checks == checks.question
    assert reader.validation_report == validation
    saved_inference = reader.inference_report
    assert saved_inference is not None
    from nof1_causal_lab.study.action_arrays import resolve_vector

    assert saved_inference.core == inference.core
    assert saved_inference.detail.initial_latent_delta is not None
    assert resolve_vector(
        saved_inference.detail.initial_latent_delta[0], reader.fit_result.arrays
    ) == (1.0, 2.0)
    assert saved_inference.detail.divergent == divergent
    np.testing.assert_array_equal(store.read_array(divergent), [False, True])
    saved_simulation = reader.simulation()
    assert saved_simulation is not None
    assert saved_simulation == simulation
    assert fitted_law_report(store, reader.records, fitted.revision) == inference.core
    from pydantic import TypeAdapter

    from nof1_causal_lab.actions.results import ActionPoll
    from nof1_causal_lab.study.action_outputs import completed_call_json

    def completed_call(workspace_id, revision):
        return TypeAdapter(ActionPoll).validate_json(completed_call_json(workspace_id, revision))

    fitted_response = completed_call(workspace, fit_revision)
    assert fitted_response.status == "success"
    assert fitted_response.action == "fit"
    assert fitted_response.body.question_checks == checks.question
    assert fitted_response.body.inference_report is not None
    assert fitted_response.body.inference_report == saved_inference
    with monkeypatch.context() as isolated:
        isolated.setattr(ModelReader, "snapshot", unexpected_evaluation)
        isolated.setattr(ModelReader, "model_output", unexpected_evaluation)
        isolated.setattr(ModelReader, "fit", unexpected_evaluation)
        response = completed_call(workspace, simulated)
        prepared_response = completed_call(workspace, prepared)
    assert response.status == "success"
    assert response.action == "simulate"
    assert response.body.report is not None
    assert response.body.report == simulation
    assert set(response.body.model_dump()) == {"report", "data", "paths", "arrays"}
    assert prepared_response.status == "success"
    assert prepared_response.action == "prepare_data"
    assert isinstance(prepared_response.body.data, Mapping)
    prepared_history = prepared_response.body.data[panel_metadata().variables[0].id]
    assert isinstance(response.body.data, tuple)
    assert len(response.body.data) == 2
    from nof1_causal_lab.study.action_arrays import resolve_vector

    def values(value):
        return resolve_vector(value, response.body.arrays)

    first, second = response.body.data
    assert type(first[variables[0].id]) is type(prepared_history)
    assert values(first[variables[0].id].values) == (0.0, 0.0)
    assert first[variables[0].id].times == (0.0, 1.0)
    assert values(first[variables[0].id].support_start) == (-1.0, 0.0)
    assert values(second[variables[0].id].values) == (None, None)
    assert values(second[variables[1].id].values) == (7.0, 7.0)
    np.testing.assert_array_equal(
        store.read_array(saved_simulation.evidence.latent_paths), np.zeros((2, 2, 0))
    )
    assert (
        repository.read_file(fit_revision.commit_id, "result.json")
        == store.repo[pygit2.Oid(hex=fit_revision.record.attempt.outcome.result)]
        .peel(pygit2.Blob)
        .data
    )
    assert all(record.record.trace_ids == () for record in repository.attempts())


def test_missing_saved_reports_remain_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    question_root("ABSENT")
    repository, store = StudyRepository("ABSENT"), ArtifactStore("ABSENT")
    model = store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={},
        json_files={"model.json": x_y_model().model_dump(mode="json", round_trip=True)},
    )
    repository.append(
        applied_record(
            store.workspace_id,
            Applied(result=None, effects=ActionEffects(produced=(model,))),
            seq=2,
        )
    )
    monkeypatch.setattr(
        "nof1_causal_lab.actions.model_checks.read_model_checks",
        lambda *_args, **_kwargs: pytest.fail("Missing reports must not trigger evaluation"),
    )
    reader = ModelReader("ABSENT", at=repository.head())
    output = reader.model_output()
    assert output is not None
    assert output.checks is None
    assert reader.identification() is None
