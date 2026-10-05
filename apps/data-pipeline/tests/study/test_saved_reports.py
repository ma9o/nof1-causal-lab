"""Action findings survive cache eviction and are never evaluated by result readers."""

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest, FitRequest, SimulateRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.temporal.activities import journal_activity
from nof1_causal_lab.actions.temporal.messages import AttemptPublication
from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import DistributionId, GitRef
from nof1_causal_lab.artifacts.model_checks import (
    ModelCheckReport,
    ModelPredictiveReport,
    QuestionCheckReport,
    UnavailablePredictiveChecks,
)
from nof1_causal_lab.artifacts.posterior import (
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from nof1_causal_lab.artifacts.posterior_diagnostics import ParameterConvergenceReport
from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.simulation import (
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
from nof1_causal_lab.study.records import (
    Applied,
    DataPreparationResult,
    ModelFitResult,
    ModelSimulationResult,
)
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data
from tests.action_fixtures import applied_record, question_root
from tests.integration.runner_fixtures import panel_frame, panel_metadata
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
        publication = AttemptPublication(
            workspace_id=workspace,
            parent_id=repository.head(),
            record=applied_record(applied, seq=repository.latest_seq() + 1, request=request),
            model_checks=model_checks,
            data_profile=data_profile,
        )
        return publication, asyncio.run(journal_activity(publication))

    _, prepared = publish(
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=(panel,))),
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
        specification=(Evaluated(subject="model_execution", outcome="passed", evidence="Saved"),),
        question=QuestionCheckReport(
            question_revision=repository.question().revision,
            panel_revision=panel.revision,
            findings=(),
        ),
        predictive=ModelPredictiveReport(
            model_revision=authored.revision,
            panel_revision=panel.revision,
            draws=200,
            seed=0,
            law=AuthoredLawProvenance(),
            evaluation=UnavailablePredictiveChecks(
                reason="SIMULATION_UNSUPPORTED", detail="Saved execution finding"
            ),
        ),
    )
    identification = IdentificationReport(outcome=None)
    validation = ValidationReportArtifact(
        data=profile,
        preflight=(Evaluated(subject="fit_preflight", outcome="failed", evidence="Saved failure"),),
    )
    _, edited = publish(
        Applied(result=None, effects=ActionEffects(produced=(authored,))),
        request=EditModelRequest(
            expected_revision=None, model=model, panel_revision=panel.revision
        ),
        model_checks=(checks, identification, validation),
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
    fit_publication, fit_revision = publish(
        Applied(
            result=ModelFitResult(
                model=GitRef(workspace_id=workspace, revision=authored.revision, path="model.json"),
                panel=GitRef(workspace_id=workspace, revision=panel.revision, path="panel.parquet"),
                evidence=InferenceEvidence(
                    distribution=DistributionId("distribution:retained"),
                    engine=None,
                    time_origin=None,
                    duration_seconds=3,
                ),
            ),
            effects=ActionEffects(
                produced=(fitted,), reports={"inference": store.write_report(inference)}
            ),
        ),
        request=FitRequest(model_revision=authored.revision, panel_revision=panel.revision),
        model_checks=(checks.revised(predictive=None), identification, validation),
    )
    zeroes = store.write_array(np.zeros((1, 2, 0)))
    evidence = SimulationEvidence(
        model=GitRef(workspace_id=workspace, revision=fitted.revision, path="model.json"),
        design=SimulationSpec(start="2026-01-01", horizon="1d"),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 1),
        draws=1,
        seed=0,
        state_ids=(),
        parameter_draws={},
        latent_paths=zeroes,
        observations=zeroes,
        observation_layout={
            "variables": [],
            "support_start_times": zeroes,
            "support_end_times": zeroes,
            "mask": zeroes,
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
        request=SimulateRequest(model_revision=fitted.revision, start="2026-01-01", horizon="1d"),
    )
    cache = Path(data.cache_dir(workspace))
    cache.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cache)
    monkeypatch.setattr("nof1_causal_lab.study.store._CODE_DIGEST", "changed-code")

    for owner in (
        "model_checks.read_model_checks",
        "data_checks.read_data_profile",
        "fit.read_inference_report",
        "simulate.read_simulation_report",
        "predictive_checks.check_model_predictive",
    ):
        monkeypatch.setattr(f"nof1_causal_lab.actions.{owner}", unexpected_evaluation)
    assert asyncio.run(journal_activity(fit_publication)) == fit_revision
    assert asyncio.run(journal_activity(simulation_publication)) == simulated
    prepared_reader = ModelReader(workspace, at=prepared.commit_id)
    saved_profile = prepared_reader.data_profile
    assert saved_profile is not None
    assert saved_profile.value == profile
    edited_reader = ModelReader(workspace, at=edited.commit_id)
    assert edited_reader.checks == (checks, identification, validation)
    saved_identification = edited_reader.identification()
    assert saved_identification is not None
    assert saved_identification.value == identification
    saved_validation = edited_reader.validation_report
    assert saved_validation is not None
    assert saved_validation.value == validation
    assert edited_reader.predictive_history(model.indicators[0].observation.id) is None
    reader = ModelReader(workspace, at=simulated.commit_id)
    assert reader.checks == (checks.revised(predictive=None), identification, validation)
    saved_inference = reader.inference_report
    assert saved_inference is not None
    assert saved_inference.value == inference
    saved_simulation = reader.simulation()
    assert saved_simulation is not None
    assert saved_simulation.value == simulation
    assert fitted_law_report(store, reader.records, fitted.revision) == inference.core
    from nof1_causal_lab.study_api import _completed_call

    response = _completed_call(workspace, simulated)
    assert response.checks == checks.revised(predictive=None)
    assert response.inference_report is not None
    assert response.inference_report.value == inference
    assert response.snapshot is not None
    assert response.snapshot.simulation is not None
    assert response.snapshot.simulation.value == simulation
    np.testing.assert_array_equal(
        store.read_array(saved_simulation.value.evidence.latent_paths), np.zeros((1, 2, 0))
    )
    assert ModelCheckReport.model_validate_json(
        repository.read_file(fit_revision.commit_id, "logs/reports/checks.json")
    ) == checks.revised(predictive=None)
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
        applied_record(Applied(result=None, effects=ActionEffects(produced=(model,))), seq=2)
    )
    monkeypatch.setattr(
        "nof1_causal_lab.actions.model_checks.read_model_checks",
        lambda *_args, **_kwargs: pytest.fail("Missing reports must not trigger evaluation"),
    )
    reader = ModelReader("ABSENT", at=repository.head())
    assert reader.checks is None
    assert reader.identification() is None
