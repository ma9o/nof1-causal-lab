"""Model-panel limitations are findings on edits and preflight errors on fits."""

import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
from nof1_causal_lab.models.ssm.runtime import bind_panel
from nof1_causal_lab.sampler_config import (
    SamplerSpec,
)
from nof1_causal_lab.study.records import DataPreparationResult
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import edit_and_check
from tests.integration.runner_fixtures import (
    panel_frame,
    panel_metadata,
)
from tests.model_fixtures import compile_fit_fixture

pytestmark = pytest.mark.contract


def test_specification_reports_each_distinct_fit_law_reason_once(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import inputs as compilation
    from nof1_causal_lab.models.ssm.compile import prior_compilation

    def unsupported(_compiled, _model):
        raise prior_compilation.PriorCompilationError(
            ["Unsupported joint law.", "Unsupported joint law.", "Invalid scale."]
        )

    monkeypatch.setattr(compilation, "compile_priors", unsupported)
    report = check_specification(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/models"
                / "common/stress_sleep_model.json"
            ).read_text()
        )
    )
    finding = next(finding for finding in report.findings if finding.subject == "fit_laws")
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert finding.evidence.count("Unsupported joint law.") == 1
    assert finding.evidence.count("Invalid scale.") == 1


def test_interval_summary_fails_shared_preflight_before_particle_dispatch():
    model, panel = (
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/models"
                / "common/stress_sleep_model.json"
            ).read_text()
        ),
        panel_frame(n_days=4),
    )
    report = check_model_data(model, panel, time_origin=panel_metadata().time_origin)
    finding = report.findings[0]
    assert finding.subject == "fit_preflight"
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert "interval summaries" in finding.evidence
    assert "stress_score" in finding.evidence
    runtime = bind_panel(
        panel, model=compile_fit_fixture(model).compiled, time_origin=panel_metadata().time_origin
    )
    from nof1_causal_lab.models.ssm.runtime import BoundPanel

    assert isinstance(runtime, BoundPanel)
    with pytest.raises(ObservationPreflightError, match="interval summaries"):
        fit(
            compile_fit_fixture(model).prior_runtime_bundle,
            runtime,
            sampler=SamplerSpec(),
            clock=time.monotonic,
        )


def test_edit_with_missing_panel_variable_saves_compatibility_findings(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    full = panel_metadata()
    metadata = full.revised(
        variables=full.variables[:1],
        preparation=full.preparation.revised(variables=full.preparation.variables[:1]),
    )
    panel = panel_frame(n_days=4).filter(pl.col("indicator_id") == metadata.variables[0].id)
    record = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": panel},
    )
    prepared = evaluate_data_checks("TEST", StudyState(), DataPreparationResult(produced=[record]))
    state = StudyState().with_artifacts(prepared.produced)
    edited = edit_and_check(
        "TEST",
        EditModelRequest(
            expected_revision=None,
            model=ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[1]
                    / "fixtures/models"
                    / "common/stress_sleep_model.json"
                ).read_text()
            ),
        ),
        state,
    )
    assert "model" in {item.artifact_id for item in edited.produced}
    validation = next(item for item in edited.produced if item.artifact_id == "validation_report")
    report = ValidationReportArtifact.model_validate(
        store.read_json_file("validation_report", validation.revision, "validation_report.json")
    )
    finding = report.preflight.findings[0]
    assert finding.subject == "fit_preflight"
    assert finding.kind == "evaluated"
    assert finding.outcome == "failed"
    assert "missing model indicators" in finding.evidence
    assert "sleep_score" in finding.evidence
    assert edited.checks.predictive.reason == "NO_COMPATIBLE_PANEL"
    messages = completion_messages(
        edited,
        datetime.now(UTC),
        store.completion_reports(edited.produced),
    )
    assert "MODEL_DATA_INCOMPATIBLE" in {message.label for message in messages}
