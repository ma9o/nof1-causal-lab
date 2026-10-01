"""Model-panel limitations are findings on edits and preflight errors on fits."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nof1_causal_lab.actions.checks import check_model_data, check_specification
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime
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
    from nof1_causal_lab.models.ssm.compile import prior_compilation

    def unsupported(_model):
        raise prior_compilation.PriorCompilationError(
            ["Unsupported joint law.", "Unsupported joint law.", "Invalid scale."]
        )

    monkeypatch.setattr(prior_compilation, "compile_priors", unsupported)
    report = check_specification(ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_data_preflight/specification_reports_each_distinct_fit_law_reason_once_scientific_model.json').read_text()))
    finding = next(finding for finding in report.findings if finding.check == "fit_laws")
    assert finding.status == "failed"
    assert finding.message.count("Unsupported joint law.") == 1
    assert finding.message.count("Invalid scale.") == 1


def test_interval_summary_fails_shared_preflight_before_particle_dispatch():
    model, panel = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_data_preflight/interval_summary_fails_shared_preflight_before_particle_dispatch_scientific_model.json').read_text()), panel_frame(n_days=4)
    report = check_model_data(model, panel, time_origin=panel_metadata().time_origin)
    finding = report.findings[0]
    assert finding.check == "fit_preflight"
    assert finding.status == "failed"
    assert "interval summaries" in finding.message
    assert "stress_score" in finding.message
    runtime = prepare_model_runtime(
        panel, inputs=compile_fit_fixture(model), time_origin=panel_metadata().time_origin
    )
    with pytest.raises(ObservationPreflightError, match="interval summaries"):
        fit(runtime.model, runtime.observations, runtime.times)


def test_edit_with_missing_panel_variable_saves_compatibility_findings(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    full = panel_metadata()
    metadata = type(full).model_validate(
        {
            **full.model_dump(),
            "variables": full.variables[:1],
            "preparation": type(full.preparation).model_validate(
                {**full.preparation.model_dump(), "variables": full.preparation.variables[:1]}
            ),
        }
    )
    panel = panel_frame(n_days=4).filter(pl.col("indicator_id") == metadata.variables[0].id)
    record = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": panel},
    )
    prepared = evaluate_data_checks("TEST", StudyState(), ActionEffects(produced=[record]))
    state = StudyState().with_artifacts(prepared.produced)
    edited = edit_and_check(
        "TEST", EditModelRequest(expected_revision=None, model=ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_data_preflight/edit_with_missing_panel_variable_saves_compatibility_findings_scientific_model.json').read_text())), state
    )
    assert "model" in {item.artifact_id for item in edited.produced}
    validation = next(item for item in edited.produced if item.artifact_id == "validation_report")
    report = ValidationReportArtifact.model_validate(
        store.read_json_file("validation_report", validation.revision, "validation_report.json")
    )
    finding = report.preflight.findings[0]
    assert finding.check == "fit_preflight"
    assert finding.status == "failed"
    assert "missing model indicators" in finding.message
    assert "sleep_score" in finding.message
    assert edited.checks.predictive.reason == "NO_COMPATIBLE_PANEL"
    messages = completion_messages(
        "TEST",
        "edit_model",
        edited.produced,
        edited.diagnostics,
        datetime.now(UTC),
        checks=edited.checks,
    )
    assert "MODEL_DATA_INCOMPATIBLE" in {message.label for message in messages}
