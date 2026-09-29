"""Model-panel limitations are findings on edits and preflight errors on fits."""

from datetime import UTC, datetime

import polars as pl
import pytest

from nof1_causal_lab.actions.checks import check_model_data
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.artifacts.data_preparation import ObservationTableRef, PreparedDataMetadata
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import TransitionEffects
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime
from tests.action_fixtures import edit_and_check
from tests.integration.transition_runner_fixtures import (
    panel_frame,
    panel_metadata,
    scientific_model,
)

pytestmark = pytest.mark.contract


def test_interval_summary_fails_shared_preflight_before_particle_dispatch():
    model, panel = scientific_model(), panel_frame(n_days=4)
    report = check_model_data(model, panel)
    finding = report.findings[0]
    assert finding.check == "fit_preflight"
    assert finding.status == "failed"
    assert "interval summaries" in finding.message
    assert "stress_score" in finding.message
    runtime = prepare_model_runtime(panel, model_spec=model)
    with pytest.raises(ObservationPreflightError, match="interval summaries"):
        fit(runtime.model, runtime.observations, runtime.times)


def test_edit_with_missing_panel_variable_saves_compatibility_findings(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    metadata = PreparedDataMetadata(
        source=ObservationTableRef(file="panel.parquet"),
        variables=panel_metadata().variables[:1],
    )
    panel = panel_frame(n_days=4).filter(pl.col("indicator_id") == metadata.variables[0].id)
    record = store.write_artifact(
        "panel",
        produced_by="run:imported_measurements",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": panel},
    )
    prepared = evaluate_data_checks("TEST", EpisodeState(), TransitionEffects(produced=[record]))
    state = EpisodeState().with_artifacts(prepared.produced)
    edited = edit_and_check(
        "TEST", EditModelRequest(expected_revision=None, model=scientific_model()), state
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
