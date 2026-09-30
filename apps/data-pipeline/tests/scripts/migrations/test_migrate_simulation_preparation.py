"""Offline migration preserves old fit coordinates and rekeys dependent evidence."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scripts.migrations.migrate_edge_timing import migrate_workspace as migrate_timing
from scripts.migrations.migrate_simulation_preparation import migrate_workspace

from nof1_causal_lab.actions.data_checks import read_data_metadata
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from tests.data_fixtures import simulation_layout
from tests.inference_fixtures import inference_log
from tests.integration.transition_runner_fixtures import (
    panel_frame,
    panel_metadata,
    scientific_model,
)

pytestmark = pytest.mark.contract


def test_format5_migration_preserves_actual_fit_origin_and_paired_draws(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source, destination = tmp_path / "old" / "study", tmp_path / "new" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    store, history = ArtifactStore("study"), StudyRepository("study")
    model = scientific_model().revised(time_points=(-1.0, 0.0, 1.0))
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": panel_frame(n_days=2)},
        json_files={
            "metadata.json": panel_metadata().model_dump(mode="json", exclude={"time_origin"})
        },
    )
    history.append(
        TransitionRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            action="prepare_data",
            operation_id="measurements",
            status="applied",
            produced=[definition, panel],
            trace_ids=[],
            resume=None,
        )
    )
    fitted = store.write_artifact(
        "model",
        derived_from={"model": definition.revision, "panel": panel.revision},
        produced_by="run:posterior",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    fit = inference_log(model, revision=fitted.revision, prior_revision=definition.revision, seq=2)
    report = dict(fit.diagnostics["report"])
    report.pop("time_origin")
    fit = fit.model_copy(
        update={
            "produced": [fitted],
            "diagnostics": {
                **fit.diagnostics,
                "report": report,
                "input_pins": {"model": definition.revision, "panel": panel.revision},
            },
        }
    )
    fit_commit = history.append(fit)
    action = np.array([[[1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [4.0, 5.0]]])
    reference = action - np.array([1.0, 3.0])[:, None, None]
    observations = action.copy()
    observations[:, 0] = np.nan
    layout = simulation_layout(model, (5.0, 6.0), np.isfinite(observations), store.write_array)
    old_report = {
        "model": {"workspace_id": "study", "revision": fitted.revision, "path": "model.json"},
        "design": {
            "start": 5.0,
            "end": 6.0,
            "interventions": [{"target": model.state_order[0], "time": 5.0, "value": 1.0}],
        },
        "times": [5.0, 6.0],
        "draws": 2,
        "seed": 0,
        "state_ids": list(model.state_order),
        "parameter_draws": {},
        "latent_paths": store.write_array(action),
        "observations": store.write_array(observations),
        "reference_latent_paths": store.write_array(reference),
        "reference_observations": store.write_array(observations - 1),
        "observation_layout": layout.model_dump(mode="json"),
        "findings": [],
        "law": None,
        "causal_unavailable_reason": None,
        "causal_result": {
            "outcome": model.state_order[1],
            "labels": {c.id: c.name for c in model.constructs},
            "summary": {
                "mean": 2.0,
                "median": 2.0,
                "lower_95": 1.05,
                "upper_95": 2.95,
                "prob_positive": 1.0,
            },
            "effect_trajectory": [{"day": 1.0, "effect": 2.0}],
            "trajectory_peak": {"day": 1.0, "effect": 2.0},
            "time_grid_days": [0.0, 1.0],
            "trajectories": {},
            "reference_mean": 2.0,
            "warnings": [],
            "manifest_effects": None,
        },
    }
    simulation_commit = history.append(
        TransitionRecord(
            seq=3,
            ts="2026-01-01T02:00:00Z",
            action="simulate",
            operation_id="simulate",
            status="applied",
            diagnostics={"report": old_report},
            trace_ids=[],
            resume=None,
        )
    )
    history.create_branch("simulation", at=simulation_commit)
    origin = datetime(1970, 1, 1)
    synthetic = panel_frame(n_days=2).with_columns(
        pl.Series(
            "anchor_time",
            [
                origin + timedelta(days=5),
                origin + timedelta(days=5),
                origin + timedelta(days=6),
                origin + timedelta(days=6),
            ],
        ),
        pl.Series(
            "support_start",
            [None, None, origin + timedelta(days=5), origin + timedelta(days=5)],
            dtype=pl.Datetime,
        ),
        pl.Series(
            "support_end",
            [None, None, origin + timedelta(days=6), origin + timedelta(days=6)],
            dtype=pl.Datetime,
        ),
        pl.Series("value", [None, None, 2.0, 3.0]),
    )
    replica = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:simulated_measurements",
        parquet_files={"panel.parquet": synthetic},
        json_files={
            "metadata.json": {
                "source": {"revision": simulation_commit, "replicate": 0},
                "preparation": None,
                "variables": [v.model_dump(mode="json") for v in layout.variables],
            }
        },
    )
    head = history.append(
        TransitionRecord(
            seq=4,
            ts="2026-01-01T03:00:00Z",
            action="prepare_data",
            operation_id="simulated_measurements",
            status="applied",
            produced=[replica],
            inputs={"input": {"revision": simulation_commit, "replicate": 0}},
            trace_ids=[],
            resume=None,
        )
    )
    history.repo.config["nof1.format"] = 4
    mapping = migrate_workspace(source, destination)
    assert history.head() == head
    assert history.repo.config.get_int("nof1.format") == 4
    from scripts.migrations.migrate_fixed_values import migrate_workspace as migrate_format7

    current = tmp_path / "current" / "study"
    format6 = tmp_path / "format6" / "study"
    timing_mapping = migrate_timing(destination, format6)
    literal_mapping = migrate_format7(format6, current)
    mapping = {old: literal_mapping[timing_mapping[new]] for old, new in mapping.items()}
    destination = current
    monkeypatch.setattr(data, "_DATA_URI", str(destination.parent))
    updated, updated_store = StudyRepository("study"), ArtifactStore("study")
    assert updated.head() == mapping[head]
    assert updated.branches()["simulation"] == mapping[simulation_commit]
    prepared = read_data_metadata(updated_store, GitOid(mapping[panel.revision]))
    assert prepared.time_origin == datetime(2024, 1, 1, tzinfo=UTC)
    fitted_report = updated.record(mapping[fit_commit]).diagnostics["report"]
    assert fitted_report["time_origin"] == "2024-01-02T00:00:00+00:00"
    assert updated_store.read_json_file("model", mapping[fitted.revision], "model.json")[
        "time_points"
    ] == [-1.0, 0.0, 1.0]
    saved = SimulationReport.model_validate(
        updated.record(mapping[simulation_commit]).diagnostics["report"]
    )
    assert saved.model.revision == mapping[fitted.revision]
    assert saved.time_origin == datetime(2024, 1, 2, tzinfo=UTC)
    assert saved.causal_result is not None
    assert [p.day for p in saved.causal_result.effect_trajectory] == [5, 6]
    assert saved.causal_result.effect_trajectory[-1].lower_95 == pytest.approx(1.05)
    assert saved.causal_result.summary.model_dump() == old_report["causal_result"]["summary"]
    migrated_replica = read_data_metadata(updated_store, GitOid(mapping[replica.revision]))
    assert migrated_replica.time_origin == datetime(2024, 1, 7, tzinfo=UTC)
    assert isinstance(migrated_replica.source, SimulationReplicateRef)
    assert migrated_replica.source.revision == mapping[simulation_commit]
    assert updated_store.read_parquet_file("panel", mapping[replica.revision], "panel.parquet")[
        "anchor_time"
    ].min() == datetime(2024, 1, 7)
    for path in (source / "store").rglob("*"):
        if path.is_file():
            assert (destination / path.relative_to(source)).read_bytes() == path.read_bytes()


def test_imported_panel_requires_a_decision_before_any_copy(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source, destination = tmp_path / "old" / "study", tmp_path / "new" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    store = ArtifactStore("study")
    store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:imported_measurements",
        json_files={"metadata.json": {"source": {"file": "import.parquet"}, "variables": []}},
    )
    store.repo.config["nof1.format"] = 4
    with pytest.raises(ValueError, match="no files recipe"):
        migrate_workspace(source, destination)
    assert not destination.exists()


def test_raised_import_attempt_is_rejected_before_any_copy(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source, destination = tmp_path / "old" / "study", tmp_path / "new" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    history = StudyRepository("study")
    # The retired operation cannot be constructed with the current runtime schema.
    attempt = TransitionRecord(
        seq=1,
        ts="2026-01-01T00:00:00Z",
        action="prepare_data",
        status="raised",
        inputs={"input": {"source": {"file": "import.parquet"}}},
        error_type="ValueError",
        error_message="Invalid imported observations",
        trace_ids=[],
        resume=None,
    ).model_copy(update={"operation_id": "imported_measurements"})
    history.append(attempt)
    history.repo.config["nof1.format"] = 4
    with pytest.raises(ValueError, match="Retired observation-table attempts"):
        migrate_workspace(source, destination)
    assert not destination.exists()
