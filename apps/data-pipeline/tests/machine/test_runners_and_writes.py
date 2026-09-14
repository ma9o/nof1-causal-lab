"""Authoring commits, optional extraction outputs, and atomic derivation cascades."""

import asyncio
import json

import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import RunOperation, apply_transition, input_pins
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import execute_write
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.models.model_inputs import observation_input
from tests.helpers import make_model


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "test_workspace"


def _model():
    model = make_model(["Stress", "Perf"], [("Stress", "Perf")])
    return model.revised(
        question="does stress hurt performance?", default_outcome=model.constructs[1].id
    )


def _exact_measurement(model):
    from nof1_causal_lab.artifacts.expressions import state
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLawSpec

    owner = model.constructs[0]
    indicator = owner.indicators[0].model_copy(
        update={
            "likelihood": LikelihoodSpec(
                law=ObservationLawSpec(distribution="Delta", arguments={"v": state(owner.id)}),
                reasoning="An exact observation of the construct",
            )
        }
    )
    return model.revised(
        edges=replace_constructs(
            model.edges,
            (
                owner.model_copy(update={"indicators": (indicator,)}),
                *model.constructs[1:],
            ),
        )
    )


def _write(store, artifact_id, payload, pins=None):
    from nof1_causal_lab.machine.artifact_files import artifact_file_spec

    return store.write_version(
        artifact_id,
        provenance="human",
        derived_from=pins or {},
        produced_by=None,
        json_files={next(iter(artifact_file_spec(artifact_id).json.values())): payload},
    )


def test_identification_records_positive_and_absent_queries():
    report = identify_model(_model())
    assert report.estimable_treatments == [_model().constructs[0].id]
    assert report.outcome == _model().constructs[1].id
    absent = identify_model(_model().revised(default_outcome=None))
    assert absent.outcome is None
    assert absent.estimable_treatments == []


def test_identification_preserves_negative_findings(monkeypatch):
    from nof1_causal_lab.models import identification

    monkeypatch.setattr(
        identification,
        "check_identifiability",
        lambda *_, **__: {
            "identifiable_treatments": {},
            "non_identifiable_treatments": {"Stress": {"confounders": [], "notes": "Unidentified"}},
        },
    )
    report = identify_model(_model())
    assert not report.estimable_treatments
    assert report.non_identifiable[_model().constructs[0].id].notes == "Unidentified"


@pytest.mark.parametrize("nonempty", [False, True])
def test_extraction_optional_panel(workspace, tmp_path, nonempty):
    from nof1_causal_lab.machine.temporal.measurement_activities import (
        finalize_measurements_activity,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        ExtractionChunkResult,
        MeasurementsFinalizeInput,
    )

    store = ArtifactStore(workspace)
    model = _model()
    state = EpisodeState().with_versions(
        [
            store.write_version(
                "raw_data", provenance="computed", derived_from={}, produced_by="run:raw_data"
            ),
            _write(store, "model", model.model_dump(mode="json")),
        ]
    )
    pins = input_pins(state, transition_spec("measurements"))
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "measurement_structure": observation_input(model),
                "computed_dicts": [
                    {
                        "indicator_id": model.indicators[0].id,
                        "value": 3.0,
                        "timestamp": "2026-01-01",
                    }
                ]
                if nonempty
                else [],
                "chunks": [{"worker_id": 7}],
            }
        )
    )
    effects = asyncio.run(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id=workspace,
                state=state,
                run_id="run-1",
                plan_ref=str(path),
                pins=pins,
                chunk_results=[
                    ExtractionChunkResult(
                        worker_id=7,
                        status="failed",
                        n_extractions=0,
                        n_windows=1,
                        error="No usable extraction",
                    )
                ],
            )
        )
    )
    assert ("panel" in {info.artifact_id for info in effects.produced}) == nonempty
    assert "measurements" not in {info.artifact_id for info in effects.produced}
    assert effects.diagnostics["workers"] == [
        {
            "worker_id": 7,
            "status": "failed",
            "n_extractions": 0,
            "n_windows": 1,
            "error": "No usable extraction",
            "n_llm_calls": 0,
        }
    ]
    assert effects.diagnostics["n_observations"] == int(nonempty)
    from nof1_causal_lab.episode_api import _needs_run, _next_auto_move
    from nof1_causal_lab.machine.store import EpisodeJournal, TransitionRecord

    journal = EpisodeJournal(workspace)
    record = TransitionRecord(
        seq=1,
        ts="2026-01-01T00:00:00Z",
        trace_ids=[],
        resume=None,
        move=RunOperation(operation_id="measurements"),
        status="applied",
        produced=effects.produced,
        retracted=effects.retracted,
        diagnostics=effects.diagnostics,
    )
    journal.append(record)
    assert journal.read_all()[0].diagnostics == effects.diagnostics
    current = apply_transition(state, effects.produced, effects.retracted)
    next_move = _next_auto_move(workspace, current)
    assert next_move is None or next_move.operation_id != "measurements"
    if not nonempty:
        assert not current.has("panel")
        assert not current.has("validation_report")
        assert next_move is None
        enriched = current.current["model"].model_copy(update={"version": 2})
        assert not _needs_run(
            current.with_versions([enriched]), transition_spec("measurements"), model, record
        )
        changed_extraction = enriched.model_copy(update={"model_inputs": {"extraction": "changed"}})
        assert _needs_run(
            current.with_versions([changed_extraction]),
            transition_spec("measurements"),
            model,
            record,
        )
        changed = current.with_versions(
            [current.current["raw_data"].model_copy(update={"version": 2})]
        )
        assert _next_auto_move(workspace, changed) == RunOperation(operation_id="measurements")
    if nonempty:
        panel = next(info for info in effects.produced if info.artifact_id == "panel")
        assert panel.derived_from == {"raw_data": 1, "model": 1}


def test_model_write_cascades_without_parallel_scientific_catalogs(workspace):
    effects = execute_write(
        workspace,
        "model",
        _model().model_dump(mode="json"),
        "human",
        EpisodeState(),
        expected_model_version=0,
    )
    assert {info.artifact_id for info in effects.produced} == {
        "model",
        "identification_report",
    }
    assert not effects.retracted
    assert all(
        info.derived_from == {"model": 1}
        for info in effects.produced
        if info.artifact_id != "model"
    )
    assert next(info for info in effects.produced if info.artifact_id == "model").derived_from == {}


def test_exact_measurement_preserves_execution_layout(workspace):
    from nof1_causal_lab.utils.model_structure import get_state_names

    model = _exact_measurement(_model())
    effects = execute_write(
        workspace,
        "model",
        model.model_dump(mode="json"),
        "human",
        EpisodeState(),
        expected_model_version=0,
    )
    store = ArtifactStore(workspace)
    info = next(info for info in effects.produced if info.artifact_id == "model")
    payload = store.read_json_file("model", info.version, "model.json")
    plan = ModelSpec.model_validate(payload)
    assert get_state_names(plan) == ["Stress", "Perf"]
    assert plan.indicators[0].likelihood is not None
    assert plan.indicators[0].likelihood.law.family == "delta"


@pytest.mark.parametrize("operation", ["latent_structure", "measurement_structure"])
def test_authoring_finalizer_commits_canonical_revision(workspace, tmp_path, operation):
    from nof1_causal_lab.machine.temporal.messages import SingleLLMTransitionFinalizeInput
    from nof1_causal_lab.machine.temporal.model_authoring import finalize_model_revision

    store = ArtifactStore(workspace)
    initial = _model()
    base = initial if operation == "measurement_structure" else ModelSpec(question=initial.question)
    infos = [_write(store, "model", base.model_dump(mode="json"))]
    if operation == "measurement_structure":
        infos.extend(
            [
                store.write_version(
                    "raw_data", provenance="computed", derived_from={}, produced_by="run:raw_data"
                ),
            ]
        )
    state = EpisodeState().with_versions(infos)
    candidate = _exact_measurement(initial)
    path = tmp_path / "model-result.json"
    path.write_text(candidate.model_dump_json())
    pins = input_pins(state, transition_spec(operation))
    effects = finalize_model_revision(
        SingleLLMTransitionFinalizeInput(
            workspace_id=workspace,
            transition_id=operation,
            state=state,
            pins=pins,
            context_ref="unused-context.json",
            result_ref=str(path),
        ),
        operation,
    )
    info = next(info for info in effects.produced if info.artifact_id == "model")
    assert info.provenance == "computed"
    assert info.derived_from == pins
    persisted = ModelSpec.model_validate(store.read_json_file("model", info.version, "model.json"))
    assert persisted == candidate
    assert "llm_trace_ref" not in persisted.model_dump()


def test_model_edit_retracts_validation_of_stale_extraction(workspace):
    store = ArtifactStore(workspace)
    model = _model()
    effects = execute_write(
        workspace,
        "model",
        model.model_dump(mode="json"),
        "human",
        EpisodeState(),
        expected_model_version=0,
    )
    state = apply_transition(EpisodeState(), effects.produced)
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame()},
    )
    validation = _write(
        store,
        "validation_report",
        {"is_valid": True, "indicators": {}, "dataset_issues": []},
        {"panel": 1, "model": 1},
    )
    state = state.with_versions([panel, validation])
    changed = model.revised(measurement_clock="2d")
    effects = execute_write(
        workspace,
        "model",
        changed.model_dump(mode="json"),
        "human",
        state,
        expected_model_version=1,
    )
    assert "validation_report" in {item.artifact_id for item in effects.retracted}
    assert "panel" not in {item.artifact_id for item in effects.produced}


def test_failed_cascade_removes_all_new_versions(workspace, monkeypatch):
    from nof1_causal_lab.models import identification

    def fail(*_args, **_kwargs):
        raise RuntimeError("identification failed")

    monkeypatch.setattr(identification, "check_identifiability", fail)
    with pytest.raises(RuntimeError, match="identification failed"):
        execute_write(
            workspace,
            "model",
            _model().model_dump(mode="json"),
            "human",
            EpisodeState(),
            expected_model_version=0,
        )
    store = ArtifactStore(workspace)
    assert store.list_versions("model") == []


def test_invalid_model_rejected_before_any_write(workspace):
    with pytest.raises(ArtifactWriteRejected):
        execute_write(
            workspace,
            "model",
            {"constructs": [{"id": "construct:invalid"}]},
            "human",
            EpisodeState(),
            expected_model_version=0,
        )
    assert ArtifactStore(workspace).list_versions("model") == []


def test_partial_version_files_removed_on_write_failure(workspace, monkeypatch):
    from nof1_causal_lab.utils import storage

    original = storage.write_text

    def fail_meta(path, value):
        if path.endswith("meta.json"):
            raise OSError("cannot write metadata")
        return original(path, value)

    monkeypatch.setattr(storage, "write_text", fail_meta)
    with pytest.raises(OSError, match="metadata"):
        _write(ArtifactStore(workspace), "model", _model().model_dump(mode="json"))
    assert ArtifactStore(workspace).list_versions("model") == []


def test_question_write_requires_text(workspace):
    with pytest.raises(ArtifactWriteRejected):
        execute_write(
            workspace,
            "model",
            {"question": "   "},
            "human",
            EpisodeState(),
            expected_model_version=0,
        )


def test_binary_artifacts_not_directly_writable(workspace):
    with pytest.raises(ArtifactWriteRejected, match="no write executor"):
        execute_write(workspace, "identification_report", {"anything": 1}, "human", EpisodeState())
