"""Authoring commits, optional extraction outputs, and atomic derivation cascades."""

import asyncio
import json

import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import apply_transition, input_pins
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.models.model_inputs import observation_input
from tests.action_fixtures import edit_and_check
from tests.data_fixtures import metadata_for_model
from tests.git_fixtures import artifact_revision
from tests.helpers import make_model

pytestmark = pytest.mark.contract


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

    return store.write_artifact(
        artifact_id,
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
def test_extraction_requires_some_observations(workspace, tmp_path, nonempty):
    from nof1_causal_lab.machine.temporal.measurement_activities import (
        finalize_measurements_activity,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        ExtractionChunkResult,
        MeasurementsFinalizeInput,
    )

    store = ArtifactStore(workspace)
    model = _model()
    state = EpisodeState().with_artifacts(
        [
            store.write_artifact("raw_data", derived_from={}, produced_by="run:raw_data"),
            _write(store, "model", model.model_dump(mode="json")),
        ]
    )
    pins = input_pins(state, transition_spec("measurements"))
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "measurement_structure": observation_input(model),
                "metadata": metadata_for_model(model).model_dump(mode="json"),
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
    pending = finalize_measurements_activity(
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
    if not nonempty:
        from temporalio.exceptions import ApplicationError

        with pytest.raises(ApplicationError, match="Extraction produced no observations"):
            asyncio.run(pending)
        return
    effects = asyncio.run(pending)

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
    from nof1_causal_lab.machine.history import StudyRepository
    from nof1_causal_lab.machine.store import TransitionRecord

    journal = StudyRepository(workspace)
    record = TransitionRecord(
        seq=1,
        ts="2026-01-01T00:00:00Z",
        trace_ids=[],
        resume=None,
        action="prepare_data",
        operation_id="measurements",
        inputs={},
        status="applied",
        produced=effects.produced,
        retracted=effects.retracted,
        diagnostics=effects.diagnostics,
        checks=effects.checks,
    )
    journal.append(record)
    assert journal.attempts()[0].diagnostics == effects.diagnostics
    current = apply_transition(state, effects.produced, effects.retracted)
    if not nonempty:
        assert not current.has("panel")
        assert not current.has("validation_report")
    if nonempty:
        panel = next(info for info in effects.produced if info.artifact_id == "panel")
        assert panel.derived_from == {
            "raw_data": artifact_revision(workspace, "raw_data", 1),
        }


def test_model_write_cascades_without_parallel_scientific_catalogs(workspace):
    effects = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {"model": _model().model_dump(mode="json"), "expected_revision": None}
        ),
        EpisodeState(),
    )
    assert {info.artifact_id for info in effects.produced} == {
        "model",
        "identification_report",
    }
    assert not effects.retracted
    assert all(
        info.derived_from == {"model": artifact_revision(workspace, "model", 1)}
        for info in effects.produced
        if info.artifact_id != "model"
    )
    assert next(info for info in effects.produced if info.artifact_id == "model").derived_from == {}


def test_exact_measurement_preserves_execution_layout(workspace):
    from nof1_causal_lab.utils.model_structure import get_state_names

    model = _exact_measurement(_model())
    effects = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {"model": model.model_dump(mode="json"), "expected_revision": None}
        ),
        EpisodeState(),
    )
    store = ArtifactStore(workspace)
    info = next(info for info in effects.produced if info.artifact_id == "model")
    payload = store.read_json_file("model", info.revision, "model.json")
    plan = ModelSpec.model_validate(payload)
    assert get_state_names(plan) == ["Stress", "Perf"]
    assert plan.indicators[0].likelihood is not None
    assert plan.indicators[0].likelihood.law.family == "delta"


def test_model_edit_reports_stale_extraction(workspace):
    store = ArtifactStore(workspace)
    model = _model()
    effects = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {"model": model.model_dump(mode="json"), "expected_revision": None}
        ),
        EpisodeState(),
    )
    state = apply_transition(EpisodeState(), effects.produced)
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        json_files={"metadata.json": metadata_for_model(model).model_dump(mode="json")},
        parquet_files={"panel.parquet": pl.DataFrame()},
    )
    validation = _write(
        store,
        "validation_report",
        {"is_valid": True, "indicators": {}, "dataset_issues": []},
        {
            "panel": artifact_revision(workspace, "panel", 1),
            "model": artifact_revision(workspace, "model", 1),
        },
    )
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks
    from nof1_causal_lab.machine.execution import TransitionEffects

    checked_data = evaluate_data_checks(workspace, state, TransitionEffects(produced=[panel]))
    state = state.with_artifacts([*checked_data.produced, validation])
    changed = model.revised(measurement_clock="2d")
    effects = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {
                "model": changed.model_dump(mode="json"),
                "expected_revision": artifact_revision(workspace, "model", 1),
            }
        ),
        state,
    )
    report = next(item for item in effects.produced if item.artifact_id == "validation_report")
    payload = store.read_json_file("validation_report", report.revision, "validation_report.json")
    assert any(
        issue["issue_type"] == "measurement_definitions" for issue in payload["dataset_issues"]
    )
    assert "panel" not in {item.artifact_id for item in effects.produced}


def test_failed_check_publishes_no_state(workspace, monkeypatch):
    from nof1_causal_lab.models import identification

    def fail(*_args, **_kwargs):
        raise RuntimeError("identification failed")

    monkeypatch.setattr(identification, "check_identifiability", fail)
    with pytest.raises(RuntimeError, match="identification failed"):
        edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {"model": _model().model_dump(mode="json"), "expected_revision": None}
            ),
            EpisodeState(),
        )
    assert StudyRepository(workspace).state(StudyRepository(workspace).head()).current == {}


def test_invalid_model_rejected_before_any_write(workspace):
    with pytest.raises(ValidationError):
        edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {"model": {"constructs": [{"id": "construct:invalid"}]}, "expected_revision": None}
            ),
            EpisodeState(),
        )
    assert ArtifactStore(workspace).list_revisions("model") == []


def test_failed_tree_write_publishes_no_artifact(workspace, monkeypatch):
    from nof1_causal_lab.machine import store as store_module

    def fail_tree(*_args, **_kwargs):
        raise OSError("cannot write metadata")

    monkeypatch.setattr(store_module, "write_tree", fail_tree)
    with pytest.raises(OSError, match="metadata"):
        _write(ArtifactStore(workspace), "model", _model().model_dump(mode="json"))
    assert ArtifactStore(workspace).list_revisions("model") == []


def test_question_write_requires_text(workspace):
    with pytest.raises(ValidationError):
        edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {"model": {"question": "   "}, "expected_revision": None}
            ),
            EpisodeState(),
        )
