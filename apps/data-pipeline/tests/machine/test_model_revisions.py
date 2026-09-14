"""Whole scientific values commit atomically and findings retain exact source revisions."""

from typing import TYPE_CHECKING

import pytest

from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.moves import is_stale
from nof1_causal_lab.machine.writes import execute_write
from nof1_causal_lab.models.model_inputs import input_fingerprints
from tests.helpers import make_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "additive-revisions"


def test_full_model_write_checks_base_before_writing(workspace):
    effects = execute_write(
        workspace,
        "model",
        make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
        "human",
        EpisodeState(),
        expected_model_version=0,
    )
    state = EpisodeState().with_versions(effects.produced)
    assert state.current["model"].version == 1
    assert state.has("identification_report")
    with pytest.raises(ArtifactWriteRejected, match="conflict"):
        execute_write(
            workspace,
            "model",
            make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
            "human",
            state,
            expected_model_version=0,
        )
    from nof1_causal_lab.machine.store import ArtifactStore

    assert ArtifactStore(workspace).list_versions("model") == [1]
    second = execute_write(
        workspace,
        "model",
        make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
        "human",
        state,
        expected_model_version=1,
    )
    assert [info.artifact_id for info in second.produced] == ["model"]
    assert second.produced[0].version == 2
    assert state.current["identification_report"].derived_from == {"model": 1}


def test_model_input_identity_preserves_findings_and_original_pins():
    values = input_fingerprints(make_model(["X", "Y"], [("X", "Y")]))
    model = ArtifactVersionInfo(
        artifact_id="model", version=2, provenance="human", model_inputs=values
    )
    extraction = ArtifactVersionInfo(
        artifact_id="panel",
        version=1,
        provenance="computed",
        derived_from={"model": 1},
        consumed_model_inputs={"extraction": values["extraction"]},
    )
    state = EpisodeState().with_versions([model, extraction])
    assert state.matches_inputs("panel", "model")
    assert not is_stale(state, "panel")
    assert extraction.derived_from == {"model": 1}
    changed = model.model_copy(
        update={"version": 3, "model_inputs": {**values, "extraction": "changed"}}
    )
    assert is_stale(state.with_versions([changed]), "panel")


def test_statistical_enrichment_preserves_structural_and_measurement_inputs():
    from tests.helpers import complete_test_model, make_model

    measured = make_model(["X", "Y"], [("X", "Y")])
    specified = complete_test_model(measured)
    before, after = input_fingerprints(measured), input_fingerprints(specified)
    for purpose in ("extraction", "identification"):
        assert before[purpose] == after[purpose]
    assert before["compilation"] != after["compilation"]

    changed = input_fingerprints(
        measured.revised(
            edges=(
                measured.edges[0].model_copy(update={"description": "Revised causal assumption"}),
            )
        )
    )
    assert changed["extraction"] == before["extraction"]
    for purpose in ("identification", "compilation"):
        assert changed[purpose] != before[purpose]

    revised_question = input_fingerprints(measured.revised(question="Does X change Y?"))
    assert revised_question["extraction"] != before["extraction"]
    for purpose in ("identification", "compilation", "belief"):
        assert revised_question[purpose] == before[purpose]


@pytest.mark.parametrize("failed_artifact", ["identification_report", None])
def test_statistical_finalizer_commits_model_and_findings_together(
    workspace, monkeypatch, failed_artifact
):
    from types import SimpleNamespace

    import polars as pl
    from temporalio.exceptions import ApplicationError

    from nof1_causal_lab.artifacts.prior import PriorValidationResult
    from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult
    from nof1_causal_lab.flows import runtime_events
    from nof1_causal_lab.flows.transitions.model_spec import assembly
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.machine.temporal import statistical_model_spec_activities as activities
    from nof1_causal_lab.machine.temporal.messages import StatisticalModelSpecFinalizeInput
    from nof1_causal_lab.models.ssm.construct_admission import AdmissionState
    from tests.helpers import complete_test_model, make_model, run_async

    model = complete_test_model(make_model(["X"]))
    plan = model
    store = ArtifactStore(workspace)
    original = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )

    state = EpisodeState().with_versions([original])

    pins: dict[ArtifactId, int] = {"model": original.version}
    checkpoint = SimpleNamespace(
        input_pins=pins,
        full_model_validated=True,
        accepted_constructs=[SimpleNamespace(construct_name="X", results=[])],
    )
    authoring = SimpleNamespace(
        admission=AdmissionState(model=model, names=("X",)),
        model=plan,
        data_for_model=pl.DataFrame(),
        search_queries={"parameter:test": "prior calibration study"},
    )
    monkeypatch.setattr(activities, "read_model_spec_checkpoint", lambda *_: checkpoint)
    monkeypatch.setattr(
        activities, "_load_model_spec_inputs", lambda *_: ("question", plan, pl.DataFrame(), None)
    )
    monkeypatch.setattr(activities, "restore_construct_state", lambda *_args, **_kwargs: authoring)
    monkeypatch.setattr(activities, "_read_model_spec_json", lambda *_: {"indicator_audits": None})
    monkeypatch.setattr(runtime_events, "emit_model_spec_admission_event", lambda *_: None)
    monkeypatch.setattr(
        assembly,
        "materialize_model_spec_result",
        lambda **_: (
            model,
            PriorPredictiveResult(samples={}),
            [
                PriorValidationResult(
                    parameter="parameter:test",
                    is_valid=True,
                    origin="compile",
                    severity="warning",
                    code="test_warning",
                    issue="Check the timescale",
                )
            ],
        ),
    )
    write = ArtifactStore.write_version

    def fail_selected(self, artifact_id, **kwargs):
        if artifact_id == failed_artifact:
            raise OSError("injected final write failure")
        return write(self, artifact_id, **kwargs)

    monkeypatch.setattr(ArtifactStore, "write_version", fail_selected)
    request = StatisticalModelSpecFinalizeInput(
        workspace_id=workspace,
        run_id="test",
        state=state,
        pins=pins,
        checkpoint_ref="checkpoint",
        context_ref="context",
    )
    if failed_artifact:
        with pytest.raises(ApplicationError, match="injected final write failure"):
            run_async(activities.finalize_statistical_model_spec_activity(request))
        assert store.list_versions("model") == [1]
        assert store.list_versions("identification_report") == []
    else:
        effects = run_async(activities.finalize_statistical_model_spec_activity(request))
        outputs = {item.artifact_id: item for item in effects.produced}
        assert outputs["model"].version == 2
        assert "admission_report" not in outputs
        assert effects.diagnostics["prior_predictive"] == {"samples": {}, "diagnostics": []}
        assert effects.diagnostics["search_queries"] == authoring.search_queries
        assert effects.diagnostics["validation_diagnostics"][0]["code"] == "test_warning"
        assert effects.diagnostics["validation_diagnostics"][0]["severity"] == "warning"
        model.check_execution()
    assert state.current["model"] == original
