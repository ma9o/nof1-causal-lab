"""A published result is the response and remains reusable without scientific reads."""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.contracts import EditQuestionRequest, call_identity
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.io import EditQuestionInput, EditQuestionOutput
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.study.action_outputs import completed_call_json
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, AttemptRecord, EditQuestionAttempt
from nof1_causal_lab.study.store import ArtifactStore, read_question
from nof1_causal_lab.utils import data as data_module
from tests.helpers import fixture_entity_id

pytestmark = pytest.mark.contract


def test_poll_and_repeated_post_return_the_saved_result_without_building(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    from nof1_causal_lab import study_api
    from nof1_causal_lab.actions import output_builder

    workspace = "RESULTS"
    store = ArtifactStore(workspace)
    repository = StudyRepository(workspace)
    question = QuestionSpec(
        text="What changes the outcome?", outcome=fixture_entity_id("construct", "outcome")
    )
    request = EditQuestionRequest(input=EditQuestionInput(question=question))
    result = EditQuestionOutput(question=question)
    ref = store.write_result(result)
    staged = store.write_artifact(
        "question",
        derived_from={},
        produced_by="edit_question",
        json_files={"question.json": question.model_dump(mode="json")},
    )
    artifact = store.result_artifact(staged, ref)
    revision = repository.append(
        AttemptRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            attempt=EditQuestionAttempt(
                action="edit_question",
                request=request,
                outcome=Applied(result=ref, effects=ActionEffects(produced=(artifact,))),
            ),
        )
    )
    expected = json.loads(completed_call_json(workspace, revision))
    assert expected["body"] == json.loads(repository.read_file(revision.commit_id, "result.json"))
    assert read_question(store, artifact.revision) == question

    def cannot_build(*args, **kwargs):
        raise AssertionError("A saved call must never construct another result")

    monkeypatch.setattr(output_builder, "build_output", cannot_build)
    monkeypatch.setattr(study_api, "actions_enabled", lambda: True)
    app = FastAPI()

    class UnusedClients:
        async def get(self):
            raise AssertionError("Cached calls do not contact Temporal")

    app.state.study_clients = UnusedClients()
    app.include_router(study_api.router)
    with TestClient(app) as client:
        for _ in range(2):
            polled = client.get(f"/api/studies/{workspace}/edit_question/{call_identity(request)}")
            assert polled.status_code == 200, polled.text
            assert polled.json() == expected
            posted = client.post(
                f"/api/studies/{workspace}/edit_question", json=request.model_dump(mode="json")
            )
            assert posted.status_code == 200, posted.text
            assert posted.json() == expected
    assert repository.latest_seq() == 1


def test_numerical_result_preserves_dtype_shape_missingness_and_vector_selection():
    from nof1_causal_lab.artifacts.arrays import ArrayVector, NumericalArray
    from nof1_causal_lab.study.action_arrays import array_value, decode_array, resolve_vector

    values = np.asarray([[1, np.nan, np.inf], [2, -np.inf, 4]], dtype=np.float64)
    saved = array_value(values)
    loaded = NumericalArray.model_validate_json(saved.model_dump_json())
    np.testing.assert_array_equal(decode_array(loaded), values)
    assert loaded.shape == (2, 3)
    assert loaded.dtype == "float64"
    ref = "a" * 64
    mask = "b" * 64
    selected = ArrayVector(
        array_ref=ref, indices=(None, 0), mask=ArrayVector(array_ref=mask, indices=(None,))
    )
    assert resolve_vector(
        selected, {ref: loaded, mask: array_value(np.asarray([True, False]))}
    ) == (1.0, None)
    empty = np.zeros((0, 3), dtype=np.int32)
    np.testing.assert_array_equal(decode_array(array_value(empty)), empty)


def test_prepared_observations_round_trip_through_the_published_body(monkeypatch, tmp_path):
    import polars as pl
    from polars.testing import assert_frame_equal

    from nof1_causal_lab.actions.contracts import PrepareDataRequest
    from nof1_causal_lab.actions.io import PrepareDataInput
    from nof1_causal_lab.actions.output_builder import complete_attempt
    from nof1_causal_lab.artifacts.data_preparation import FileSourceRef
    from nof1_causal_lab.artifacts.data_ref import DataRef
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.study.data import read_data_history
    from nof1_causal_lab.study.records import DataPreparationResult, applied_attempt
    from nof1_causal_lab.study.state import ArtifactResult
    from tests.action_fixtures import question_root
    from tests.integration.runner_fixtures import panel_frame, panel_metadata, seed_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    workspace = "PREPARED"
    question_root(workspace)
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    model = seed_model(store)
    metadata = panel_metadata()
    frame = panel_frame(n_days=2).with_columns(
        pl.col("anchor_time", "support_start", "support_end").str.to_datetime(),
        pl.when(pl.col("value") == 1.0).then(None).otherwise(pl.col("value")).alias("value"),
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": metadata.model_dump(mode="json")},
        parquet_files={"panel.parquet": frame},
    )
    expected = read_data_history(
        store, DataRef[GitOid, int](revision=panel.revision, replicate_index=0)
    ).observations.recorded.frame
    request = PrepareDataRequest[GitOid, FileSourceRef](
        input=PrepareDataInput[GitOid, FileSourceRef](
            model_ref=model.revision,
            source=metadata.source,
            extraction={
                item.observation.id: item.extraction for item in metadata.preparation.variables
            },
        )
    )
    attempt = complete_attempt(
        workspace,
        applied_attempt(
            request,
            Applied(
                result=DataPreparationResult(),
                effects=ActionEffects(produced=(panel,)),
            ),
        ),
    )
    revision = repository.append(AttemptRecord(seq=2, ts="2026-01-01T00:00:00Z", attempt=attempt))
    assert isinstance(attempt.outcome, Applied)
    published = attempt.outcome.effects.produced[0]
    assert isinstance(published.source, ArtifactResult)
    assert published.source.result == attempt.outcome.result
    assert published.revision != panel.revision
    actual = read_data_history(
        store, DataRef[GitOid, int](revision=revision.commit_id, replicate_index=0)
    ).observations.recorded.frame
    assert_frame_equal(actual, expected)
    assert json.loads(completed_call_json(workspace, revision))["body"] == json.loads(
        repository.read_file(revision.commit_id, "result.json")
    )


def test_model_comparisons_resolve_published_artifact_trees_and_real_file_references(
    monkeypatch, tmp_path
):
    from nof1_causal_lab.actions.contracts import EditModelRequest
    from nof1_causal_lab.actions.edit_model import edit_model
    from nof1_causal_lab.actions.io import EditModelInput
    from nof1_causal_lab.actions.output_builder import complete_attempt
    from nof1_causal_lab.actions.revisions import model_diff
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.study.records import applied_attempt
    from tests.action_fixtures import question_root
    from tests.helpers import make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    workspace = "MODEL_REFS"
    question_root(
        workspace, QuestionSpec(text="What changes Y?", outcome=fixture_entity_id("construct", "Y"))
    )
    store, repository = ArtifactStore(workspace), StudyRepository(workspace)
    model = make_model(["X", "Y"], [("X", "Y")])
    request = EditModelRequest[GitOid](
        input=EditModelInput[GitOid](
            parent_ref=repository.question().revision,
            model=model,
        )
    )
    staged = edit_model(workspace, request)
    assert isinstance(staged, Applied)
    attempt = complete_attempt(workspace, applied_attempt(request, staged))
    repository.append(AttemptRecord(seq=2, ts="2026-01-01T00:00:00Z", attempt=attempt))
    assert isinstance(attempt.outcome, Applied)
    tree = attempt.outcome.effects.produced[0].revision
    comparison = model_diff(workspace, tree, tree)
    assert comparison.before_model == comparison.after_model == model
    assert comparison.before == comparison.after == store.model_ref(tree)
    assert comparison.before is not None
    assert comparison.before.path == "result.json"
    assert json.loads(repository.read_file(tree, comparison.before.path))[
        "model"
    ] == model.model_dump(mode="json")


def test_published_arrays_remain_readable_after_execution_buffers_are_removed(
    monkeypatch, tmp_path
):
    from nof1_causal_lab.actions.contracts import SimulateRequest
    from nof1_causal_lab.actions.io import SimulateInput, SimulateOutput
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.study.action_arrays import array_value
    from nof1_causal_lab.study.records import SimulateAttempt
    from tests.git_fixtures import git_oid

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, repository = ArtifactStore("BUFFERS"), StudyRepository("BUFFERS")
    values = np.asarray([[[1.0, np.nan]], [[np.inf, -np.inf]]], dtype=np.float64)
    identity = store.write_array(values)
    result = SimulateOutput(
        report=None, paths=None, data=(), arrays={identity: array_value(values)}
    )
    ref = store.write_result(result)
    request = SimulateRequest[GitOid](
        input=SimulateInput[GitOid](
            model_ref=git_oid(1), simulation=SimulationSpec(start="2026-01-01", horizon="1d")
        )
    )
    repository.append(
        AttemptRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            attempt=SimulateAttempt(
                action="simulate",
                request=request,
                outcome=Applied(result=ref, effects=ActionEffects()),
            ),
        )
    )
    assert not (tmp_path / "BUFFERS/store/arrays" / f"{identity}.npy").exists()
    np.testing.assert_array_equal(ArtifactStore("BUFFERS").read_array(identity), values)
    assert json.loads(repository.read_file(repository.head(), "result.json"))["arrays"][
        identity
    ] == result.arrays[identity].model_dump(mode="json")
