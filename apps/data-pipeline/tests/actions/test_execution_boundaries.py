"""Partial models stay inspectable; numerical operations validate their own inputs."""

from tests.git_fixtures import artifact_revisions

from typing import TYPE_CHECKING
from unittest.mock import Mock

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest, FitRequest
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record, edit_and_check, question_root
from tests.data_fixtures import metadata_for_model
from tests.git_fixtures import artifact_revision
from tests.helpers import make_model, run_async, write_question
from tests.model_fixtures import x_y_model

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    monkeypatch.setenv("DEPLOYMENT_ENV", "development")
    monkeypatch.setattr(storage, "is_remote", lambda: False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return "EXECUTION"


def test_partial_model_revisions_remain_readable_with_capability_findings(workspace):
    partial = make_model(["X", "Y"], [("X", "Y")])
    complete = x_y_model()
    missing_law = complete.revised(
        distributions={},
        parameters=tuple(p.revised(distribution=None) for p in complete.parameters),
    )
    journal = StudyRepository(workspace)
    state = journal.state(question_root(workspace).commit_id)
    snapshots = []
    for revision, model in (
        (1, partial),
        (2, complete),
        (3, missing_law),
    ):
        effects = edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {
                    "model": model.model_dump(mode="json"),
                    "expected_revision": state.current["model"].revision
                    if state.has("model")
                    else None,
                }
            ),
            state,
        )
        state = state.with_artifacts(effects.effects.produced)
        journal.append(
            applied_record(effects, seq=revision + 1, ts="2026-09-14T12:00:00Z", trace_ids=[])
        )
        snapshot = ModelReader(workspace, at=StudyRepository(workspace).head()).snapshot()
        assert snapshot.model is not None
        assert snapshot.model.value == model
        assert snapshot.specification is not None
        execution = snapshot.specification.value[0]
        assert execution.kind == ("evaluated" if revision == 2 else "not_evaluated")
        if revision == 2:
            assert execution.kind == "evaluated"
            assert execution.outcome == "passed"
        assert "execution" not in snapshot.model_dump()
        assert "execution_readiness" not in snapshot.model.value.model_dump()
        assert snapshot.can_simulate == (revision == 2)
        snapshots.append(snapshot)
    assert "compiled_ssm" not in ARTIFACT_IDS
    for snapshot in snapshots:
        assert ModelReader(workspace, at=snapshot.commit_id).snapshot() == snapshot


def test_incomplete_model_is_rejected_before_inference(workspace, monkeypatch):
    from nof1_causal_lab.actions import fit as flow

    local = Mock()
    monkeypatch.setattr(flow, "fit", local)
    store = ArtifactStore(workspace)
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )
    state = StudyState().with_artifacts([write_question(store), model, panel])
    with pytest.raises(IncompleteModelError):
        run_async(
            run_action(
                workspace,
                FitRequest(model_revision=model.revision, panel_revision=panel.revision),
                state,
            )
        )
    local.assert_not_called()
    assert artifact_revisions(store, "model") == [model.revision]


def test_refit_after_an_edit_uses_selected_model_and_preserves_the_edit(workspace, monkeypatch):
    from nof1_causal_lab.actions import fit as flow
    from nof1_causal_lab.actions.runners import _run_fit
    from nof1_causal_lab.study.store import read_model

    prior = x_y_model()
    fitted = prior
    edited = fitted.revised(
        edges=(fitted.edges[0].revised(description="X changes Y within a day"),)
    )
    store = ArtifactStore(workspace)
    for revision, (model, producer) in enumerate(
        ((prior, "edit_model"), (fitted, "fit"), (edited, None)), 1
    ):
        store.write_artifact(
            "model",
            derived_from={"model": artifact_revision(workspace, "model", revision - 1)}
            if revision > 1
            else {},
            produced_by=producer,
            json_files={"model.json": model.model_dump(mode="json")},
        )
    store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": metadata_for_model(edited).model_dump(mode="json")},
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )

    from nof1_causal_lab.artifacts.posterior import InferenceEvidence
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from tests.inference_fixtures import compile_model_fixture, parameter_draws, particle_posterior
    import jax.numpy as jnp

    def fit(**kwargs):
        model = kwargs["selection"].model
        assert model == edited
        conditioned, identity = condition_model(model, compile_model_fixture(model),
            particle_posterior(JointPosteriorDraws(parameter_draws(model, 4), jnp.zeros((4, 2, 2)))),
            times=jnp.array([0.0, 1.0]), array_writer=store.write_array, array_loader=store.read_array)
        return {"_model": conditioned, "evidence": InferenceEvidence(distribution=identity,
            time_origin=None, duration_seconds=0, num_chains=1)}

    monkeypatch.setattr(flow, "fit", fit)
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision(workspace, "model", 3),
        "panel": artifact_revision(workspace, "panel", 1),
        "question": write_question(store).revision,
    }
    effects = run_async(_run_fit(store, pins, FitSettingsSpec()))
    assert effects.effects.produced[0].derived_from == pins
    assert {"model": effects.result.model.revision, "panel": effects.result.panel.revision} == {
        key: pins[key] for key in ("model", "panel")
    }
    assert read_model(store, artifact_revision(workspace, "model", 4)).edges == edited.edges
