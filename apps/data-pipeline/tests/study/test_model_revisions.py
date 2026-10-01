"""Whole scientific values commit atomically and findings retain exact source revisions."""

from pathlib import Path

import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.study.errors import ArtifactWriteRejected
from nof1_causal_lab.study.state import ArtifactRecord, StudyState, is_stale
from tests.action_fixtures import edit_and_check
from tests.git_fixtures import artifact_revision, git_oid
from tests.helpers import make_model

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "additive-revisions"


def test_full_model_write_checks_base_before_writing(workspace):
    effects = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {
                "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                "expected_revision": None,
            }
        ),
        StudyState(),
    )
    state = StudyState(
        current={info.artifact_id: info for info in effects.produced}, checks=effects.checks
    )
    assert state.current["model"].revision == effects.produced[0].revision
    assert state.has("identification_report")
    with pytest.raises(ArtifactWriteRejected, match="conflict"):
        edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {
                    "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                    "expected_revision": None,
                }
            ),
            state,
        )
    from nof1_causal_lab.study.store import ArtifactStore

    assert ArtifactStore(workspace).list_revisions("model") == [state.current["model"].revision]
    second = edit_and_check(
        workspace,
        EditModelRequest.model_validate(
            {
                "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                "expected_revision": artifact_revision(workspace, "model", 1),
            }
        ),
        state,
    )
    assert second.checks is not None
    assert "identification" in second.checks.reused
    assert second.produced[0].revision != effects.produced[0].revision
    assert state.current["identification_report"].derived_from == {
        "model": artifact_revision(workspace, "model", 1)
    }


def test_model_input_identity_preserves_findings_and_original_pins():
    values = input_fingerprints(make_model(["X", "Y"], [("X", "Y")]))
    model = ArtifactRecord(artifact_id="model", revision=git_oid(2), model_inputs=values)
    panel = ArtifactRecord(artifact_id="panel", revision=git_oid(1))
    state = StudyState().with_artifacts([model, panel])
    assert not is_stale(state, "panel")
    changed = type(model).model_validate(
        {**model.model_dump(), "revision": git_oid(3), "model_inputs": {}}
    )
    assert not is_stale(state.with_artifacts([changed]), "panel")


def test_statistical_enrichment_preserves_structural_and_measurement_inputs():
    from tests.helpers import make_model

    measured = make_model(["X", "Y"], [("X", "Y")])
    specified = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )
    before, after = input_fingerprints(measured), input_fingerprints(specified)
    for purpose in ("observations", "identification"):
        assert before[purpose] == after[purpose]
    assert before["compilation"] != after["compilation"]

    changed = input_fingerprints(
        measured.revised(
            edges=(
                type(measured.edges[0]).model_validate(
                    {**measured.edges[0].model_dump(), "description": "Revised causal assumption"}
                ),
            )
        )
    )
    assert changed["observations"] == before["observations"]
    for purpose in ("identification", "compilation"):
        assert changed[purpose] != before[purpose]

    revised_question = input_fingerprints(measured.revised(question="Does X change Y?"))
    assert revised_question["observations"] == before["observations"]
    for purpose in ("identification", "compilation", "belief"):
        assert revised_question[purpose] == before[purpose]
