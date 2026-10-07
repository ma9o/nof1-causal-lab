"""Whole scientific values commit atomically and findings retain exact source revisions."""

import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.state import ArtifactRecord, StudyState, is_stale
from tests.action_fixtures import edit_and_check, question_root
from tests.git_fixtures import artifact_revision, artifact_revisions, git_oid
from tests.helpers import fixture_entity_id, make_model
from tests.model_fixtures import x_y_model

pytestmark = pytest.mark.contract


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    return "additive-revisions"


def test_full_model_write_keeps_the_named_base_without_a_head_gate(workspace):
    question = QuestionSpec(text="Does X change Y?", outcome=fixture_entity_id("construct", "Y"))
    root = StudyRepository(workspace).state(question_root(workspace, question).commit_id)
    effects = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": root.current["question"].revision,
                    "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                }
            }
        ),
        root,
    )
    state = root.with_artifacts(effects.effects.produced)
    assert state.current["model"].revision == effects.effects.produced[0].revision
    assert set(state.current) == {"question", "model"}
    from nof1_causal_lab.study.store import ArtifactStore

    assert artifact_revisions(ArtifactStore(workspace), "model") == [
        state.current["model"].revision
    ]
    second = edit_and_check(
        workspace,
        EditModelRequest[GitOid].model_validate(
            {
                "input": {
                    "parent_ref": artifact_revision(workspace, "model", 1),
                    "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                }
            }
        ),
        state,
    )
    from nof1_causal_lab.actions.model_checks import read_model_checks

    checks, _, _ = read_model_checks(
        workspace, state.with_artifacts(second.effects.produced), action="edit_model"
    )
    assert checks.question is not None
    assert checks.question.question_revision == root.current["question"].revision
    assert second.effects.produced[0].revision != effects.effects.produced[0].revision
    assert second.effects.produced[0].derived_from == {
        "question": root.current["question"].revision,
        "model": state.current["model"].revision,
    }


def test_model_input_identity_preserves_findings_and_original_pins():
    model = ArtifactRecord(artifact_id="model", revision=git_oid(2))
    panel = ArtifactRecord(artifact_id="panel", revision=git_oid(1))
    state = StudyState().with_artifacts([model, panel])
    assert not is_stale(state, "panel")
    changed = model.revised(revision=git_oid(3))
    assert not is_stale(state.with_artifacts([changed]), "panel")


def test_statistical_enrichment_preserves_structural_and_measurement_inputs():
    from tests.helpers import make_model

    measured = make_model(["X", "Y"], [("X", "Y")])
    specified = x_y_model()
    before, after = input_fingerprints(measured), input_fingerprints(specified)
    for purpose in ("observations", "identification"):
        assert before[purpose] == after[purpose]
    assert before["compilation"] != after["compilation"]

    changed = input_fingerprints(
        measured.revised(
            edges=(measured.edges[0].revised(description="Revised causal assumption"),)
        )
    )
    assert changed["observations"] == before["observations"]
    for purpose in ("identification", "compilation"):
        assert changed[purpose] != before[purpose]

    reasoned = input_fingerprints(
        specified.revised(
            parameters=tuple(
                parameter.revised(reasoning="A literature range for this quantity.")
                for parameter in specified.parameters
            )
        )
    )
    for purpose in ("observations", "identification", "compilation"):
        assert reasoned[purpose] == after[purpose]
    assert reasoned["belief"] != after["belief"]
