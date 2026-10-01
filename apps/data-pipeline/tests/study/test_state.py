"""Result installation, input versions and artifact freshness."""

import pytest

from nof1_causal_lab.study.lineage import inference_is_current
from nof1_causal_lab.study.model_dependencies import MODEL_INPUTS
from nof1_causal_lab.study.state import (
    ArtifactRecord,
    RetractedArtifact,
    StudyState,
    apply_effects,
    freshness_report,
    is_stale,
)
from tests.git_fixtures import git_oid

pytestmark = pytest.mark.contract

_FIRST_REVISION = git_oid(1)


def _version(
    artifact_id,
    revision=_FIRST_REVISION,
    derived_from=None,
    produced_by=None,
):
    return ArtifactRecord(
        artifact_id=artifact_id,
        revision=revision,
        derived_from=derived_from or {},
        produced_by=produced_by,
        created_at="2026-07-03T00:00:00Z",
    )


def _state(*infos):
    return StudyState().with_artifacts(list(infos))


class TestApplyTransition:
    def test_produced_versions_become_current(self):
        state = apply_effects(StudyState(), [_version("raw_data")])
        assert state.has("raw_data")
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.revision == git_oid(1)

    def test_rerun_supersedes_version(self):
        state = _state(_version("raw_data", revision=git_oid(1)))
        state = apply_effects(state, [_version("raw_data", revision=git_oid(2))])
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.revision == git_oid(2)

    def test_retracted_artifact_leaves_the_state(self):
        state = _state(_version("validation_report", revision=git_oid(1)))
        retracted = [RetractedArtifact(artifact_id="validation_report", reason_ref="panel.changed")]
        assert not apply_effects(state, [], retracted).has("validation_report")


class TestStaleness:
    def _fitted_chain(self):
        inputs = dict.fromkeys(MODEL_INPUTS.values(), "same-scientific-input")
        model = _version(
            "model",
            revision=git_oid(2),
            derived_from={"model": git_oid(1), "panel": git_oid(1)},
            produced_by="fit",
        )
        model = ArtifactRecord.model_validate({**model.model_dump(), "model_inputs": inputs})
        dependents = [
            _version("identification_report", derived_from={"model": git_oid(1)}),
        ]
        return _state(
            model,
            _version("panel"),
            *[
                type(item).model_validate(
                    {
                        **item.model_dump(),
                        "consumed_model_inputs": {
                            MODEL_INPUTS[item.artifact_id]: inputs[MODEL_INPUTS[item.artifact_id]]
                        },
                    }
                )
                for item in dependents
            ],
        )

    def test_fresh_chain_is_not_stale(self):
        state = self._fitted_chain()
        assert inference_is_current(state)
        assert not is_stale(state, "identification_report")

    def test_editing_model_stales_produced_descendants(self):
        state = apply_effects(self._fitted_chain(), [_version("model", revision=git_oid(3))])
        assert not inference_is_current(state)
        assert not is_stale(state, "panel")
        assert not is_stale(state, "model")

    def test_retracted_input_invalidates_fit(self):
        state = self._fitted_chain().without(["panel"])
        assert not inference_is_current(state)

    def test_republishing_identification_preserves_conditioned_science(self):
        state = self._fitted_chain()
        current = state.with_artifacts(
            [
                type(state.current["identification_report"]).model_validate(
                    {**state.current["identification_report"].model_dump(), "revision": git_oid(2)}
                )
            ]
        )
        assert inference_is_current(current)

    def test_absent_artifact_is_not_stale(self):
        assert not is_stale(StudyState(), "identification_report")


def test_freshness_report_shape():
    state = _state(_version("model"))
    report = freshness_report(state)
    by_id = {status.artifact_id: status for status in report}
    assert by_id["model"].exists
    assert "provenance" not in by_id["model"].model_dump()
    assert not by_id["panel"].exists
    assert not by_id["panel"].stale
