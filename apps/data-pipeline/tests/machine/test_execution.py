"""Result installation, input versions and artifact freshness."""

import pytest

from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState
from nof1_causal_lab.machine.execution import (
    RetractedArtifact,
    apply_transition,
    freshness_report,
    input_pins,
    is_stale,
    run_retractions,
)
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.inference import inference_is_current
from nof1_causal_lab.machine.model_dependencies import MODEL_INPUTS
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
    return EpisodeState().with_artifacts(list(infos))


class TestApplyTransition:
    def test_produced_versions_become_current(self):
        state = apply_transition(EpisodeState(), [_version("raw_data")])
        assert state.has("raw_data")
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.revision == git_oid(1)

    def test_rerun_supersedes_version(self):
        state = _state(_version("raw_data", revision=git_oid(1)))
        state = apply_transition(state, [_version("raw_data", revision=git_oid(2))])
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.revision == git_oid(2)

    def test_optional_artifact_retracted_when_withheld(self):
        spec = transition_spec("measurements")
        state = _state(
            _version("panel", revision=git_oid(1)),
        )
        produced = []
        retracted = run_retractions(state, spec, produced)
        assert retracted == [
            RetractedArtifact(
                artifact_id="panel",
                reason_ref="measurements.produces_optional.panel",
            )
        ]
        next_state = apply_transition(state, produced, retracted)
        assert not next_state.has("panel")

    def test_no_retraction_when_optional_still_produced(self):
        spec = transition_spec("measurements")
        state = _state(_version("panel", revision=git_oid(1)))
        produced = [
            _version("panel", revision=git_oid(2)),
        ]
        assert run_retractions(state, spec, produced) == []


class TestStaleness:
    def _fitted_chain(self):
        inputs = dict.fromkeys(MODEL_INPUTS.values(), "same-scientific-input")
        model = _version(
            "model",
            revision=git_oid(2),
            derived_from={"model": git_oid(1), "panel": git_oid(1)},
            produced_by="run:posterior",
        ).model_copy(update={"model_inputs": inputs})
        dependents = [
            _version("identification_report", derived_from={"model": git_oid(1)}),
        ]
        return _state(
            model,
            _version("panel"),
            *[
                item.model_copy(
                    update={
                        "consumed_model_inputs": {
                            MODEL_INPUTS[item.artifact_id]: inputs[MODEL_INPUTS[item.artifact_id]]
                        }
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
        state = apply_transition(self._fitted_chain(), [_version("model", revision=git_oid(3))])
        assert not inference_is_current(state)
        assert not is_stale(state, "panel")
        assert not is_stale(state, "model")

    def test_retracted_input_invalidates_fit(self):
        state = self._fitted_chain().without(["panel"])
        assert not inference_is_current(state)

    def test_republishing_identification_preserves_conditioned_science(self):
        state = self._fitted_chain()
        current = state.with_artifacts(
            [state.current["identification_report"].model_copy(update={"revision": git_oid(2)})]
        )
        assert inference_is_current(current)

    def test_absent_artifact_is_not_stale(self):
        assert not is_stale(EpisodeState(), "identification_report")


class TestInputPins:
    def test_pins_current_versions(self):
        state = _state(
            _version("model", revision=git_oid(3)),
            _version("panel", revision=git_oid(2)),
        )
        pins = input_pins(state, transition_spec("posterior"))
        assert pins == {"model": git_oid(3), "panel": git_oid(2)}


def test_freshness_report_shape():
    state = _state(_version("model"))
    report = freshness_report(state)
    by_id = {status.artifact_id: status for status in report}
    assert by_id["model"].exists
    assert "provenance" not in by_id["model"].model_dump()
    assert not by_id["panel"].exists
    assert not by_id["panel"].stale
