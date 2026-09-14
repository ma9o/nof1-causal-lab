"""legal_moves / apply_transition / staleness / freshness semantics."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.graph import WRITABLE_ARTIFACTS, transition_spec
from nof1_causal_lab.machine.inference import inference_is_current
from nof1_causal_lab.machine.model_dependencies import MODEL_INPUTS
from nof1_causal_lab.machine.moves import (
    RetractedArtifact,
    RunOperation,
    WriteArtifact,
    apply_transition,
    freshness_report,
    input_pins,
    is_stale,
    legal_moves,
    run_retractions,
    validate_move,
)


def _version(artifact_id, version=1, derived_from=None, produced_by=None, provenance="computed"):
    return ArtifactVersionInfo(
        artifact_id=artifact_id,
        version=version,
        provenance=provenance,
        derived_from=derived_from or {},
        produced_by=produced_by,
        created_at="2026-07-03T00:00:00Z",
    )


def _state(*infos):
    return EpisodeState().with_versions(list(infos))


def _runnable(state):
    return {move.operation_id for move in legal_moves(state) if isinstance(move, RunOperation)}


class TestLegalMoves:
    def test_empty_state_enables_only_raw_data(self):
        assert _runnable(EpisodeState()) == {"raw_data"}

    def test_question_write_enables_latent_structure(self):
        state = _state(_version("question", provenance="human"))
        assert _runnable(state) == {"raw_data", "latent_structure"}

    def test_declared_writes_are_offered(self):
        offered = {
            move.artifact_id
            for move in legal_moves(EpisodeState())
            if isinstance(move, WriteArtifact)
        }
        assert offered == set(WRITABLE_ARTIFACTS)
        assert "question" in offered
        assert "model" in offered
        assert "causal_design" not in offered
        assert "raw_data" not in offered

    def test_no_identification_report_disables_fit_chain(self):
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data"),
            _version("model"),
            _version("panel"),
            _version("validation_report"),
        )
        runnable = _runnable(state)
        assert "statistical_model_spec" not in runnable
        assert "measurements" in runnable

    def test_identification_report_enables_statistical_model_spec(self):
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data"),
            _version("model"),
            _version("identification_report"),
            _version("panel"),
            _version("validation_report"),
        )
        assert "statistical_model_spec" in _runnable(state)

    def test_baseline_report_does_not_require_question(self):
        state = _state(
            _version("model"),
            _version("identification_report"),
            _version("panel"),
        )
        assert "baseline_report" in _runnable(state)


class TestValidateMove:
    def test_missing_inputs_rejected_with_names(self):
        reason = validate_move(EpisodeState(), RunOperation(operation_id="measurement_structure"))
        assert reason is not None
        assert "question" in reason
        assert "model" in reason

    def test_unknown_transition_rejected(self):
        with pytest.raises(ValidationError):
            RunOperation.model_validate({"operation_id": "validation_report"})

    def test_computed_provenance_write_rejected(self):
        reason = validate_move(
            EpisodeState(),
            WriteArtifact(artifact_id="question", provenance="computed"),
        )
        assert reason is not None

    def test_derived_artifact_write_rejected(self):
        reason = validate_move(EpisodeState(), WriteArtifact(artifact_id="identification_report"))
        assert reason is not None
        assert "not writable" in reason

    def test_legal_run_accepted(self):
        assert validate_move(EpisodeState(), RunOperation(operation_id="raw_data")) is None


class TestApplyTransition:
    def test_produced_versions_become_current(self):
        state = apply_transition(EpisodeState(), [_version("raw_data")])
        assert state.has("raw_data")
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.version == 1

    def test_rerun_supersedes_version(self):
        state = _state(_version("raw_data", version=1))
        state = apply_transition(state, [_version("raw_data", version=2)])
        raw_data = state.get("raw_data")
        assert raw_data is not None
        assert raw_data.version == 2

    def test_optional_artifact_retracted_when_withheld(self):
        spec = transition_spec("measurements")
        state = _state(
            _version("panel", version=1),
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
        state = _state(_version("panel", version=1))
        produced = [
            _version("panel", version=2),
        ]
        assert run_retractions(state, spec, produced) == []


class TestStaleness:
    def _fitted_chain(self):
        inputs = dict.fromkeys(MODEL_INPUTS.values(), "same-scientific-input")
        model = _version(
            "model",
            version=2,
            derived_from={"model": 1, "panel": 1},
            produced_by="run:posterior",
        ).model_copy(update={"model_inputs": inputs})
        dependents = [
            _version("identification_report", derived_from={"model": 1}),
            _version("panel", derived_from={"model": 1}),
            _version(
                "baseline_report", derived_from={"model": 2, "panel": 1, "identification_report": 1}
            ),
        ]
        return _state(
            model,
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
        assert not is_stale(state, "baseline_report")

    def test_editing_model_stales_produced_descendants(self):
        state = apply_transition(
            self._fitted_chain(), [_version("model", version=3, provenance="human")]
        )
        assert not inference_is_current(state)
        assert is_stale(state, "panel")
        assert is_stale(state, "baseline_report")
        assert not is_stale(state, "model")

    def test_retracted_input_invalidates_fit_and_report(self):
        state = self._fitted_chain().without(["panel"])
        assert not inference_is_current(state)
        assert is_stale(state, "baseline_report")

    def test_republishing_identification_preserves_conditioned_science(self):
        state = self._fitted_chain()
        current = state.with_versions(
            [state.current["identification_report"].model_copy(update={"version": 2})]
        )
        assert inference_is_current(current)
        assert is_stale(current, "baseline_report")

    def test_absent_artifact_is_not_stale(self):
        assert not is_stale(EpisodeState(), "baseline_report")


class TestInputPins:
    def test_pins_current_versions(self):
        state = _state(
            _version("question", version=3, provenance="human"),
        )
        pins = input_pins(state, transition_spec("latent_structure"))
        assert pins == {"question": 3}


def test_freshness_report_shape():
    state = _state(_version("question", provenance="human"))
    report = freshness_report(state)
    by_id = {status.artifact_id: status for status in report}
    assert by_id["question"].exists
    assert by_id["question"].provenance == "human"
    assert not by_id["model"].exists
    assert not by_id["model"].stale
