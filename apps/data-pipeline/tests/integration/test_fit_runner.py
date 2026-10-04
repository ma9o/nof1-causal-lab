"""Persistence and refit lifecycle with mocked inference, never numerical fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from nof1_causal_lab.actions.contracts import FitRequest
from nof1_causal_lab.actions.runners import run_action_locally
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import read_model
from tests.git_fixtures import artifact_revisions
from tests.helpers import run_async, write_question
from tests.inference_fixtures import (
    bind_panel_fixture,
    compile_model_fixture,
    model_draws,
    parameter_draws,
    particle_posterior,
)
from tests.integration import runner_fixtures as fx
from tests.model_fixtures import stress_sleep_model

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.study.store import ArtifactStore


def test_inference_advances_model_and_uses_the_selected_input(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    import jax.numpy as jnp

    from nof1_causal_lab.actions.inference import fit as stage5_fit
    from nof1_causal_lab.models.ssm import numerics
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    original = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=original.revision)
    authored = stress_sleep_model()
    parameters = parameter_draws(authored, 4)
    states = tuple(numerics.state_ids(compile_model_fixture(authored)))
    paths = jnp.arange(4 * 2 * len(states), dtype=jnp.float32).reshape(4, 2, len(states))
    fitted_inputs = []

    def fake_fit_model(selection, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        model = selection.model
        fitted_inputs.append(model.model_dump(mode="json"))
        return {
            "fitted": True,
            "duration_seconds": 0.01,
            "result": particle_posterior(
                draws=JointPosteriorDraws(
                    parameters=parameters, latent_paths=paths, state_ids=states
                )
            ),
            "panel": bind_panel_fixture(
                compile_model_fixture(model),
                jnp.zeros((2, 2)),
                jnp.array([0.0, 1.0], dtype=jnp.float32),
            ),
        }

    monkeypatch.setattr(stage5_fit, "fit_model", fake_fit_model)
    state = fx.state_from(original, panel)
    revisions = [original.revision]
    # Refit the selected authored revision; conditioned joint laws cannot be fit inputs.
    request = FitRequest(model_revision=original.revision, panel_revision=panel.revision)
    question = write_question(artifact_store)
    pins: dict[ArtifactId, GitOid] = {
        "model": original.revision,
        "panel": panel.revision,
        "question": question.revision,
    }
    for _ in range(2):
        applied = run_async(run_action_locally(integration_workspace, request, pins))
        info = next(info for info in applied.effects.produced if info.artifact_id == "model")
        assert info.revision not in revisions
        revisions.append(info.revision)
        # Both provenance and computation follow the selected model revision.
        assert info.derived_from == pins
        assert {
            "model": applied.result.model.revision,
            "panel": applied.result.panel.revision,
            "question": question.revision,
        } == info.derived_from
        from nof1_causal_lab.actions.fit import read_inference_report

        report = read_inference_report(artifact_store, info.revision, applied.result.evidence)
        assert report.core.inference_diagnostics.num_chains == 1
        assert report.core.inference_diagnostics.num_samples == 4
        assert report.core.inference_metadata.n_samples == 4
        assert "report" not in applied.result.model_dump()
        assert (
            applied.result.evidence.distribution
            in read_model(artifact_store, info.revision).law_layouts
        )
        conditioned = read_model(artifact_store, info.revision)
        assert type(conditioned) is type(authored)
        assert conditioned.distributions
        assert "inference_diagnostics" not in conditioned.model_dump()
        retained = model_draws(compile_model_fixture(conditioned))
        np.testing.assert_array_equal(retained.latent_paths, paths)
        for name, values in parameters.items():
            np.testing.assert_array_equal(retained.parameters[name], values)
        state = apply_effects(state, applied.effects.produced, applied.effects.retracted)
        assert state.current["model"].produced_by == "fit"
    assert fitted_inputs == [
        read_model(artifact_store, original.revision).model_dump(mode="json") for _ in range(2)
    ]
    assert read_model(artifact_store, original.revision).model_dump(mode="json") == fitted_inputs[0]
    assert artifact_revisions(artifact_store, "model") == revisions
