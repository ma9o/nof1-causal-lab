"""Checks follow scientific input changes and never become authoring gates."""

from datetime import UTC, datetime

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import TransitionEffects, apply_transition
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from tests.action_fixtures import edit_and_check
from tests.integration.transition_runner_fixtures import (
    panel_frame,
    panel_metadata,
    scientific_model,
)

pytestmark = pytest.mark.inference(concern="predictive")


@pytest.fixture
def study(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    return ArtifactStore("checks"), StudyRepository("checks")


def _publish(study, effects, action="edit_model"):
    store, repository = study
    messages = completion_messages(
        store.workspace_id,
        action,
        effects.produced,
        effects.diagnostics,
        datetime.now(UTC),
        checks=effects.checks,
    )
    repository.append(
        TransitionRecord(
            seq=len(repository.attempts()) + 1,
            ts=datetime.now(UTC).isoformat(),
            status="applied",
            action=action,
            inputs={},
            produced=effects.produced,
            retracted=effects.retracted,
            checks=effects.checks,
            messages=messages,
            trace_ids=[],
            resume=None,
        )
    )
    return repository.state(repository.head()), messages


def _prepare(study, state, *, n_days=4):
    store, _ = study
    panel = store.write_artifact(
        "panel",
        produced_by="run:measurements",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=n_days)},
    )
    return evaluate_data_checks(
        store.workspace_id,
        state,
        TransitionEffects(produced=[panel]),
    )


@pytest.mark.inference(concern="simulation")
def test_automatic_exact_batch_reuse_and_input_invalidation(study, monkeypatch):
    from nof1_causal_lab.actions import predictive_checks
    from nof1_causal_lab.models.ssm.predictive import simulation

    # Exercise the production generator once; subsequent calls verify selection.
    monkeypatch.setattr(predictive_checks, "PREDICTIVE_DRAWS", 4)
    generate = simulation.generate_simulation_batch
    calls = []
    batches = []

    def counted(model, design, **kwargs):
        calls.append((model, design))
        if not batches:
            batches.append(generate(model, design, **kwargs))
        return batches[0]

    monkeypatch.setattr(simulation, "generate_simulation_batch", counted)
    store, repository = study
    model = scientific_model()
    initial = edit_and_check(
        store.workspace_id,
        EditModelRequest(expected_revision=None, model=model),
        EpisodeState(),
    )
    assert initial.checks.predictive.reason == "NO_COMPATIBLE_PANEL"
    state, _ = _publish(study, initial)
    prepared = _prepare(study, state)
    state, _ = _publish(study, prepared, "prepare_data")
    assert not calls
    checked = edit_and_check(
        store.workspace_id,
        EditModelRequest(expected_revision=state.current["model"].revision, model=model),
        state,
    )
    state, _ = _publish(study, checked)
    report = state.checks.predictive
    assert len(calls) == 1
    assert report.draws == 4
    assert report.design.interventions == ()
    assert report.law.interpretation == "prior_predictive"
    assert any(f.check == "C1a finiteness" and f.passed for f in report.findings)
    historical = repository.head()

    def edit(candidate):
        return edit_and_check(
            store.workspace_id,
            EditModelRequest(
                expected_revision=state.current["model"].revision,
                model=candidate,
            ),
            state,
        )

    unchanged = edit(model)
    assert set(unchanged.checks.reused) == {
        "specification",
        "identification",
        "compatibility",
        "predictive",
    }
    state, messages = _publish(study, unchanged)
    assert "PREDICTIVE_CHECKS_REUSED" in {m.label for m in messages}
    assert len(calls) == 1
    assert state.checks.predictive.model_revision == report.model_revision
    assert state.current["model"].revision != report.model_revision

    law_id = next(iter(model.distributions))
    laws = dict(model.distributions)
    import numpyro.distributions as dist

    laws[law_id] = dist.Beta(3.0, 2.0)
    changed = model.revised(distributions=laws)
    revised = edit(changed)
    assert set(revised.checks.reused) == {"identification"}
    state, _ = _publish(study, revised)
    assert len(calls) == 2

    monkeypatch.setattr(predictive_checks, "PREDICTIVE_POLICY_VERSION", "changed-test-policy")
    policy_edit = edit(changed)
    state, _ = _publish(study, policy_edit)
    assert len(calls) == 3

    changed = changed.revised(question="Does the changed measurement question fit this study?")
    same_data = edit(changed)
    assert "predictive" in same_data.checks.reused
    state, _ = _publish(study, same_data)
    assert len(calls) == 3
    refreshed = _prepare(study, state)
    assert refreshed.checks is None
    assert len(calls) == 3
    assert repository.state(historical).checks.predictive == report


@pytest.mark.parametrize("failure", ["paths", "emission_mean"])
def test_nonfinite_findings_save_but_generator_errors_do_not_publish(study, monkeypatch, failure):
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory

    def nonfinite(model, samples, times, **_kwargs):
        if failure == "emission_mean":
            from nof1_causal_lab.models.predictive_simulation import (
                PredictiveObservationMeanOverflow,
            )

            raise PredictiveObservationMeanOverflow(
                bad_manifest_names=("stress_score",),
                manifest_indices=(0,),
                failing_draw_indices=(0,),
                n_draws=2,
                first_bad_time_index=1,
                max_linear_predictor=1000.0,
                overflow_threshold=700.0,
            )
        paths = jnp.full((2, len(times), len(model.state_order)), np.nan)
        emissions = jnp.full((2, len(times), len(model.manifest_indicator_order)), np.nan)
        return PredictiveDraws(
            {name: value[:2] for name, value in samples.items()},
            {},
            PredictiveTrajectory(
                paths, emissions, emissions, jnp.ones_like(emissions, dtype=bool), emissions
            ),
        )

    from nof1_causal_lab.models.ssm.predictive import simulation

    monkeypatch.setattr(simulation, "simulate_predictive_draws", nonfinite)
    store, repository = study
    initial = edit_and_check(
        store.workspace_id,
        EditModelRequest(expected_revision=None, model=scientific_model()),
        EpisodeState(),
    )
    state, _ = _publish(study, initial)
    prepared = _prepare(study, state)
    state, _ = _publish(study, prepared, "prepare_data")
    checked = edit_and_check(
        store.workspace_id,
        EditModelRequest(
            expected_revision=state.current["model"].revision, model=scientific_model()
        ),
        state,
    )
    state, labels = _publish(study, checked)
    report = state.checks.predictive
    assert report.status == "failed"
    assert any(f.passed is False and f.check == "C1a finiteness" for f in report.findings)
    assert any(
        f.passed is None and f.reason in {"NONFINITE_PATHS", "NONFINITE_EMISSION_MEAN"}
        for f in report.findings
    )
    assert "PREDICTIVE_CHECK_FAILED" in {m.label for m in labels}
    assert all(set(m.model_dump()) == {"timestamp", "level", "label"} for m in labels)
    head = repository.head()

    def broken(*_args, **_kwargs):
        raise RuntimeError("unexpected generator defect")

    monkeypatch.setattr(simulation, "simulate_predictive_draws", broken)
    prepared = _prepare(study, state, n_days=5)
    selected = apply_transition(state, prepared.produced, prepared.retracted)
    with pytest.raises(RuntimeError, match="unexpected generator defect"):
        edit_and_check(
            store.workspace_id,
            EditModelRequest(
                expected_revision=state.current["model"].revision, model=scientific_model()
            ),
            selected,
        )
    assert repository.head() == head
    assert repository.state(repository.head()).checks.predictive == report


def test_joint_laws_can_be_checked_when_refitting_is_unsupported(study, monkeypatch):
    import jax

    from nof1_causal_lab.artifacts.checks import PredictiveCheckFinding
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
        ParticleMCMCPosterior,
    )
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
    from tests.model_fixtures import parameter_draws

    store, _ = study
    model = scientific_model()
    initial = edit_and_check(
        store.workspace_id, EditModelRequest(expected_revision=None, model=model), EpisodeState()
    )
    state, _ = _publish(study, initial)
    panel = store.write_artifact(
        "panel",
        produced_by="run:measurements",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=4)},
    )
    conditioned = condition_model(
        model,
        ParticleMCMCPosterior(
            JointPosteriorDraws(
                parameter_draws(model, 3),
                jnp.zeros((3, 5, 2)),
            )
        ),
        times=jnp.arange(-1.0, 4.0),
    )
    fitted = store.write_artifact(
        "model",
        produced_by="run:posterior",
        derived_from={"model": state.current["model"].revision, "panel": panel.revision},
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    data_effects = evaluate_data_checks(
        store.workspace_id, state, TransitionEffects(produced=[panel])
    )
    state = apply_transition(state, [*data_effects.produced, fitted], [])
    calls = []

    def sample(model, design, **kwargs):
        draws = sample_model_laws(
            model, draws=kwargs["draws"], key=jax.random.PRNGKey(kwargs["seed"])
        )
        calls.append(draws)
        return

    monkeypatch.setattr(simulation, "generate_simulation_batch", sample)
    monkeypatch.setattr(
        simulation,
        "measure_simulation_batch",
        lambda *_a, **_k: (
            (
                PredictiveCheckFinding(
                    check="C1a finiteness",
                    target="all states",
                    value="0%",
                    band="0%",
                    passed=True,
                    note="Finite.",
                ),
            ),
            None,
        ),
    )
    checked = edit_and_check(
        store.workspace_id,
        EditModelRequest(
            expected_revision=fitted.revision,
            model=conditioned,
        ),
        state,
    )
    assert len(calls) == 1
    assert any(
        f.check == "fit_laws" and f.status == "failed"
        for f in checked.checks.specification.findings
    )
    assert checked.checks.predictive.status == "passed"
    assert checked.checks.predictive.law.kind == "fitted"
    assert checked.checks.predictive.law.interpretation == "in_sample_posterior_predictive"
    state = apply_transition(state, checked.produced, checked.retracted, checked.checks)
    # Preparing the same observations creates a new revision, not a held-out study.
    prepared = _prepare(study, state)
    assert prepared.checks is None
    state = apply_transition(state, prepared.produced, prepared.retracted)
    checked = edit_and_check(
        store.workspace_id,
        EditModelRequest(expected_revision=state.current["model"].revision, model=conditioned),
        state,
    )
    assert checked.checks.predictive.law.fitted_panel_revision == panel.revision
    assert checked.checks.predictive.law.interpretation == "posterior_predictive"
