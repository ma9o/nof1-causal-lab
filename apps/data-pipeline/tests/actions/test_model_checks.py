"""Checks follow scientific input changes and never become authoring gates."""

from datetime import UTC, datetime, timedelta

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.model_checks import evaluate_model_checks, read_model_checks
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, DataPreparationResult, ModelFitResult
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record, question_root
from tests.inference_fixtures import compile_model_fixture, particle_posterior
from tests.integration.runner_fixtures import panel_frame, panel_metadata
from tests.model_fixtures import stress_sleep_model

pytestmark = pytest.mark.inference(concern="predictive")


@pytest.fixture
def study(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    question_root("checks")
    return ArtifactStore("checks"), StudyRepository("checks")


def _root(study):
    _, repository = study
    return repository.state(repository.head())


def _edit(workspace, request, state):
    staged = edit_model(workspace, request, state)
    assert isinstance(staged, Applied)
    checks, identification, validation = evaluate_model_checks(
        workspace, state, staged, action="edit_model"
    )
    return staged, checks, identification, validation


def _publish(study, effects, action="edit_model", *, checks=None):
    store, repository = study
    selected = apply_effects(
        repository.state(repository.head()), effects.effects.produced, effects.effects.retracted
    )
    if action == "prepare_data":
        from nof1_causal_lab.actions.data_checks import read_data_profile

        reports = (read_data_profile(store, selected.current["panel"].revision),)
    else:
        current, identification, validation = read_model_checks(
            store.workspace_id, selected, action="edit_model"
        )
        checks = current if checks is None else checks
        reports = (identification, validation) if validation is not None else (identification,)
    messages = completion_messages(effects, datetime.now(UTC), reports, checks=checks)
    repository.append(
        applied_record(
            effects,
            seq=len(repository.attempts()) + 1,
            ts=datetime.now(UTC).isoformat(),
            messages=messages,
        )
    )
    return repository.state(repository.head()), messages


def _prepare(study, state, *, n_days=4):
    store, _ = study
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=n_days)},
    )
    applied = Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel]))
    evaluate_data_checks(store.workspace_id, state, applied)
    return applied


@pytest.mark.inference(concern="simulation")
def test_automatic_exact_batch_reuse_and_input_invalidation(study, monkeypatch):
    from nof1_causal_lab.actions import model_checks, predictive_checks
    from nof1_causal_lab.models.ssm.predictive import simulation

    # Exercise the production generator once; subsequent calls verify selection.
    monkeypatch.setattr(predictive_checks, "PREDICTIVE_DRAWS", 4)
    generate = simulation.generate_simulation_batch
    calls = []
    batches = []

    def counted(model, **kwargs):
        calls.append((model, kwargs["start"], kwargs["end"]))
        if not batches:
            batches.append(generate(model, **kwargs))
        return batches[0]

    monkeypatch.setattr(simulation, "generate_simulation_batch", counted)
    compilation_calls = []
    compile_model = model_checks.compile_model
    compile_fit_inputs = model_checks.compile_fit_inputs

    def counted_model(selection):
        compilation_calls.append("model")
        return compile_model(selection)

    def counted_fit(compiled, selection):
        compilation_calls.append("fit")
        return compile_fit_inputs(compiled, selection)

    monkeypatch.setattr(model_checks, "compile_model", counted_model)
    monkeypatch.setattr(model_checks, "compile_fit_inputs", counted_fit)
    store, repository = study
    model = stress_sleep_model()
    initial, initial_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(expected_revision=None, model=model),
        _root(study),
    )
    assert initial_checks.predictive.evaluation.reason == "NO_COMPATIBLE_PANEL"
    state, _ = _publish(study, initial)
    prepared = _prepare(study, state)
    state, _ = _publish(study, prepared, "prepare_data")
    assert not calls
    compilation_calls.clear()
    checked, checked_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(expected_revision=state.current["model"].revision, model=model),
        state,
    )
    assert compilation_calls == ["model", "fit"]
    state, _ = _publish(study, checked, checks=checked_checks)
    report = read_model_checks(store.workspace_id, state, action="edit_model")[0].predictive
    assert len(calls) == 1
    assert report.draws == 4
    assert report.law.interpretation == "prior_predictive"
    assert any(
        f.subject.check == "C1a finiteness" and f.kind == "evaluated" and f.outcome == "passed"
        for f in report.evaluation.findings
    )
    historical = repository.head()

    def edit(candidate):
        return _edit(
            store.workspace_id,
            EditModelRequest(
                expected_revision=state.current["model"].revision,
                model=candidate,
            ),
            state,
        )

    compilation_calls.clear()
    unchanged, unchanged_checks, _, _ = edit(model)
    assert not compilation_calls
    assert set(unchanged_checks.reused) == {
        "specification",
        "identification",
        "question",
        "compatibility",
        "predictive",
    }
    state, messages = _publish(study, unchanged, checks=unchanged_checks)
    assert "PREDICTIVE_CHECKS_REUSED" in {m.label for m in messages}
    assert len(calls) == 1
    assert (
        read_model_checks(store.workspace_id, state, action="edit_model")[
            0
        ].predictive.model_revision
        == state.current["model"].revision
    )
    assert state.current["model"].revision != report.model_revision

    law_id = next(iter(model.distributions))
    laws = dict(model.distributions)
    import numpyro.distributions as dist

    laws[law_id] = dist.Beta(3.0, 2.0)
    changed = model.revised(distributions=laws)
    revised, revised_checks, _, _ = edit(changed)
    assert set(revised_checks.reused) == {"identification", "question"}
    state, _ = _publish(study, revised)
    assert len(calls) == 2

    monkeypatch.setattr("nof1_causal_lab.study.store._CODE_DIGEST", "changed-test-policy")
    compilation_calls.clear()
    policy_edit, policy_edit_checks, _, _ = edit(changed)
    assert compilation_calls == ["model", "fit"]
    state, _ = _publish(study, policy_edit)
    assert len(calls) == 3

    refreshed = _prepare(study, state)
    assert {info.artifact_id for info in refreshed.effects.produced} == {"panel"}
    assert len(calls) == 3
    assert (
        read_model_checks(store.workspace_id, repository.state(historical), action="edit_model")[
            0
        ].predictive
        == report
    )


@pytest.mark.parametrize(
    (
        "failure",
        "scientific_model_payload",
        "scientific_model_2_payload",
        "scientific_model_3_payload",
    ),
    [
        pytest.param(
            "paths",
            stress_sleep_model,
            stress_sleep_model,
            stress_sleep_model,
            id="paths",
        ),
        pytest.param(
            "emission_mean",
            stress_sleep_model,
            stress_sleep_model,
            stress_sleep_model,
            id="emission_mean",
        ),
    ],
)
def test_nonfinite_findings_save_but_generator_errors_do_not_publish(
    study,
    monkeypatch,
    failure,
    scientific_model_payload,
    scientific_model_2_payload,
    scientific_model_3_payload,
):
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
        paths = jnp.full((2, len(times), len(model.states)), np.nan)
        emissions = jnp.full(
            (2, len(times), len(model.observations)),
            np.nan,
        )
        return PredictiveDraws(
            {name: value[:2] for name, value in samples.items()},
            PredictiveTrajectory(
                paths, emissions, emissions, jnp.ones_like(emissions, dtype=bool), emissions
            ),
        )

    from nof1_causal_lab.models.ssm.predictive import simulation

    monkeypatch.setattr(simulation, "simulate_predictive_draws", nonfinite)
    store, repository = study
    initial, initial_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(
            expected_revision=None,
            model=scientific_model_payload(),
        ),
        _root(study),
    )
    state, _ = _publish(study, initial)
    prepared = _prepare(study, state)
    state, _ = _publish(study, prepared, "prepare_data")
    checked, checked_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(
            expected_revision=state.current["model"].revision,
            model=scientific_model_2_payload(),
        ),
        state,
    )
    state, labels = _publish(study, checked, checks=checked_checks)
    report = read_model_checks(store.workspace_id, state, action="edit_model")[0].predictive
    assert report.status == "failed"
    assert any(
        f.kind == "evaluated" and f.outcome == "failed" and f.subject.check == "C1a finiteness"
        for f in report.evaluation.findings
    )
    assert any(
        f.kind == "not_evaluated" and f.reason in {"NONFINITE_PATHS", "NONFINITE_EMISSION_MEAN"}
        for f in report.evaluation.findings
    )
    assert "PREDICTIVE_CHECK_FAILED" in {m.label for m in labels}
    assert all(set(m.model_dump()) == {"timestamp", "level", "label"} for m in labels)
    head = repository.head()

    def broken(*_args, **_kwargs):
        raise RuntimeError("unexpected generator defect")

    monkeypatch.setattr(simulation, "simulate_predictive_draws", broken)
    prepared = _prepare(study, state, n_days=5)
    selected = apply_effects(state, prepared.effects.produced, prepared.effects.retracted)
    with pytest.raises(RuntimeError, match="unexpected generator defect"):
        _edit(
            store.workspace_id,
            EditModelRequest(
                expected_revision=state.current["model"].revision,
                model=scientific_model_3_payload(),
            ),
            selected,
        )
    assert repository.head() == head
    assert (
        read_model_checks(
            store.workspace_id, repository.state(repository.head()), action="edit_model"
        )[0].predictive
        == report
    )


def test_joint_laws_can_be_checked_when_refitting_is_unsupported(study, monkeypatch):
    import jax

    from nof1_causal_lab.artifacts.checks import (
        Evaluated,
        NumericCriterionEvidence,
        PredictiveSubject,
    )
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
    )
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
    from tests.inference_fixtures import parameter_draws

    store, _ = study
    model = stress_sleep_model()
    initial, initial_checks, _, _ = _edit(
        store.workspace_id, EditModelRequest(expected_revision=None, model=model), _root(study)
    )
    state, _ = _publish(study, initial)
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=4)},
    )
    conditioned, _ = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(
            JointPosteriorDraws(
                parameter_draws(model, 3),
                jnp.zeros((3, 5, 2)),
            )
        ),
        times=jnp.arange(-1.0, 4.0),
    )
    fitted = store.write_artifact(
        "model",
        produced_by="fit",
        derived_from={"model": state.current["model"].revision, "panel": panel.revision},
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    evaluate_data_checks(
        store.workspace_id,
        state,
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel])),
    )
    from tests.inference_fixtures import inference_log

    fitted_log = inference_log(conditioned)
    # Historical fits used the first selected anchor, one day after this panel's
    # new support-boundary origin. Predictive checks must keep those coordinates.
    from nof1_causal_lab.artifacts.identity import GitRef

    historical_evidence = fitted_log.record.attempt.outcome.result.evidence.revised(
        time_origin=datetime(2024, 1, 2, tzinfo=UTC)
    )
    study[1].append(
        applied_record(
            Applied(
                result=ModelFitResult(
                    model=GitRef(
                        workspace_id=store.workspace_id,
                        revision=state.current["model"].revision,
                        path="model.json",
                    ),
                    panel=GitRef(
                        workspace_id=store.workspace_id,
                        revision=panel.revision,
                        path="panel.parquet",
                    ),
                    evidence=historical_evidence,
                ),
                effects=ActionEffects(produced=(panel, fitted)),
            ),
            seq=len(study[1].attempts()) + 1,
        )
    )
    state = apply_effects(state, [panel, fitted], [])
    calls = []

    def sample(panel, **kwargs):
        assert panel.time_origin == datetime(2024, 1, 2, tzinfo=UTC)
        np.testing.assert_array_equal(panel.times, [-1.0, 0.0, 1.0, 2.0, 3.0])
        draws = sample_model_laws(
            panel.model,
            draws=kwargs["draws"],
            key=jax.random.PRNGKey(kwargs["seed"]),
        )
        calls.append(draws)
        return

    monkeypatch.setattr(simulation, "generate_simulation_batch", sample)
    monkeypatch.setattr(
        simulation,
        "measure_simulation_batch",
        lambda *_a, **_k: (
            (
                Evaluated(
                    subject=PredictiveSubject(check="C1a finiteness", target="whole_model"),
                    outcome="passed",
                    evidence=(
                        NumericCriterionEvidence(
                            criterion="finite_fraction", value=1.0, lower=1.0, note="Finite."
                        ),
                    ),
                ),
            ),
            None,
        ),
    )
    checked, checked_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(
            expected_revision=fitted.revision,
            model=conditioned,
        ),
        state,
    )
    assert len(calls) == 1
    assert any(
        f.subject == "fit_laws" and f.kind == "evaluated" and f.outcome == "failed"
        for f in checked_checks.specification
    )
    assert checked_checks.predictive.status == "passed"
    assert checked_checks.predictive.law.kind == "fitted"
    assert checked_checks.predictive.law.interpretation == "in_sample_posterior_predictive"
    state = apply_effects(state, checked.effects.produced, checked.effects.retracted)
    # Preparing the same observations creates a new revision, not a held-out study.
    prepared = _prepare(study, state)
    assert {info.artifact_id for info in prepared.effects.produced} == {"panel"}
    state = apply_effects(state, prepared.effects.produced, prepared.effects.retracted)
    checked, checked_checks, _, _ = _edit(
        store.workspace_id,
        EditModelRequest(expected_revision=state.current["model"].revision, model=conditioned),
        state,
    )
    assert checked_checks.predictive.law.fitted_panel_revision == panel.revision
    assert checked_checks.predictive.law.interpretation == "posterior_predictive"

    import polars as pl

    # Neither a calendar mismatch nor observations before the retained fit state
    # may turn an otherwise valid model edit into a failed action.
    for calendar_free, detail in (
        (False, "before the fit's first retained state"),
        (True, "Calendar-free laws and calendar-bound observations"),
    ):
        metadata = panel_metadata()
        frame = panel_frame(n_days=4)
        if calendar_free:
            metadata = metadata.revised(time_origin=None)
        else:
            frame = frame.with_columns(
                pl.col("anchor_time", "support_start", "support_end").str.to_datetime()
                - timedelta(days=1)
            )
            metadata = metadata.revised(time_origin=datetime(2023, 12, 31, tzinfo=UTC))
        incompatible = store.write_artifact(
            "panel",
            produced_by="prepare_data",
            derived_from={},
            json_files={"metadata.json": metadata.model_dump(mode="json")},
            parquet_files={"panel.parquet": frame},
        )
        state = state.with_artifacts([incompatible])
        edited, edited_checks, _, _ = _edit(
            store.workspace_id,
            EditModelRequest(expected_revision=state.current["model"].revision, model=conditioned),
            state,
        )
        assert edited_checks.predictive.status == "not_evaluated"
        assert edited_checks.predictive.evaluation.reason == "NO_COMPATIBLE_PANEL"
        assert detail in edited_checks.predictive.evaluation.detail
        assert len(calls) == 2  # Only the two earlier, compatible checks generated a batch.
        state, messages = _publish(
            study,
            Applied(
                result=edited.result,
                effects=edited.effects.revised(produced=[incompatible, *edited.effects.produced]),
            ),
        )
        assert (
            read_model_checks(
                store.workspace_id, study[1].state(study[1].head()), action="edit_model"
            )[0].predictive
            == edited_checks.predictive
        )
        assert any(message.label == "SIMULATION_CHECK_NOT_EVALUATED" for message in messages)
