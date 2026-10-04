"""Scientific actions depend on selected inputs, independently of authoring recipes."""

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.actions.contracts import PrepareDataRequest, SimulateRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.checks import NumericCriterionEvidence
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import StudyState
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revision, commit_id, git_oid
from tests.helpers import write_question
from tests.inference_fixtures import compile_model_fixture, parameter_draws, particle_posterior
from tests.integration.runner_fixtures import panel_metadata
from tests.model_fixtures import stress_sleep_model, x_model, x_y_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid


@pytest.mark.inference(concern="predictive")
def test_current_law_sampling_preserves_joint_parameter_atoms():
    model = x_y_model()
    original = parameter_draws(model, 3)
    # Each atom has distinct coordinated values; independently resampling marginals
    # would produce combinations absent from the joint law.
    original = {
        name: value * jnp.arange(1, 4).reshape((3,) + (1,) * (value.ndim - 1))
        for name, value in original.items()
    }
    conditioned, _ = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(
            JointPosteriorDraws(
                original,
                jnp.zeros((3, 2, 2)),
            )
        ),
        times=jnp.array([0.0, 1.0]),
    )
    draws = sample_model_laws(
        compile_model_fixture(conditioned), draws=24, key=jax.random.PRNGKey(4)
    ).parameters
    for draw in range(24):
        assert any(
            all(np.allclose(draws[name][draw], values[atom]) for name, values in original.items())
            for atom in range(3)
        )
    authored = sample_model_laws(
        compile_model_fixture(model), draws=5, key=jax.random.PRNGKey(4)
    ).parameters
    assert all(value.shape[0] == 5 and np.isfinite(value).all() for value in authored.values())


@pytest.mark.inference(concern="simulation")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("fitted_laws", "scientific_model_payload"),
    [
        pytest.param(
            False,
            stress_sleep_model,
            id="False",
        ),
        pytest.param(
            True,
            stress_sleep_model,
            id="True",
        ),
    ],
)
def test_durable_replication_preserves_current_laws_without_comparison(
    tmp_path, monkeypatch, fitted_laws, scientific_model_payload
):
    from nof1_causal_lab.actions.runners import run_action_locally
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import ModelFitResult
    from nof1_causal_lab.study.snapshots import ModelReader
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import run_async
    from tests.inference_fixtures import inference_log
    from tests.integration.runner_fixtures import panel_frame

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("TEST"), StudyRepository("TEST")
    model = scientific_model_payload()
    definition = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": model.model_dump(mode="json")},
    )
    produced = [definition]
    question = write_question(store)
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision("TEST", "model", 1),
        "question": question.revision,
    }
    produced.append(
        store.write_artifact(
            "panel",
            produced_by="prepare_data",
            derived_from={},
            json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
            parquet_files={"panel.parquet": panel_frame(n_days=4)},
        )
    )
    if fitted_laws:
        from nof1_causal_lab.models.ssm.inference.persistence import condition_model
        from nof1_causal_lab.models.ssm.inference.types import (
            JointPosteriorDraws,
        )
        from tests.inference_fixtures import parameter_draws

        model, _ = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 3), jnp.zeros((3, 5, 2)))
            ),
            times=jnp.arange(-1.0, 4.0),
        )
        fitted = store.write_artifact(
            "model",
            produced_by="fit",
            derived_from={"model": definition.revision, "panel": produced[-1].revision},
            json_files={"model.json": model.model_dump(mode="json")},
        )
        pins["model"] = fitted.revision
        produced.append(fitted)
    state = StudyState().with_artifacts(produced)
    journal.append(
        applied_record(
            Applied(
                result=ModelFitResult(
                    model=GitRef(
                        workspace_id="TEST", revision=definition.revision, path="model.json"
                    ),
                    panel=GitRef(
                        workspace_id="TEST", revision=produced[1].revision, path="panel.parquet"
                    ),
                    evidence=inference_log(model).record.attempt.outcome.result.evidence,
                ),
                effects=ActionEffects(produced=tuple(produced)),
            )
            if fitted_laws
            else Applied(result=None, effects=ActionEffects(produced=tuple(produced))),
            seq=1,
            ts="2026-09-15T12:00:00Z",
        )
    )
    applied = run_async(
        run_action_locally(
            "TEST",
            SimulateRequest(
                model_revision=pins["model"],
                # The fit retains states from model day -1; its origin is 2024-01-01.
                start=date(2023, 12, 31) if fitted_laws else date(2026, 1, 1),
                horizon="4d",
            ),
            pins,
        )
    )
    from nof1_causal_lab.actions.simulate import read_simulation_report
    report = read_simulation_report(store, applied.result.evidence, pins["question"])
    evidence = report.evidence
    assert store.read_array(evidence.latent_paths).shape == (evidence.draws, 5, 2)
    assert store.read_array(evidence.observations).shape == (evidence.draws, 5, 2)
    assert evidence.model.revision == pins["model"]
    assert set(pins) == {"model", "question"}
    assert "predictive_checks" not in evidence.model_dump()
    assert "comparison_panel" not in evidence.model_dump()
    assert report.law is not None
    assert report.law.kind == ("fitted" if fitted_laws else "authored")
    if fitted_laws:
        assert report.law.interpretation == "posterior_predictive"
        assert report.law.fitted_panel_revision == produced[1].revision
    assert not applied.effects.produced
    assert any(finding.subject.check.startswith("C5c") for finding in report.findings)
    assert not any(finding.subject.check.startswith("C5d") for finding in report.findings)
    journal.append(applied_record(applied, seq=2, ts="2026-09-15T12:01:00Z"))
    current = ModelReader("TEST", at=StudyRepository("TEST").head()).simulation()
    assert current is not None
    assert current.value == report
    assert current.source.validity == "fresh"
    edited = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={
            "model.json": model.revised(
                edges=(
                    model.edges[0].revised(description="Revised justification"),
                    *model.edges[1:],
                )
            ).model_dump(mode="json")
        },
    )
    journal.append(
        applied_record(
            Applied(result=None, effects=ActionEffects(produced=[edited])),
            seq=3,
            ts="2026-09-15T12:02:00Z",
            trace_ids=[],
        )
    )
    revised = ModelReader("TEST", at=StudyRepository("TEST").head()).simulation()
    retained = ModelReader("TEST", at=commit_id("TEST", 2)).simulation()
    assert revised is not None
    assert retained is not None
    assert revised.source.validity == "stale"
    assert retained.source.validity == "fresh"

    # A retained generator result can be prepared without running it again.
    from nof1_causal_lab.actions.runners import run_action
    from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef

    preparation = PrepareDataRequest(
        input=SimulationReplicateRef(revision=commit_id("TEST", 2), replicate=1),
    )
    prepared = run_async(run_action("TEST", preparation, state))
    panel_info = next(info for info in prepared.effects.produced if info.artifact_id == "panel")
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    from nof1_causal_lab.models.ssm.runtime import project_observation_data

    projected = project_observation_data(
        panel, model_spec=compile_model_fixture(model), time_origin=report.time_origin
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    (wide, _) = projected
    np.testing.assert_allclose(
        wide.select(["stress_score", "sleep_score"]).to_numpy(),
        store.read_array(evidence.observations)[1],
        equal_nan=True,
    )


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize(
    ("start_time", "origin", "complete_test_model_payload"),
    [
        pytest.param(1.0, datetime(2026, 1, 1, tzinfo=UTC), x_y_model, id="1.0"),
        pytest.param(0.0, datetime(2026, 1, 2, tzinfo=UTC), x_y_model, id="0.0"),
        pytest.param(0.25, datetime(2026, 1, 1, 18, tzinfo=UTC), x_y_model, id="0.25"),
    ],
)
def test_retained_forecast_and_timed_intervention_preserve_joint_starts(
    tmp_path, monkeypatch, start_time, origin, complete_test_model_payload
):
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.scenarios import InterventionSpec
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = complete_test_model_payload()
    from nof1_causal_lab.models.ssm import numerics as numeric

    states = numeric.state_ids(compile_model_fixture(model))
    paths = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    model, _ = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(JointPosteriorDraws(parameter_draws(model, 2), paths)),
        times=jnp.array([0.0, 1.0]),
    )
    store = ArtifactStore("TEST")
    design = SimulationSpec(
        start=date(2026, 1, 2),
        horizon="1d",
        interventions=(InterventionSpec(target=states[0], value=2.0, after="12h"),),
    )
    assert design.start_day(origin) == start_time
    report = simulate(
        StructuralSelection(model, None),
        design,
        time_origin=origin,
        revision=GitRef(workspace_id="TEST", revision=git_oid(2), path="model.json"),
        write_array=store.write_array,
    )
    assert not isinstance(report, ObservationPreflightFailure)
    action = store.read_array(report.latent_paths)
    assert report.reference_latent_paths is not None
    reference = store.read_array(report.reference_latent_paths)
    np.testing.assert_allclose(action[:, 0], reference[:, 0])
    if start_time in (1.0, 0.0):
        for initial in action[:, 0]:
            assert any(np.allclose(initial, atom) for atom in np.asarray(paths[:, int(start_time)]))
    np.testing.assert_allclose(action[:, 1, 0], 2.0, atol=1e-5)
    assert not np.allclose(action[:, -1, 0], 2.0)
    assert np.isfinite(action).all()
    emissions = store.read_array(report.observations)
    assert np.isnan(emissions[:, :-1]).all()
    assert np.isfinite(emissions[:, -1]).all()


@pytest.mark.inference(concern="simulation")
def test_causal_action_uses_common_generator_and_requires_matching_engine_evidence(
    tmp_path, monkeypatch
):
    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.scenarios import InterventionSpec
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.inference_fixtures import inference_log

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = x_y_model()
    states = numeric.state_ids(compile_model_fixture(model))
    model, _ = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))),
        times=jnp.array([0.0, 1.0]),
    )
    design = SimulationSpec(
        start=date(2024, 1, 2),
        horizon="1d",
        interventions=(InterventionSpec(target=states[0], value=1.0),),
    )
    record = inference_log(model)
    store = ArtifactStore("TEST")
    evidence = simulate(
        StructuralSelection(model, states[1]),
        design,
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        revision=GitRef(workspace_id="TEST", revision=git_oid(2), path="model.json"),
        write_array=store.write_array,
    )
    assert not isinstance(evidence, ObservationPreflightFailure)
    from nof1_causal_lab.artifacts.availability import Unavailable
    from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from tests.inference_fixtures import _report
    generated = SimulationReport(evidence=evidence, law=AuthoredLawProvenance(), findings=(),
        fit_reliability="converged", causal=Unavailable(reason="Pending certification"))
    monkeypatch.setattr("nof1_causal_lab.study.store.read_model", lambda *_args: model)
    monkeypatch.setattr("nof1_causal_lab.study.lineage.fitted_law_report", lambda *_args: _report(model).core)
    result = summarize_causal_simulation(
        StructuralSelection(model, states[1]), generated, store=store, inference=record
    )
    assert result.causal.kind == "available"
    assert result.evidence.reference_latent_paths is not None
    assert result.evidence.model.revision == git_oid(2)
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.models.ssm.runtime import project_observation_data

    store = ArtifactStore("TEST")
    panel = prepare_simulation_panel(result.evidence, 0, read_array=store.read_array)
    projected = project_observation_data(
        panel, model_spec=compile_model_fixture(model), time_origin=panel_metadata().time_origin
    )
    assert not isinstance(projected, ObservationPreflightFailure)
    (wide, _) = projected
    np.testing.assert_allclose(
        wide.select(numeric.observation_names(compile_model_fixture(model))).to_numpy(),
        store.read_array(result.evidence.observations)[0],
        equal_nan=True,
    )
    from nof1_causal_lab.artifacts.checks import NotEvaluated

    report = _report(model)
    unavailable_report = report.core.revised(engine=NotEvaluated(
        subject="production_engine", reason="ARCHIVED_ENGINE_NOT_RETAINED", detail="Engine evidence not retained"))
    monkeypatch.setattr("nof1_causal_lab.study.lineage.fitted_law_report", lambda *_args: unavailable_report)
    bad = record
    rejected = summarize_causal_simulation(
        StructuralSelection(model, states[1]), generated, store=store, inference=bad
    )
    assert rejected.causal.kind == "unavailable"
    assert "retained exact-engine evidence" in rejected.causal.reason
    assert rejected.evidence.latent_paths == generated.evidence.latent_paths
    unavailable = summarize_causal_simulation(
        StructuralSelection(model, states[1]), generated, store=store, inference=None
    )
    assert unavailable.causal.kind == "unavailable"


@pytest.mark.contract
def test_data_profile_survives_model_edits(tmp_path, monkeypatch):
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks, require_data_binding
    from nof1_causal_lab.study.records import DataPreparationResult
    from nof1_causal_lab.study.state import apply_effects
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.integration.runner_fixtures import seed_panel

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    panel = seed_panel(store, model_revision=None)
    profile = evaluate_data_checks(
        "TEST",
        StudyState(),
        Applied(result=DataPreparationResult(), effects=ActionEffects(produced=[panel])),
    )
    state = StudyState().with_artifacts((panel,))
    assert set(profile.indicators) == {item.id for item in panel_metadata().variables}
    assert not state.has("model")
    revised_model = stress_sleep_model()
    revised_model = revised_model.revised(
        edges=(
            revised_model.edges[0].revised(description="Another scientific goal"),
            *revised_model.edges[1:],
        )
    )
    revised = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": revised_model.model_dump(mode="json")},
    )
    require_data_binding(store, revised_model, panel.revision)
    current = state.with_artifacts([revised])
    from nof1_causal_lab.actions.data_checks import read_data_profile
    assert read_data_profile(store, current.current["panel"].revision) == profile


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("groups", "complete_test_model_payload"),
    [
        pytest.param(
            (),
            x_model,
            id="groups0",
        ),
        pytest.param(
            ("dynamics",),
            x_model,
            id="groups1",
        ),
        pytest.param(
            ("measurement",),
            x_model,
            id="groups2",
        ),
        pytest.param(
            ("dynamics", "measurement"),
            x_model,
            id="groups3",
        ),
    ],
)
def test_simulation_selects_checks_before_execution_and_persists_only_parameters(
    monkeypatch, groups, complete_test_model_payload
):
    from importlib import import_module

    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.models.ssm import simulation_checks
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.reachability import CheckResult

    action = import_module("nof1_causal_lab.actions.simulate")
    simulation = import_module("nof1_causal_lab.models.ssm.predictive.simulation")
    model = complete_test_model_payload()
    paths = jnp.zeros((2, 3, 1))
    prediction = PredictiveDraws(
        parameters={"future_parameter_site": jnp.array([1.0, 2.0])},
        trajectory=PredictiveTrajectory(
            paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths
        ),
    )
    monkeypatch.setattr(
        simulation,
        "sample_model_laws",
        lambda *_a, **_k: JointPosteriorDraws(prediction.parameters),
    )
    monkeypatch.setattr(simulation, "simulate_predictive_draws", lambda *_a, **_k: prediction)
    called = []

    def measure(group):
        def execute(*_args, **_kwargs):
            assert group in groups, f"Unselected {group} checks executed"
            called.append(group)
            # Selection cannot depend on C-labels or other presentation text.
            return [
                CheckResult.measured(
                    f"{group} renamed check",
                    "X",
                    "ok",
                    "ok",
                    "ok",
                    outcome="passed",
                    measurements=(NumericCriterionEvidence(criterion="fixture", value=1.0),),
                )
            ], []

        return execute

    monkeypatch.setattr(simulation_checks, "measure_construct_dynamics", measure("dynamics"))
    monkeypatch.setattr(simulation_checks, "measure_construct_measurement", measure("measurement"))
    arrays = {}

    def write(values):
        key = str(len(arrays))
        arrays[key] = values
        return key

    evidence = action.simulate(
        StructuralSelection(model, None),
        SimulationSpec(start=date(2026, 1, 1), horizon="2d"),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        revision=GitRef(workspace_id="TEST", revision=git_oid(1), path="model.json"),
        write_array=write,
    )
    assert not isinstance(evidence, ObservationPreflightFailure)
    assert called == []
    assert evidence.parameter_draws.keys() == prediction.parameters.keys()
    np.testing.assert_array_equal(arrays[evidence.parameter_draws["future_parameter_site"]], [1.0, 2.0])
