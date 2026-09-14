from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

import nof1_causal_lab.tool_server as tool_server
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identification import (
    IdentifiabilityStatus,
    IdentificationReport,
    IdentifiedTreatmentStatus,
)
from nof1_causal_lab.artifacts.identity import ConstructRef, ModelRevision
from nof1_causal_lab.artifacts.scenarios import SimulationResult
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from tests.dynamics_fixtures import decay_term, hill_term
from tests.helpers import fixture_entity_id
from tests.inference_fixtures import inference_log
from tests.model_fixtures import model_fixture, parameter_draws


def _identification(treatment: str, outcome: str) -> IdentificationReport:
    return IdentificationReport(
        outcome=fixture_entity_id("construct", outcome),
        status=IdentifiabilityStatus(
            identifiable_treatments={
                fixture_entity_id("construct", treatment): IdentifiedTreatmentStatus(
                    method="do_calculus",
                    estimand=f"E[{outcome} | do({treatment})]",
                )
            }
        ),
    )


def _certified_simulation_context(
    *,
    samples: dict[str, jnp.ndarray],
    spec: Any,
    runtime: Any,
    treatment: str,
    outcome: str,
    latent_paths: jnp.ndarray | None = None,
    timestamps: list[datetime] | None = None,
) -> dict[str, Any]:
    design = spec.revised(default_outcome=ConstructRef(id=fixture_entity_id("construct", outcome)))
    design_ref = ModelRevision(workspace_id="test-workspace", version=2)
    design = condition_model(
        design,
        ParticleMCMCPosterior(
            draws=JointPosteriorDraws(
                parameters={
                    **parameter_draws(design, next(iter(samples.values())).shape[0]),
                    **samples,
                },
                latent_paths=latent_paths
                if latent_paths is not None
                else jnp.zeros(
                    (
                        next(iter(samples.values())).shape[0],
                        len(runtime.times),
                        numeric.n_states(design),
                    )
                ),
            )
        ),
        times=runtime.times,
    )
    record = inference_log(design)

    analysis = CertifiedCausalAnalysis(
        model=design,
        identification=_identification(treatment, outcome),
        model_revision=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                _identification(treatment, outcome),
                model_revision=design_ref,
                treatment=treatment,
                outcome=outcome,
            ),
        ),
        inference=record,
    )
    return {
        "_workspace_id": "test",
        "_causal_analysis": analysis,
        "_prepared_runtime": runtime,
        "_simulation": tool_server._LoadedSimulation(
            design,
            record,
            runtime,
        ),
        "_identifiable_treatments": [treatment],
        "_outcome_name": outcome,
        "_observation_timestamps": timestamps or [],
        "causal_design": {"causal_design": design.model_dump(mode="json")},
        "baseline_report": {},
        "posterior": {"assessment": {"ppc": {"per_variable_warnings": []}}},
    }


def test_execute_tool_rejects_invalid_input_before_invoking_tool(monkeypatch):
    client = TestClient(tool_server.app)
    called = False

    def fake_impl(_ctx, _args):
        nonlocal called
        called = True
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_execute_tool_surfaces_unexpected_exception_detail(monkeypatch):
    client = TestClient(tool_server.app)

    monkeypatch.setattr(tool_server, "_build_context", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        lambda _ctx, _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {"model_json": "{}"}},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "message": "boom",
            "exception_type": "RuntimeError",
            "context_id": "latent-structure",
            "tool_name": "validate_latent_structure",
        }
    }


def test_build_ranking_context_rehydrates_runtime_from_persisted_spec(monkeypatch, tmp_path):
    import polars as pl

    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.moves import RunOperation
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import complete_test_model, make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    design = complete_test_model(
        make_model(["screen_time", "sleep_quality"], [("screen_time", "sleep_quality")])
    )
    design = design.revised(
        default_outcome=ConstructRef(id=fixture_entity_id("construct", "sleep_quality"))
    )
    check_execution(design)
    conditioned = condition_model(
        design,
        ParticleMCMCPosterior(
            JointPosteriorDraws(parameter_draws(design, 1), jnp.zeros((1, 1, 2)))
        ),
        times=jnp.array([0.0]),
    )
    report = InferenceReport.model_validate(
        inference_log(conditioned).diagnostics["report"]
    ).model_dump(mode="json")
    model_data = pl.DataFrame(
        {
            "indicator_id": [design.indicators[1].id],
            "value": [1.0],
            "anchor_time": ["2024-01-01T00:00:00"],
        }
    )
    store = ArtifactStore("user-123")
    definition = store.write_version(
        "model",
        provenance="llm",
        derived_from={},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": design.model_dump(mode="json")},
    )

    identification_report = store.write_version(
        "identification_report",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="derive:identification_report",
        json_files={
            "identification_report.json": _identification(
                "screen_time", "sleep_quality"
            ).model_dump(mode="json")
        },
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": model_data},
    )
    fitted = store.write_version(
        "model",
        provenance="computed",
        derived_from={"model": 1, "panel": 1},
        produced_by="run:posterior",
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    journal = EpisodeJournal("user-123")
    for seq, operation, produced in (
        (
            1,
            "statistical_model_spec",
            [definition, identification_report],
        ),
        (2, "measurements", [panel]),
        (3, "posterior", [fitted]),
    ):
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                move=RunOperation(operation_id=operation),
                status="applied",
                produced=produced,
                diagnostics=inference_log(conditioned, report=report).diagnostics
                if operation == "posterior"
                else {},
                trace_ids=[],
                resume=None,
            )
        )

    rebuilt_runtime = SimpleNamespace(
        observation_support="support-runtime",
        observation_data=None,
    )
    captured: dict[str, Any] = {}
    loads = 0

    def fake_prepare_model_runtime(*, data_for_model, model, model_spec, sampler_config=None):
        nonlocal loads
        loads += 1
        del sampler_config
        assert model_spec == conditioned
        captured["data_for_model"] = data_for_model
        captured["model"] = model
        return rebuilt_runtime

    monkeypatch.setattr(tool_server, "prepare_model_runtime", fake_prepare_model_runtime)

    ctx = tool_server._build_ranking_context("user-123")

    assert isinstance(captured["model"], tool_server.SSMModel)
    # The runtime uses the model revision and panel pinned by the posterior.
    assert captured["model"].spec == conditioned
    assert list(numeric.state_names(captured["model"].spec)) == ["screen_time", "sleep_quality"]
    assert captured["data_for_model"].equals(model_data)
    assert ctx["_prepared_runtime"] is rebuilt_runtime
    assert ctx["model"] == conditioned.model_dump(mode="json")
    assert ctx["inference_report"] == report
    assert ctx["_outcome_name"] == "sleep_quality"
    assert ctx["_identifiable_treatments"] == ["screen_time"]
    again = tool_server._build_ranking_context("user-123")
    assert again["_simulation"] is ctx["_simulation"]
    assert loads == 1
    new_panel = store.write_version(
        "panel", provenance="computed", derived_from={"model": 1}, produced_by="run:measurements"
    )
    journal.append(
        TransitionRecord(
            seq=4,
            ts="2026-07-03T01:00:00+00:00",
            move=RunOperation(operation_id="measurements"),
            status="applied",
            produced=[new_panel],
            trace_ids=[],
            resume=None,
        )
    )
    with pytest.raises(tool_server.HTTPException, match="conditioned model") as stale:
        tool_server._build_ranking_context("user-123")
    assert stale.value.status_code == 409
    assert loads == 1
    tool_server._load_simulation.cache_clear()


def test_simulate_counterfactual_respects_estimand_shape(monkeypatch):
    from nof1_causal_lab.models.ssm.dynamics import DynamicsSpec

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            *(decay_term(target=i) for i in range(2)),
            hill_term(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 2
    samples = {
        "vf_0_decay": jnp.full((n_draws,), 0.5),
        "vf_1_decay": jnp.full((n_draws,), 0.5),
        "vf_2_Emax": jnp.full((n_draws,), 1.5),
        "vf_2_EC50": jnp.full((n_draws,), 1.0),
        "vf_2_n": jnp.full((n_draws,), 2.0),
    }
    captured_initial_states: list[jnp.ndarray] = []

    def fake_vmap_simulate(
        vector_field,
        param_samples,
        initial_states,
        clamps,
        *,
        time_grid,
        config,
    ):
        del vector_field, clamps
        captured_initial_states.append(initial_states)
        n_draws = len(param_samples)
        n_t = time_grid.shape[0]
        # n_latent = 2 in this test setup
        n_latent = 2
        baseline_per_t = jnp.array([0.5, 5.0], dtype=jnp.float32)
        effect_per_t = jnp.array([1.5, 3.0], dtype=jnp.float32)
        baseline = jnp.broadcast_to(baseline_per_t, (n_draws, n_t, n_latent))
        effect = jnp.broadcast_to(effect_per_t, (n_draws, n_t, n_latent))
        counterfactual = baseline + effect
        return baseline, counterfactual, effect

    monkeypatch.setattr(tool_server, "vmap_simulate_clamps_from_state", fake_vmap_simulate)

    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=model_fixture(n_latent=2, dynamics_spec=spec, latent_names=["treat", "outcome"]),
        runtime=runtime,
        treatment="treat",
        outcome="outcome",
        latent_paths=jnp.array(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]],
                [[0.0, 0.0], [4.0, 5.0], [6.0, 7.0]],
            ]
        ),
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
    args = {
        "start": {"kind": "abducted"},
        "outcome": {"id": fixture_entity_id("construct", "outcome")},
        "clamps": [
            {
                "target": {"id": fixture_entity_id("construct", "treat")},
                "mode": "shift",
                "amount": 1.0,
            }
        ],
        "readout": {"horizon_days": 3},
    }

    expected_visualization = {
        "reference_node_trajectories": {
            fixture_entity_id("construct", "treat"): [0.5, 0.5, 0.5],
            fixture_entity_id("construct", "outcome"): [5.0, 5.0, 5.0],
        },
        "action_node_trajectories": {
            fixture_entity_id("construct", "treat"): [2.0, 2.0, 2.0],
            fixture_entity_id("construct", "outcome"): [8.0, 8.0, 8.0],
        },
        "node_effect_trajectories": {
            fixture_entity_id("construct", "treat"): [1.5, 1.5, 1.5],
            fixture_entity_id("construct", "outcome"): [3.0, 3.0, 3.0],
        },
        "start_state": {
            fixture_entity_id("construct", "treat"): 4.0,
            fixture_entity_id("construct", "outcome"): 5.0,
        },
    }
    expected_start = {
        "kind": "abducted",
        "time_index": 2,
        "time": "2024-01-03T00:00:00+00:00",
        "state_source": "fitted_latent_paths",
    }

    end_state = tool_server._execute_simulate(
        ctx,
        {**args, "readout": {**args["readout"], "estimand": "end_state"}},
    )["result"]
    trajectory = tool_server._execute_simulate(
        ctx,
        {**args, "readout": {**args["readout"], "estimand": "trajectory"}},
    )["result"]

    assert end_state["request"]["readout"]["estimand"] == "end_state"
    assert end_state["summary"]["mean"] == pytest.approx(3.0)
    assert end_state["reference_mean"] == pytest.approx(5.0)
    assert end_state["effect_trajectory"] is None
    assert "rung" not in end_state
    assert end_state["provenance"]["start_time_index"] == expected_start["time_index"]
    assert end_state["provenance"]["start_time"] == expected_start["time"]
    assert end_state["request"]["start"] == {"kind": "abducted", "time_index": None, "time": None}
    SimulationResult.model_validate(end_state)
    SimulationResult.model_validate(trajectory)
    clamp = end_state["request"]["clamps"][0]
    assert clamp["mode"] == "shift"
    assert clamp["amount"] == 1.0
    assert clamp["target"]["id"] == fixture_entity_id("construct", "treat")
    assert end_state["visualization"] == expected_visualization
    assert all(
        jnp.array_equal(initial_states, jnp.array([[2.0, 3.0], [6.0, 7.0]]))
        for initial_states in captured_initial_states
    )

    assert trajectory["request"]["readout"]["estimand"] == "trajectory"
    assert trajectory["summary"]["mean"] == pytest.approx(3.0)
    assert trajectory["reference_mean"] == pytest.approx(5.0)
    assert trajectory["effect_trajectory"] == [
        {"day": 1.0, "effect": 3.0},
        {"day": 2.0, "effect": 3.0},
        {"day": 3.0, "effect": 3.0},
    ]
    assert trajectory["provenance"]["start_time_index"] == expected_start["time_index"]
    assert trajectory["visualization"] == expected_visualization


@pytest.mark.cpu_expensive
def test_simulate_intervention_dispatches_to_vector_field_path():
    """A particle posterior drives the true nonlinear baseline-start simulation."""

    from nof1_causal_lab.models.ssm.dynamics import DynamicsSpec

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            *(decay_term(target=i) for i in range(2)),
            hill_term(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 4
    samples = {
        "vf_0_decay": jnp.full((n_draws,), 0.5),
        "vf_1_decay": jnp.full((n_draws,), 0.5),
        "vf_2_Emax": jnp.full((n_draws,), 1.5),
        "vf_2_EC50": jnp.full((n_draws,), 1.0),
        "vf_2_n": jnp.full((n_draws,), 2.0),
    }

    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=model_fixture(n_latent=2, dynamics_spec=spec, latent_names=["src", "tgt"]),
        runtime=runtime,
        treatment="src",
        outcome="tgt",
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
    args = {
        "start": {"kind": "baseline"},
        "outcome": {"id": fixture_entity_id("construct", "tgt")},
        "clamps": [
            {
                "target": {"id": fixture_entity_id("construct", "src")},
                "mode": "shift",
                "amount": 0.5,
            }
        ],
        "readout": {"horizon_days": 3, "estimand": "end_state"},
    }

    response = tool_server._execute_simulate(ctx, args)
    result = response["result"]
    assert "error" not in result, f"vector-field dispatch failed: {result}"
    assert result["request"]["start"]["kind"] == "baseline"
    assert result["request"]["readout"]["estimand"] == "end_state"
    # Effect on tgt from shifting src up should be positive (Hill saturates).
    summary = result["summary"]
    assert summary["mean"] > 0
    # Almost-certainly-positive contrast — Hill is monotonic in src
    assert summary["prob_positive"] == pytest.approx(1.0)


@pytest.mark.cpu_expensive
def test_simulate_counterfactual_dispatches_to_vector_field_path():
    """Rung-3 on vector-field starts from retained fitted trajectory draws."""

    from nof1_causal_lab.models.ssm.dynamics import DynamicsSpec

    spec = DynamicsSpec(
        n_latent=2,
        components=(
            *(decay_term(target=i) for i in range(2)),
            hill_term(
                source=0,
                target=1,
            ),
        ),
    )
    n_draws = 3
    samples = {
        "vf_0_decay": jnp.full((n_draws,), 0.5),
        "vf_1_decay": jnp.full((n_draws,), 0.5),
        "vf_2_Emax": jnp.full((n_draws,), 1.5),
        "vf_2_EC50": jnp.full((n_draws,), 1.0),
        "vf_2_n": jnp.full((n_draws,), 2.0),
    }
    latent_paths = jnp.tile(jnp.array([[1.0, 0.5], [0.9, 0.6], [0.8, 0.7]]), (n_draws, 1, 1))
    runtime = SimpleNamespace(
        observations=jnp.zeros((3, 1)),
        times=jnp.array([0.0, 1.0, 2.0]),
    )
    ctx = _certified_simulation_context(
        samples=samples,
        spec=model_fixture(n_latent=2, dynamics_spec=spec, latent_names=["src", "tgt"]),
        runtime=runtime,
        treatment="src",
        outcome="tgt",
        latent_paths=latent_paths,
        timestamps=[
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
        ],
    )
    args = {
        "start": {"kind": "abducted"},
        "outcome": {"id": fixture_entity_id("construct", "tgt")},
        "clamps": [
            {
                "target": {"id": fixture_entity_id("construct", "src")},
                "mode": "shift",
                "amount": 0.5,
            }
        ],
        "readout": {"horizon_days": 3, "estimand": "end_state"},
    }

    response = tool_server._execute_simulate(ctx, args)
    result = response["result"]
    assert "error" not in result, f"vector-field abducted start failed: {result}"
    assert result["request"]["readout"]["estimand"] == "end_state"
    assert result["request"]["start"]["kind"] == "abducted"
    assert result["provenance"]["start_time_index"] == 2
    assert result["provenance"]["start_time"] == "2024-01-03T00:00:00+00:00"
    # Shift on src should produce a positive effect on tgt via the Hill chain
    assert result["summary"]["mean"] > 0


def test_get_tool_schemas_exposes_declared_result_schema():
    client = TestClient(tool_server.app)

    response = client.get("/api/tools/ranking")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["get_model_info"]["result"] is None
    assert tools["simulate"]["result"] is not None


def test_manifest_effects_include_interval_supported_outcome_indicators():
    samples = {
        "lambda": jnp.array(
            [
                [
                    [0.0, -1.0],
                    [0.0, 0.5],
                ]
            ]
        )
    }

    effects = tool_server._manifest_effects(
        samples,
        outcome_idx=1,
        effect_mean=0.25,
        manifest_names=["sleep_problem_search_count", "sleep_duration_hours"],
    )

    assert effects == {
        "sleep_problem_search_count": pytest.approx(-0.25),
        "sleep_duration_hours": pytest.approx(0.125),
    }


def test_get_model_info_uses_structure_for_variables_and_treatments():

    from tests.helpers import complete_test_model, make_model

    model = make_model(["screen_time", "sleep"], [("screen_time", "sleep")])
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(
                    update={"indicators": (c.indicators[0].model_copy(update={"name": name}),)}
                )
                for c, name in zip(
                    model.constructs, ["daily_event_count", "sleep_issue_searches"], strict=True
                )
            ),
        ),
        default_outcome=ConstructRef(id=model.constructs[1].id),
    )
    model = complete_test_model(model)
    spec = model
    ctx = {
        "model": model.model_dump(mode="json"),
        "posterior": {"inference_metadata": {"method": "marginal_particle_gibbs"}},
        "baseline_report": {},
        "_prepared_runtime": SimpleNamespace(spec=spec),
        "_fitted_artifact": SimpleNamespace(spec=spec),
        "_identifiable_treatments": ["screen_time"],
        "_outcome_name": "sleep",
        "_observation_timestamps": [],
    }

    payload = tool_server._build_model_info_payload(
        ctx,
        {"sections": ["overview", "variables", "capabilities"]},
    )

    assert payload["overview"]["treatments"] == ["screen_time"]
    assert [item["name"] for item in payload["variables"]["constructs"]] == ["screen_time", "sleep"]
    assert [item["id"] for item in payload["variables"]["constructs"]] == [
        construct.id for construct in model.constructs
    ]
    assert [item["name"] for item in payload["variables"]["indicators"]] == [
        "daily_event_count",
        "sleep_issue_searches",
    ]
    assert payload["capabilities"]["simulate"]["supported_targets"] == ["screen_time"]
