"""End-to-end tests for the gradual construct-admission engine."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.prior import ExecutablePrior
from nof1_causal_lab.artifacts.statistical_model_spec import (
    DistributionFamily,
    LikelihoodSpec,
    LinkFunction,
    ParameterSpec,
)
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    AdmissionTiming,
    ConstructContribution,
    DesignInfo,
    _conditional_variance_for_signal,
    _incoming_edge_off_target,
    _resimulate_edge_off,
    _run_battery,
    admit_construct,
    build_construct_order,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.utils.structural_plan import (
    get_edges,
    get_known_inputs,
    get_manifest_indicators,
    get_state_names,
    restrict_structural_plan,
)
from tests.helpers import fixture_entity_id
from tests.models.ssm.test_dag_to_ssm import _make_causal_design_dict

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.model import SSMSpec

_SOFT_CHECKS = {
    "C1b confinement",
    "C2 latent scale",
    "C3 resolvability",
    "C4b edge overwhelm",
    "C4c saturation",
    "C5a location reach",
    "C5b width",
    "C5c transmission",
    "C5d data availability",
}
_TARGETS = {"X", "Y", "Z", "x1", "x2", "y1", "z1", "X->Y", "Y->Z"}
_ALL_SOFT = {(check, target): "t" for check in _SOFT_CHECKS for target in _TARGETS}


def _structural_plan() -> StructuralPlan:
    return build_structural_plan(CausalDesign.model_validate(_make_causal_design_dict()))


def _lik(var: str) -> LikelihoodSpec:
    return LikelihoodSpec(
        indicator_id=fixture_entity_id("indicator", var),
        distribution=DistributionFamily.GAUSSIAN,
        link=LinkFunction.IDENTITY,
        reasoning="test",
    )


def _p(name: str) -> ParameterSpec:
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import ParamCatalog

    return ParameterSpec.model_validate(
        ParamCatalog.from_structural_plan(_structural_plan()).metadata_for(name)
    )


def _normal(parameter: str, mu: float, sigma: float) -> ExecutablePrior:
    return ExecutablePrior.model_validate(
        {
            "parameter_id": _p(parameter).id,
            "distribution": "Normal",
            "params": {"mu": mu, "sigma": sigma},
        }
    )


def _halfnormal(parameter: str, sigma: float) -> ExecutablePrior:
    return ExecutablePrior.model_validate(
        {
            "parameter_id": _p(parameter).id,
            "distribution": "HalfNormal",
            "params": {"sigma": sigma},
        }
    )


def _mechanisms_for(name):
    from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import ParamCatalog
    from nof1_causal_lab.models.model_mechanisms import declare_dynamics_mechanisms

    plan = _structural_plan()
    target = next(item.id for item in plan.semantics.constructs.values() if item.name == name)
    parameters = [
        ParameterSpec.model_validate(row)
        for row in ParamCatalog.from_structural_plan(plan).metadata.values()
    ]
    return tuple(
        mechanism
        for mechanism in declare_dynamics_mechanisms(plan, parameters)
        if (
            mechanism.target_id == target
            if (mechanism.kind == "node_potential" or mechanism.kind == "constant_drift")
            else plan.semantics.edges[mechanism.edge_id].effect_id == target
        )
    )


def _contrib_X() -> ConstructContribution:
    return ConstructContribution(
        name="X",
        mechanisms=_mechanisms_for("X"),
        likelihoods=(_lik("x1"), _lik("x2")),
        parameters=(
            _p("rho_X"),
            _p("sigma_X"),
            _p("lambda_x2_X"),
            _p("obs_sd_x1"),
            _p("obs_sd_x2"),
        ),
        priors={
            "rho_X": _normal("rho_X", 0.6, 0.1),
            "sigma_X": _halfnormal("sigma_X", 0.5),
            "lambda_x2_X": _normal("lambda_x2_X", 1.0, 0.2),
            "obs_sd_x1": _halfnormal("obs_sd_x1", 0.3),
            "obs_sd_x2": _halfnormal("obs_sd_x2", 0.3),
        },
    )


def _contrib_child(name: str, indicator: str, parent: str) -> ConstructContribution:
    return ConstructContribution(
        name=name,
        mechanisms=_mechanisms_for(name),
        likelihoods=(_lik(indicator),),
        parameters=(
            _p(f"rho_{name}"),
            _p(f"sigma_{name}"),
            _p(f"beta_{parent}_{name}"),
        ),
        priors={
            f"rho_{name}": _normal(f"rho_{name}", 0.6, 0.1),
            f"sigma_{name}": _halfnormal(f"sigma_{name}", 0.5),
            f"beta_{parent}_{name}": _normal(f"beta_{parent}_{name}", 0.3, 0.1),
        },
        edge_parents=(parent,),
    )


def _design(seed: int = 0) -> DesignInfo:
    t_grid = jnp.linspace(0.0, 10.0, 201)
    obs_idx = np.arange(1, 201, 2)  # 100 observations, shared across indicators here
    rng = np.random.default_rng(0)
    indicators = tuple(fixture_entity_id("indicator", name) for name in ("x1", "x2", "y1", "z1"))
    return DesignInfo(
        manifest_ids=indicators,
        t_grid=t_grid,
        obs_index_by_indicator=dict.fromkeys(indicators, obs_idx),
        values_by_indicator={v: rng.normal(0.0, 0.9, obs_idx.size) for v in indicators},
        n_draws=64,
        seed=seed,
    )


def test_build_construct_order_is_topological():
    order = build_construct_order(_structural_plan())
    assert order.index("X") < order.index("Y") < order.index("Z")


def test_admit_construct_records_shared_and_diagnostic_timings(monkeypatch):
    from nof1_causal_lab.models.ssm import construct_admission as admission_module

    diagnostic = AdmissionTiming(
        phase="c1_confinement",
        label="C1 confinement",
        duration_ms=4.0,
        checks=("C1a finiteness",),
    )
    monkeypatch.setattr(admission_module, "_compile_partial", lambda *_args: (object(), object()))
    monkeypatch.setattr(admission_module, "_sample_partial", lambda *_args: {})
    monkeypatch.setattr(
        admission_module,
        "_run_battery",
        lambda *_args: (
            [CheckResult("C1a finiteness", "X", "0%", "0%", True, "ok")],
            [diagnostic],
        ),
    )

    _state, report = admission_module.admit_construct(
        AdmissionState(),
        ConstructContribution(name="X"),
        _structural_plan(),
        _design(),
    )

    assert [timing.phase for timing in report.timings] == [
        "model_compilation",
        "prior_predictive",
        "c1_confinement",
        "admission_decision",
    ]
    assert all(timing.duration_ms >= 0 for timing in report.timings)


def test_conditional_variance_uses_observation_family_moments():
    signal = np.array([[0.1, 0.5], [0.2, 0.8]])
    pred = {
        "manifest_cov": np.array([[[0.25]], [[1.0]]]),
        "obs_df": np.array([5.0, 1.5]),
        "obs_shape": np.array([2.0, 4.0]),
        "obs_r": np.array([3.0, 6.0]),
        "obs_concentration": np.array([9.0, 19.0]),
    }

    gaussian = _conditional_variance_for_signal(DistributionFamily.GAUSSIAN, signal, pred, 0)
    np.testing.assert_allclose(gaussian, [[0.25, 0.25], [1.0, 1.0]])
    np.testing.assert_allclose(
        _conditional_variance_for_signal(DistributionFamily.POISSON, signal, pred, 0),
        signal,
    )
    np.testing.assert_allclose(
        _conditional_variance_for_signal(DistributionFamily.BERNOULLI, signal, pred, 0),
        signal * (1.0 - signal),
    )
    student = _conditional_variance_for_signal(DistributionFamily.STUDENT_T, signal, pred, 0)
    np.testing.assert_allclose(student[0], np.full(2, 0.25 * 5.0 / 3.0))
    assert np.isinf(student[1]).all()


def test_time_invariant_construct_omits_temporal_transmission_check():
    draws = 200
    times = 20
    static_values = np.linspace(-1.5, 1.5, draws)
    latent = np.broadcast_to(static_values[:, None, None], (draws, times, 1))
    expected = latent.copy()
    pred = {
        "latents": latent,
        "observations": expected,
        "expected_observations": expected,
        "manifest_cov": np.broadcast_to(np.array([[[0.25]]]), (draws, 1, 1)),
    }
    spec: Any = SimpleNamespace(
        latent_names=["static"],
        manifest_names=["static_indicator"],
        manifest_links=[LinkFunction.IDENTITY],
        manifest_level_counts=None,
        dynamics_spec=SimpleNamespace(components=()),
        diffusion_block=SimpleNamespace(time_invariant_mask=np.array([True])),
    )
    obs_idx = np.arange(times)
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "static_indicator"),),
        t_grid=jnp.arange(times, dtype=float),
        obs_index_by_indicator={fixture_entity_id("indicator", "static_indicator"): obs_idx},
        values_by_indicator={
            fixture_entity_id("indicator", "static_indicator"): np.linspace(-1.0, 1.0, times)
        },
    )
    target = ConstructContribution(
        name="static",
        likelihoods=(_lik("static_indicator"),),
    )

    results, _timings = _run_battery(spec, pred, design, target)
    checks = {result.check for result in results}
    assert {"C5a location reach", "C5b width"} <= checks
    assert "C5c transmission" not in checks


def test_build_construct_order_covers_only_estimation_states():
    """Constructs marginalized/anchored/dropped out of the estimation
    projection carry no state — nothing to admit for them."""
    causal_design = _make_causal_design_dict()
    causal_design["latent"]["constructs"].append(
        {
            "id": "construct:77b5a7df0a9c99cf2e02",
            "name": "M",
            "description": "Marginalized confounder",
            "role": "endogenous",
            "temporal_status": "time_varying",
        }
    )
    order = build_construct_order(build_structural_plan(CausalDesign.model_validate(causal_design)))
    assert order == ["X", "Y", "Z"]


def test_build_construct_order_admits_lagged_feedback_cycles():
    """Lagged feedback loops sort as a unit: cycle members adjacent, parents first."""
    causal_design = _make_causal_design_dict()
    feedback = {
        "cause_id": "construct:a6b7873d58dac1ff1a02",
        "effect_id": "construct:d90c52e59b79004188dc",
        "id": "edge:e122f01ff2b4e016111d",
        "description": "Z feeds back on Y",
        "lagged": True,
    }
    causal_design["latent"]["edges"].append(dict(feedback))
    order = build_construct_order(build_structural_plan(CausalDesign.model_validate(causal_design)))
    assert order == ["X", "Y", "Z"]


def test_restrict_structural_plan_to_subset():
    restricted = restrict_structural_plan(_structural_plan(), {"X", "Y"})
    assert get_state_names(restricted) == ["X", "Y"]
    assert {indicator["name"] for indicator in get_manifest_indicators(restricted)} == {
        "x1",
        "x2",
        "y1",
    }
    assert all(edge["effect"] != "Z" for edge in get_edges(restricted))


def test_restrict_structural_plan_preserves_known_input_dependency():
    causal_design = _make_causal_design_dict()
    causal_design["known_inputs"] = [
        {
            "construct_id": "construct:311c9047b5ede16a8f26",
            "source_indicator_id": "indicator:0f93ce57e1f1d1c96f5c",
            "scale": 10.0,
            "missing_policy": "forward_fill",
        }
    ]

    plan = build_structural_plan(CausalDesign.model_validate(causal_design))
    restricted = restrict_structural_plan(plan, {"Y"})

    assert get_state_names(restricted) == ["Y"]
    assert [
        {
            "construct_id": fixture_entity_id("construct", item["construct"]),
            "source_indicator_id": fixture_entity_id("indicator", item["source_indicator"]),
            "scale": item["scale"],
            "missing_policy": item["missing_policy"],
        }
        for item in get_known_inputs(restricted)
    ] == [
        {
            "construct_id": "construct:311c9047b5ede16a8f26",
            "source_indicator_id": "indicator:0f93ce57e1f1d1c96f5c",
            "scale": 10.0,
            "missing_policy": "forward_fill",
        }
    ]
    assert [(edge["cause"], edge["effect"]) for edge in get_edges(restricted)] == [("X", "Y")]
    assert {indicator["name"] for indicator in get_manifest_indicators(restricted)} == {"y1"}


def test_known_input_edge_off_zeroes_only_the_compiled_input_cell(monkeypatch):
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    captured: dict[str, np.ndarray] = {}

    def _capture_samples(
        _spec,
        samples,
        _times,
        *,
        transition_inputs,
        rng_key,
    ):
        del transition_inputs, rng_key
        captured["input_effect"] = np.asarray(samples["input_effect"])
        return jnp.zeros((2, 3, 2)), jnp.zeros((2, 3, 1))

    monkeypatch.setattr(
        registry_runtime,
        "_simulate_vector_field_predictive_latents",
        _capture_samples,
    )
    spec = cast("SSMSpec", SimpleNamespace(input_names=["dose", "exercise"]))
    contribution = ConstructContribution(name="mood", edge_parents=("dose",))
    edge_target = _incoming_edge_off_target(spec, contribution, ["mood", "sleep"], 0)
    input_effect = jnp.arange(8, dtype=float).reshape(2, 2, 2) + 1.0

    _resimulate_edge_off(
        spec,
        {"input_effect": input_effect},
        jnp.arange(3, dtype=float),
        edge_target,
        seed=1,
    )

    expected = np.asarray(input_effect).copy()
    expected[:, 0, 0] = 0.0
    np.testing.assert_allclose(captured["input_effect"], expected)


@pytest.mark.slow
def test_admit_root_runs_full_battery():
    structural_plan = _structural_plan()
    state, report = admit_construct(
        AdmissionState(), _contrib_X(), structural_plan, _design(), accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert {"C1a finiteness", "C1b confinement", "C2 latent scale", "C3 resolvability"} <= ids
    assert {"C5a location reach", "C5b width", "C5c transmission"} <= ids
    assert "C4b edge overwhelm" not in ids  # root has no incoming edge
    timing_phases = {timing.phase for timing in report.timings}
    assert {
        "model_compilation",
        "prior_predictive",
        "c1_confinement",
        "c2_latent_scale",
        "c3_resolvability",
        "admission_decision",
    } <= timing_phases
    assert all(timing.duration_ms > 0 for timing in report.timings)
    # Hard checks (finite sim + reachable data) hold, so X is admitted.
    assert not report.outcome.startswith("BLOCKED")
    assert report.admitted
    assert state.names == ("X",)


@pytest.mark.slow
def test_admit_child_runs_edge_check_via_edge_off_resim():
    structural_plan = _structural_plan()
    design = _design()
    state, _ = admit_construct(
        AdmissionState(), _contrib_X(), structural_plan, design, accepted=_ALL_SOFT
    )
    state, report = admit_construct(
        state, _contrib_child("Y", "y1", "X"), structural_plan, design, accepted=_ALL_SOFT
    )
    ids = {r.check for r in report.results}
    assert "C4b edge overwhelm" in ids
    c4b = next(r for r in report.results if r.check == "C4b edge overwhelm")
    # The edge-off re-simulation must actually differ from edge-on (the edge moves
    # the child); a zero displacement would mean the resim was a no-op.
    assert c4b.evidence is not None
    assert float(np.median(c4b.evidence["e"])) > 0.0
    assert report.admitted
    assert state.names == ("X", "Y")


@pytest.mark.slow
def test_full_chain_builds_and_compiles_to_ssm_artifact():
    import polars as pl

    from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
    from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model

    structural_plan = _structural_plan()
    contributions = {
        "X": _contrib_X(),
        "Y": _contrib_child("Y", "y1", "X"),
        "Z": _contrib_child("Z", "z1", "Y"),
    }
    accepted = dict.fromkeys(contributions, _ALL_SOFT)
    state = AdmissionState()
    reports = []
    for name in build_construct_order(structural_plan):
        state, report = admit_construct(
            state,
            contributions[name],
            structural_plan,
            _design(),
            accepted[name],
        )
        reports.append(report)
        assert report.admitted
    assert [r.name for r in reports] == ["X", "Y", "Z"]
    assert state.names == ("X", "Y", "Z")

    # The accumulated StatisticalModelSpec + priors compile to the real compiled_ssm artifact
    # the stage produces, and build a live, fittable 3-latent structure.
    compiled = compile_ssm_artifact(
        state.statistical_model_spec(structural_plan),
        state.prior_plan(structural_plan),
        structural_plan,
    )
    assert compiled.spec is not None
    assert compiled.schema_version == 2
    wide = pl.DataFrame(
        {
            "time": list(range(10)),
            "x1": [0.1] * 10,
            "x2": [0.2] * 10,
            "y1": [0.3] * 10,
            "z1": [0.4] * 10,
        }
    )
    model = hydrate_compiled_model(compiled, wide)
    assert model.spec.n_latent == 3


def test_fixed_hill_coefficients_participate_in_admission_and_edge_off(monkeypatch):
    """Fixed coefficients still affect the Hill checks and the exact edge-off contrast."""
    from nof1_causal_lab.models.ssm.dynamics.spec import (
        DynamicsSpec,
        HillEdgeSpec,
        NodePotentialSpec,
    )
    from nof1_causal_lab.models.ssm.predictive import registry_runtime
    from nof1_causal_lab.models.ssm.structure.parameters import Fixed, Free
    from tests.ssm_spec_fixtures import block_ssm_spec

    draws, ticks = 4, 21
    hill = HillEdgeSpec(source=0, target=1, emax=Fixed(0.8), ec50=Fixed(1), n=Fixed(2))
    spec = block_ssm_spec(
        n_latent=2,
        n_manifest=1,
        latent_names=["X", "Y"],
        manifest_names=["y1"],
        manifest_links=[LinkFunction.IDENTITY],
        dynamics_spec=DynamicsSpec(
            2,
            (
                NodePotentialSpec(
                    target=1, center=Fixed(0), stiffness=Fixed(0.5), quartic=Fixed(0)
                ),
                hill,
            ),
        ),
    )
    latents = np.broadcast_to(np.linspace(0.2, 2, ticks)[None, :, None], (draws, ticks, 2)).copy()
    predictive = {
        "latents": latents,
        "observations": latents[:, :, 1:],
        "expected_observations": latents[:, :, 1:],
        "manifest_cov": np.broadcast_to(np.array([[[0.25]]]), (draws, 1, 1)),
    }
    captured = []

    def exact_resimulation(intervened_spec, samples, _times, **_kwargs):
        component = intervened_spec.dynamics_spec.components[1]
        assert isinstance(component, HillEdgeSpec)
        assert isinstance(component.emax, Free)
        assert component.ec50 == hill.ec50
        assert component.n == hill.n
        np.testing.assert_array_equal(samples["vf_1_Emax"], np.zeros(draws))
        captured.append(intervened_spec)
        return jnp.asarray(latents * 0.8), jnp.asarray(latents[:, :, 1:])

    monkeypatch.setattr(
        registry_runtime, "_simulate_vector_field_predictive_latents", exact_resimulation
    )
    indicator_id = fixture_entity_id("indicator", "y1")
    design = DesignInfo(
        manifest_ids=(indicator_id,),
        t_grid=jnp.arange(ticks, dtype=float),
        obs_index_by_indicator={indicator_id: np.arange(ticks)},
        values_by_indicator={indicator_id: np.linspace(0.2, 2, ticks)},
        n_draws=draws,
    )
    results, _ = _run_battery(
        spec,
        predictive,
        design,
        ConstructContribution(
            name="Y",
            likelihoods=(_lik("y1"),),
            edge_parents=("X",),
            hill_parents=("X",),
        ),
    )
    assert len(captured) == 1
    assert spec.dynamics_spec.components[1] is hill
    checks = {result.check: result for result in results}
    assert checks["C3 resolvability"].passed
    assert checks["C4c saturation"].passed
    evidence = checks["C4c saturation"].evidence
    assert evidence is not None
    np.testing.assert_array_equal(evidence["hill_n"], np.full(draws, 2))
