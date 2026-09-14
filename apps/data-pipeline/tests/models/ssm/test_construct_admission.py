"""End-to-end tests for the gradual construct-admission engine."""

from __future__ import annotations

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import model_for_constructs
from nof1_causal_lab.models.ssm import numerics as numeric
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
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.reachability import CheckResult
from nof1_causal_lab.utils.model_structure import (
    get_edges,
    get_known_inputs,
    get_manifest_indicators,
    get_state_names,
)
from tests.dynamics_fixtures import decay_term, hill_term, potential_term
from tests.helpers import (
    complete_test_model,
    fixture_entity_id,
    graph_constructs,
    make_model,
)
from tests.model_fixtures import model_fixture
from tests.models.ssm.test_dag_to_ssm import _model_payload

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
_TARGETS = {
    "X",
    "Y",
    "Z",
    "X->Y",
    "Y->Z",
    *(fixture_entity_id("indicator", n) for n in ("x1", "x2", "y1", "z1")),
}
_ALL_SOFT = {(check, target): "t" for check in _SOFT_CHECKS for target in _TARGETS}


def _structure() -> ModelSpec:
    return ModelSpec.model_validate(_model_payload())


def _contribution(name: str, priors: dict[str, dist.Distribution], parent: str | None = None):
    model = complete_test_model(_structure())
    construct = next(item for item in model.constructs if item.name == name)
    edges = tuple(edge for edge in model.edges if edge.effect.id == construct.id)
    owned_ids = {
        construct.id,
        *(item.id for item in construct.indicators),
        *(term.id for term in construct.dynamics),
        *(edge.id for edge in edges),
        *(term.id for edge in edges for term in edge.mechanisms),
    }
    parameters = tuple(
        item.model_copy(update={"distribution": priors[item.name]}) if item.name in priors else item
        for item in model.parameters
        if {owner.id for owner in model.parameter_context(item.id).owners} <= owned_ids
        or any(
            owner.kind == "mechanism" and owner.id in owned_ids
            for owner in model.parameter_context(item.id).owners
        )
    )
    return ConstructContribution(
        construct=construct,
        edges=edges,
        parameters=parameters,
        edge_parents=(parent,) if parent else (),
    )


def _contrib_X() -> ConstructContribution:
    return _contribution(
        "X",
        {
            "rho_X": dist.Normal(0.6, 0.1),
            "sigma_X": dist.HalfNormal(0.5),
            "lambda_x2_X": dist.Normal(1.0, 0.2),
            "obs_sd_x1": dist.HalfNormal(0.3),
            "obs_sd_x2": dist.HalfNormal(0.3),
        },
    )


def _contrib_child(name: str, indicator: str, parent: str) -> ConstructContribution:
    return _contribution(
        name,
        {
            f"rho_{name}": dist.Normal(0.6, 0.1),
            f"sigma_{name}": dist.HalfNormal(0.5),
            f"beta_{parent}_{name}": dist.Normal(0.3, 0.1),
        },
        parent,
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
    order = build_construct_order(_structure())
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
        AdmissionState(model=_structure()),
        ConstructContribution(construct=_structure().constructs[0]),
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
    spec = model_fixture(
        n_latent=1,
        dynamics_spec=DynamicsSpec(1, ()),
        latent_names=["static"],
        manifest_names=["static_indicator"],
        time_invariant_mask=np.array([True]),
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
        construct=complete_test_model(make_model(["static"]))
        .constructs[0]
        .model_copy(
            update={
                "indicators": (
                    complete_test_model(make_model(["static"]))
                    .indicators[0]
                    .model_copy(
                        update={
                            "id": fixture_entity_id("indicator", "static_indicator"),
                            "name": "static_indicator",
                        }
                    ),
                )
            }
        ),
    )

    results, _timings = _run_battery(spec, pred, design, target)
    checks = {result.check for result in results}
    assert {"C5a location reach", "C5b width"} <= checks
    assert "C5c transmission" not in checks


def test_build_construct_order_covers_only_estimation_states():
    """Constructs marginalized/anchored/dropped out of the estimation
    projection carry no state — nothing to admit for them."""
    causal_design = _model_payload()
    causal_design["edges"].append(
        {
            "id": "edge:unobserved-y",
            "cause": {
                "id": "construct:77b5a7df0a9c99cf2e02",
                "name": "M",
                "description": "Unobserved cause",
                "role": "exogenous",
                "temporal_status": "time_varying",
            },
            "effect": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
            "description": "M causes Y",
        }
    )
    order = build_construct_order(ModelSpec.model_validate(causal_design))
    assert order == ["X", "Y", "Z"]


def test_build_construct_order_admits_lagged_feedback_cycles():
    """Lagged feedback loops sort as a unit: cycle members adjacent, parents first."""
    causal_design = _model_payload()
    feedback = {
        "cause": {"kind": "construct", "id": "construct:a6b7873d58dac1ff1a02"},
        "effect": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
        "id": "edge:e122f01ff2b4e016111d",
        "description": "Z feeds back on Y",
        "lagged": True,
    }
    causal_design["edges"].append(dict(feedback))
    order = build_construct_order(ModelSpec.model_validate(causal_design))
    assert order == ["X", "Y", "Z"]


def test_model_for_constructs_to_subset():
    restricted = model_for_constructs(_structure(), {"X", "Y"})
    assert get_state_names(restricted) == ["X", "Y"]
    assert {indicator["name"] for indicator in get_manifest_indicators(restricted)} == {
        "x1",
        "x2",
        "y1",
    }
    assert all(edge["effect"] != "Z" for edge in get_edges(restricted))


def test_model_for_constructs_preserves_known_input_dependency():
    causal_design = _model_payload()
    graph_constructs(causal_design)[0]["usage"] = {
        "kind": "known_input",
        "source_indicator_id": "indicator:0f93ce57e1f1d1c96f5c",
        "scale": 10.0,
        "missing_policy": "forward_fill",
    }

    plan = ModelSpec.model_validate(causal_design)
    restricted = model_for_constructs(plan, {"Y"})

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
        dynamics,
    ):
        del transition_inputs, rng_key
        captured["input_effect"] = np.asarray(samples["input_effect"])
        return jnp.zeros((2, 3, 2)), jnp.zeros((2, 3, 1))

    monkeypatch.setattr(
        registry_runtime,
        "_simulate_vector_field_predictive_latents",
        _capture_samples,
    )
    from dataclasses import replace

    from tests.model_fixtures import default_input_effect_block

    spec = model_fixture(
        n_latent=2,
        dynamics_spec=DynamicsSpec(2, (*(decay_term(target=i) for i in range(2)),)),
        latent_names=["mood", "sleep"],
        input_names=["dose", "exercise"],
        input_effect_block=replace(
            default_input_effect_block(2),
            n_cols=2,
            free_support=np.ones((2, 2), dtype=bool),
            template=jnp.zeros((2, 2)),
        ),
    )
    contribution = ConstructContribution(
        construct=make_model(["mood"]).constructs[0], edge_parents=("dose",)
    )
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
    _structure()
    state, report = admit_construct(
        AdmissionState(model=_structure()),
        _contrib_X(),
        _design(),
        accepted=_ALL_SOFT,
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
    _structure()
    design = _design()
    state, _ = admit_construct(
        AdmissionState(model=_structure()),
        _contrib_X(),
        design,
        accepted=_ALL_SOFT,
    )
    state, report = admit_construct(
        state, _contrib_child("Y", "y1", "X"), design, accepted=_ALL_SOFT
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
def test_full_chain_builds_an_executable_model():
    import polars as pl

    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.models.ssm.runtime import build_ssm_model

    model = _structure()
    contributions = {
        "X": _contrib_X(),
        "Y": _contrib_child("Y", "y1", "X"),
        "Z": _contrib_child("Z", "z1", "Y"),
    }
    accepted = dict.fromkeys(contributions, _ALL_SOFT)
    state = AdmissionState(model=_structure())
    reports = []
    for name in build_construct_order(model):
        state, report = admit_construct(
            state,
            contributions[name],
            _design(),
            accepted[name],
        )
        reports.append(report)
        assert report.admitted
    assert [r.name for r in reports] == ["X", "Y", "Z"]
    assert state.names == ("X", "Y", "Z")

    # The completed scientific ModelSpec supports execution and a live 3-latent structure.
    compiled = check_execution(
        state.completed_model(),
    )
    assert state.completed_model() is not None
    assert len(compiled) == 3
    wide = pl.DataFrame(
        {
            "time": list(range(10)),
            "x1": [0.1] * 10,
            "x2": [0.2] * 10,
            "y1": [0.3] * 10,
            "z1": [0.4] * 10,
        }
    )
    model = build_ssm_model(wide, model_spec=state.completed_model())
    assert numeric.n_states(model.spec) == 3


def test_fixed_hill_coefficients_participate_in_admission_and_edge_off(monkeypatch):
    """Fixed coefficients still affect the Hill checks and the exact edge-off contrast."""
    from nof1_causal_lab.artifacts.expressions import LiteralExpression, hill_applications
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.predictive import registry_runtime
    
    from tests.model_fixtures import model_fixture

    draws, ticks = 4, 21
    hill = hill_term(source=0, target=1, emax=FixedCoefficient(value=0.8), ec50=FixedCoefficient(value=1), n=FixedCoefficient(value=2))
    spec = model_fixture(
        n_latent=2,
        n_manifest=2,
        latent_names=["X", "Y"],
        manifest_names=["x1", "y1"],
        manifest_links=[LinkFunction.IDENTITY] * 2,
        dynamics_spec=DynamicsSpec(
            2,
            (
                potential_term(target=1, center=FixedCoefficient(value=0), stiffness=FixedCoefficient(value=0.5), quartic=FixedCoefficient(value=0)),
                hill,
            ),
        ),
    )
    latents = np.broadcast_to(np.linspace(0.2, 2, ticks)[None, :, None], (draws, ticks, 2)).copy()
    predictive = {
        "latents": latents,
        "observations": latents,
        "expected_observations": latents,
        "manifest_cov": np.broadcast_to(np.eye(2) * 0.25, (draws, 2, 2)),
    }
    captured = []
    original = numeric.dynamics_expressions(spec)[2]

    def exact_resimulation(intervened_spec, samples, _times, **_kwargs):
        component = _kwargs["dynamics"].components[2]
        assert component.expression == LiteralExpression(value=0)
        for name, values in predictive.items():
            np.testing.assert_array_equal(samples[name], jnp.asarray(values))
        captured.append(intervened_spec)
        return jnp.asarray(latents * 0.8), jnp.asarray(latents[:, :, 1:])

    monkeypatch.setattr(
        registry_runtime, "_simulate_vector_field_predictive_latents", exact_resimulation
    )
    indicator_id = fixture_entity_id("indicator", "y1")
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "x1"), indicator_id),
        t_grid=jnp.arange(ticks, dtype=float),
        obs_index_by_indicator={
            identity: np.arange(ticks) for identity in numeric.observation_ids(spec)
        },
        values_by_indicator={
            identity: np.linspace(0.2, 2, ticks) for identity in numeric.observation_ids(spec)
        },
        n_draws=draws,
    )
    results, _ = _run_battery(
        spec,
        predictive,
        design,
        ConstructContribution(
            construct=_contrib_child("Y", "y1", "X").construct,
            edge_parents=("X",),
            hill_parents=("X",),
        ),
    )
    assert len(captured) == 1
    assert numeric.dynamics_expressions(spec)[2] == original
    assert len(tuple(hill_applications(original.expression))) == 1
    checks = {result.check: result for result in results}
    assert checks["C3 resolvability"].passed
    assert checks["C4c saturation"].passed
    evidence = checks["C4c saturation"].evidence
    assert evidence is not None
    np.testing.assert_array_equal(evidence["hill_n"], np.full(draws, 2))
