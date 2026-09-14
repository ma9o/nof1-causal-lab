"""Unit tests for the gradual construct-admission model-spec flow (data-free pieces).

The Temporal workflow owns the live-data orchestration; here we pin the pure
payload → contribution mapping, feedback rendering, prompt assembly, and the
out-of-order submission guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import numpyro.distributions as dist
import polars as pl
from notebooks.prior_specification_support import parameter_with_prior

from nof1_causal_lab.artifacts.construct import replace_constructs, serialize_edge_references
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
    SUBMIT_CONSTRUCT_SCHEMA,
    ConstructBuildState,
    ParamCatalog,
    _admission_report_payload,
    _closed_loop_target,
    _closing_edge_effects,
    construct_parents,
    contribution_from_payload,
    render_admission_feedback,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_prompt import (
    build_construct_messages,
)
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    AdmissionTiming,
    ConstructAdmissionReport,
    ConstructContribution,
    _signal_from_linear_predictor,
)
from nof1_causal_lab.models.ssm.reachability import CheckResult
from tests.dynamics_fixtures import decay_term
from tests.helpers import complete_test_model, fixture_entity_id, graph_constructs
from tests.models.ssm.test_dag_to_ssm import _model_payload


def _normal(mu: float, sigma: float) -> dict[str, Any]:
    return {"distribution": "Normal", "params": {"mu": mu, "sigma": sigma}}


def _parameter_with_prior(parameter, payload, plan=None):
    catalog = ParamCatalog.from_model(plan or _typed_structure())
    metadata = catalog.metadata_for(parameter)
    return parameter_with_prior(
        ParameterSpec.model_validate(
            {key: value for key, value in metadata.items() if key in ParameterSpec.model_fields}
        ),
        payload,
    )


def _model_definition():
    return _model_payload()


def _typed_structure(payload=None):
    return ModelSpec.model_validate(payload or _model_definition())


def _construct(payload, name):
    return next(item for item in graph_constructs(payload) if item["name"] == name)


def _indicator(payload, name):
    return next(
        indicator
        for construct in graph_constructs(payload)
        for indicator in construct["indicators"]
        if indicator["name"] == name
    )


def _add_feedback_edge(payload, *, cause="Z", effect="Y"):
    _construct(payload, effect)["role"] = "endogenous"
    payload["edges"].append(
        {
            "id": fixture_entity_id("edge", cause + "->" + effect),
            "cause": {"kind": "construct", "id": _construct(payload, cause)["id"]},
            "effect": {"kind": "construct", "id": _construct(payload, effect)["id"]},
            "description": cause + " feeds back on " + effect,
            "lagged": True,
        }
    )


def _completed(plan, *, hill_pairs=(), self_limiting=()):
    model = plan
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(
                    update={
                        "indicators": tuple(
                            i.model_copy(
                                update={
                                    "likelihood": LikelihoodSpec(
                                        law=observation_law(
                                            c.id, "ordered_logistic", "cumulative_logit"
                                        ),
                                        reasoning="Ordered observations",
                                        standardized=False,
                                    )
                                }
                            )
                            if i.measurement_dtype == "ordinal"
                            else i
                            for i in c.indicators
                        )
                    }
                )
                for c in model.constructs
            ),
        )
    )
    hill_ids = tuple(
        e.id
        for e in model.edges
        if (model.get_construct(e.cause.id).name, model.get_construct(e.effect.id).name)
        in hill_pairs
    )
    return complete_test_model(model, hill_edges=hill_ids, self_limiting=self_limiting)


def _payload(plan, name, *, edge_pairs=None, hill_pairs=(), self_limiting=False):
    target_id = next(c.id for c in plan.constructs if c.name == name)
    model = _completed(
        plan, hill_pairs=hill_pairs, self_limiting=(target_id,) if self_limiting else ()
    )
    construct = model.get_construct(target_id)
    edges = tuple(
        e
        for e in model.edges
        if (
            e.effect.id == target_id
            if edge_pairs is None
            else (model.get_construct(e.cause.id).name, model.get_construct(e.effect.id).name)
            in edge_pairs
        )
    )
    term_ids = {t.id for t in construct.dynamics} | {t.id for e in edges for t in e.mechanisms}
    local_ids = {construct.id, *(i.id for i in construct.indicators)}
    parameters = tuple(
        p
        for p in model.parameters
        if (
            any(o.id in term_ids for o in model.parameter_context(p.id).owners)
            or {o.id for o in model.parameter_context(p.id).owners} <= local_ids
        )
    )
    return {
        "construct": construct.model_dump(mode="json"),
        "edges": serialize_edge_references(edges),
        "parameters": [p.model_dump(mode="json") for p in parameters],
    }


def test_param_catalog_reflects_compiler_free_params():
    catalog = ParamCatalog.from_model(_typed_structure())
    # X has two indicators → both measurement-noise + the free loading are authorable.
    assert "obs_sd_x1" in catalog.by_construct["X"]
    assert "lambda_x2_X" in catalog.by_construct["X"]
    assert catalog.role_for("lambda_x2_X") == (
        ParameterRole.LOADING,
        ParameterConstraint.POSITIVE,
    )
    # Single-indicator Y: its measurement noise is NOT a free parameter (absorbed
    # into process noise); authoring obs_sd_y1 must be rejected, but the edge is free.
    y_allowed = set(catalog.by_construct["Y"]) | catalog.structural_names("Y", ["X"])
    assert "obs_sd_y1" not in y_allowed
    assert "beta_X_Y" in y_allowed
    assert catalog.role_for("beta_X_Y") == (ParameterRole.FIXED_EFFECT, ParameterConstraint.NONE)
    # Structural extensions (self-limiting quartic, Hill edge) are always offerable.
    assert "self_limit_Y" in y_allowed
    assert "hill_emax_X_Y" in y_allowed
    assert catalog.role_for("self_limit_Y")[0] == ParameterRole.DYNAMICS_PARAMETER_POSITIVE
    # Policy-pinned surfaces the admission-time compile never frees (STATIONARY
    # initialization; no equilibrium forcing) are NOT offered — surfacing them is
    # what made the agent author priors the compiler then rejects as "not free".
    # X is a dynamic construct (has rho_X/sigma_X), so its initial state and
    # well-centre are pinned.
    assert "cint_X" not in set(catalog.by_construct["X"])
    assert "t0_mean_X" not in catalog.by_construct["X"]
    assert "t0_sd_X" not in catalog.by_construct["X"]


def test_param_catalog_surfaces_static_mean_when_standardization_can_activate_it():
    model = _model_definition()
    _construct(model, "X")["temporal_status"] = "time_invariant"
    catalog = ParamCatalog.from_model(_typed_structure(model))
    assert "t0_mean_X" in catalog.by_construct["X"]


def test_submit_construct_rejects_parameter_from_another_contribution():
    plan = _typed_structure()
    state = ConstructBuildState(plan, pl.DataFrame(), ["X", "Y", "Z"])
    payload = _payload(plan, "X")
    foreign = _parameter_with_prior(
        "sigma_Z", {"distribution": "HalfNormal", "params": {"sigma": 1.0}}, plan
    )
    payload["parameters"].append(foreign.model_dump(mode="json"))
    feedback = state.submit_construct(**payload)
    assert "not referenced" in feedback
    assert foreign.id in feedback
    assert state.current_construct == "X"  # not admitted


def test_submit_construct_rejects_intercept_inactive_for_locked_likelihood():
    definition = _model_definition()
    _indicator(definition, "x1").update(
        measurement_dtype="ordinal", aggregation="last", ordinal_levels=["low", "medium", "high"]
    )
    plan = _typed_structure(definition)
    state = ConstructBuildState(plan, pl.DataFrame(), ["X", "Y", "Z"])
    payload = _payload(plan, "X")
    from nof1_causal_lab.artifacts.identity import scientific_id

    intercept = ParameterSpec(
        id=scientific_id("parameter", "unused intercept"),
        name="unused",
        description="Unused intercept",
        distribution=dist.Normal(0.0, 1.0),
    )
    payload["parameters"].append(intercept.model_dump(mode="json"))
    feedback = state.submit_construct(**payload)
    assert "not referenced" in feedback
    assert intercept.id in feedback
    assert state.current_construct == "X"


def test_construct_parents_reads_the_dag():
    spec = _typed_structure()
    assert construct_parents(spec, "Y") == ["X"]
    assert construct_parents(spec, "Z") == ["Y"]
    assert construct_parents(spec, "X") == []


def test_restricted_compiler_delta_surfaces_feedback_closing_edge():
    """The restricted compiler materializes the closing edge on the second member's turn."""
    spec = _model_definition()
    _add_feedback_edge(spec)
    plan = _typed_structure(spec)
    state = ConstructBuildState(model=plan, data_for_model=pl.DataFrame(), order=["Y", "Z"])
    state.admission = AdmissionState(model=state.model, names=("X",))

    y_inventory = state.parameter_inventory_for("Y")
    assert "beta_X_Y" in y_inventory.compiler_prior_names
    assert "beta_Z_Y" not in y_inventory.compiler_prior_names
    assert not y_inventory.closing_beta_names

    state.admission = AdmissionState(names=("X", "Y"), model=state.model)
    z_inventory = state.parameter_inventory_for("Z")
    assert "beta_Y_Z" in z_inventory.compiler_prior_names
    assert z_inventory.closing_beta_names == frozenset({"beta_Z_Y"})
    assert "hill_emax_Z_Y" in z_inventory.structural_prior_names


def test_prompt_and_submission_exclude_deferred_incoming_feedback_prior():
    payload = _model_definition()
    _add_feedback_edge(payload)
    plan = _typed_structure(payload)
    state = ConstructBuildState(model=plan, data_for_model=pl.DataFrame(), order=["Y", "Z"])
    state.admission = AdmissionState(model=state.model, names=("X",))

    _system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        model=plan,
        validation_report={"indicators": {}},
    )

    assert "`beta_X_Y`" in user
    assert "`beta_Z_Y`" not in user
    assert "deferred feedback parent; its incoming effect is not authorable on this turn" in user

    payload = _payload(plan, "Y", edge_pairs=(("X", "Y"),))
    payload["parameters"].append(
        _parameter_with_prior("beta_Z_Y", _normal(0.2, 0.1), plan).model_dump(mode="json")
    )
    feedback = state.submit_construct(**payload)
    assert "not referenced" in feedback


def test_closing_edge_effects_detects_the_rechecked_member():
    spec = _model_definition()
    _add_feedback_edge(spec)
    plan = _typed_structure(spec)
    # Admitting Z with Y already admitted closes the Y<->Z loop → Y is the member to recheck.
    assert _closing_edge_effects(plan, "Z", {"X", "Y"}) == ["Y"]
    # Admitting Y (only X admitted) closes no loop — Y->Z's effect isn't admitted yet.
    assert _closing_edge_effects(plan, "Y", {"X"}) == []


def test_closed_loop_target_includes_the_closing_feedback_edge():
    payload = _model_definition()
    _add_feedback_edge(payload)
    plan = _typed_structure(payload)
    member = ConstructContribution(
        construct=next(c for c in plan.constructs if c.name == "Y"), edge_parents=("X",)
    )
    target = _closed_loop_target(member, _completed(plan).edges)
    assert target.edge_parents == ("X", "Z")
    assert target.hill_parents == ()
    hill_target = _closed_loop_target(member, _completed(plan, hill_pairs=(("Z", "Y"),)).edges)
    assert hill_target.edge_parents == ("X", "Z")
    assert hill_target.hill_parents == ("Z",)


def test_contribution_from_payload_linear_edge():
    plan = _typed_structure()
    payload = _payload(plan, "Y")
    contribution = contribution_from_payload(plan, payload)
    assert contribution.name == "Y"
    assert [i.id for i in contribution.construct.indicators] == [
        fixture_entity_id("indicator", "y1")
    ]
    likelihood = contribution.construct.indicators[0].likelihood
    assert likelihood is not None
    assert likelihood.law.family == DistributionFamily.GAUSSIAN
    assert {p.name for p in contribution.parameters} == {"rho_Y", "sigma_Y", "beta_X_Y"}
    assert contribution.edge_parents == ("X",)
    assert contribution.hill_parents == ()


def test_contribution_from_payload_hill_edge_and_self_limit():
    plan = _typed_structure()
    contribution = contribution_from_payload(
        plan, _payload(plan, "Y", hill_pairs=(("X", "Y"),), self_limiting=True)
    )
    assert contribution.edge_parents == contribution.hill_parents == ("X",)
    assert (
        "dynamics_potential_quartic"
        if any(p.name == "self_limit_Y" for p in contribution.parameters)
        else None == "dynamics_potential_quartic"
    )
    from nof1_causal_lab.artifacts.expressions import hill_applications

    assert len(tuple(hill_applications(contribution.edges[0].mechanisms[0].expression))) == 1


def test_submit_construct_rejects_out_of_order():
    plan = _typed_structure()
    state = ConstructBuildState(plan, pl.DataFrame(), ["X", "Y", "Z"])
    feedback = state.submit_construct(**_payload(plan, "Y"))
    assert "Out-of-order" in feedback
    assert state.current_construct == "X"
    assert state.submission_made


def test_native_diffusion_site_accepts_different_authored_families():
    from notebooks.prior_specification_support import parameter_with_prior

    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    model = _completed(_typed_structure())
    model = model.revised(
        parameters=tuple(
            parameter_with_prior(
                p,
                {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.5, "sigma": 0.1, "lower": 0.1, "upper": 1.0},
                },
            )
            if p.name == "sigma_X"
            else p
            for p in model.parameters
        )
    )
    priors, _, _, _, _ = compile_ssm_inputs_from_model(model)
    law = priors["diffusion_diag_free"]
    assert law.batch_shape == (3,)
    assert np.isfinite(law.log_prob(np.array([0.5, 0.5, 0.5]))).all()


def test_feedback_closure_hard_recheck_blocks_commit(monkeypatch):
    from nof1_causal_lab.flows.transitions.model_spec.agentic import construct_flow as module

    model = _model_definition()
    _add_feedback_edge(model, cause="Y", effect="X")
    plan = _typed_structure(model)
    initial = AdmissionState(names=("X",), model=plan)
    state = ConstructBuildState(model=plan, data_for_model=pl.DataFrame(), order=["Y"])
    state.admission = initial
    active_pass = CheckResult("C1a finiteness", "Y", "0%", "0%", True, "ok")
    hard_recheck = CheckResult("C1a finiteness", "X", "1%", "0%", False, "bad")
    tentative = AdmissionState(names=("X", "Y"), model=plan)
    admitted_report = ConstructAdmissionReport(
        name="Y",
        results=(active_pass,),
        timings=(),
        outcome="ADMITTED",
        annotations=(),
        admitted=True,
    )
    monkeypatch.setattr(module, "build_design_info", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "admit_construct",
        lambda *_args, **_kwargs: (tentative, admitted_report),
    )
    monkeypatch.setattr(
        ConstructBuildState,
        "_coupled_recheck",
        lambda *_args, **_kwargs: ([hard_recheck], {"results": [], "timings": []}),
    )

    feedback = state.submit_construct(**_payload(plan, "Y", edge_pairs=(("X", "Y"), ("Y", "X"))))

    assert "BLOCKED" in feedback
    assert state.admission is initial
    assert state.current_construct == "Y"


def test_render_admission_feedback_lists_failed_checks():
    report = ConstructAdmissionReport(
        name="Y",
        results=(
            CheckResult("C1a finiteness", "Y", "0%", "0%", True, "ok"),
            CheckResult("C3 resolvability", "Y", "0.05 d", "[0.3, 2.5]", False, "too fast"),
        ),
        timings=(),
        outcome="NEEDS DECISION: C3 resolvability",
        annotations=(),
        admitted=False,
    )
    text = render_admission_feedback(report)
    assert "NEEDS DECISION" in text
    assert "C3 resolvability" in text
    assert "[FAIL]" in text
    assert "[PASS]" in text


def test_admission_report_payload_includes_backend_timing_breakdown():
    report = ConstructAdmissionReport(
        name="X",
        results=(CheckResult("C1a finiteness", "X", "0%", "0%", True, "ok"),),
        timings=(
            AdmissionTiming("model_compilation", "ModelSpec compilation", 12.5),
            AdmissionTiming(
                "c1_confinement",
                "C1 confinement",
                3.25,
                ("C1a finiteness",),
            ),
        ),
        outcome="ADMITTED",
        annotations=(),
        admitted=True,
    )

    payload = _admission_report_payload(
        report,
        ConstructContribution(construct=_typed_structure().constructs[0]),
        attempt=2,
    )

    assert payload["attempt"] == 2
    assert payload["timings"] == [
        {
            "phase": "model_compilation",
            "label": "ModelSpec compilation",
            "duration_ms": 12.5,
            "checks": [],
        },
        {
            "phase": "c1_confinement",
            "label": "C1 confinement",
            "duration_ms": 3.25,
            "checks": ["C1a finiteness"],
        },
    ]


def test_ordered_logistic_signal_uses_sampled_cutpoints():
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from tests.model_fixtures import model_fixture

    linear_predictor = np.array([[-2.0, 0.0, 2.0], [-1.0, 0.5, 1.5]])
    predictive = {
        "obs_ordered_base": np.zeros((2, 1)),
        "obs_ordered_gaps": np.ones((2, 1, 2)),
    }

    signal = _signal_from_linear_predictor(
        LinkFunction.CUMULATIVE_LOGIT,
        linear_predictor,
        spec=model_fixture(
            n_latent=1,
            dynamics_spec=DynamicsSpec(1, (*(decay_term(target=i) for i in range(1)),)),
            manifest_dists=["ordered_logistic"],
            manifest_level_counts=[4],
        ),
        pred=predictive,
        manifest_index=0,
    )

    assert signal.shape == (*linear_predictor.shape, 4)
    assert np.allclose(signal.sum(axis=2), 1.0)
    expected_category = np.sum(signal * np.arange(4), axis=2)
    assert np.all(np.diff(expected_category, axis=1) > 0)


def test_build_construct_messages_surfaces_params_and_feedback():
    spec = _typed_structure()
    state = ConstructBuildState(
        model=spec, data_for_model=pl.DataFrame(), order=["X", "Y", "Z"], cursor=1
    )
    from nof1_causal_lab.models.ssm.construct_admission import trial_admission_state

    state.admission = trial_admission_state(
        state.admission, contribution_from_payload(spec, _payload(spec, "X"))
    )
    system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        model=spec,
        validation_report={"indicators": {}},
    )
    assert "continuous-time latent state-space model" in system
    assert "invoking the registered MCP tool `submit_construct`" in system
    assert "canonical" in system
    assert "Active construct: `Y`" in user
    assert "`rho_Y`" in user  # own-dynamics param offered
    assert "`beta_X_Y`" in user  # parent edge param offered
    assert "Does X drive Y?" in user
    assert "MUST use `TruncatedNormal` to match admitted parameters" not in user

    # On a re-attempt, the last failing report for this construct is injected.
    state.last_report = ConstructAdmissionReport(
        name="Y",
        results=(CheckResult("C2 latent scale", "Y", "8.0", "[0.3, 3]", False, "too wide"),),
        timings=(),
        outcome="NEEDS DECISION: C2 latent scale",
        annotations=(),
        admitted=False,
    )
    _system2, user2 = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        model=spec,
        validation_report={"indicators": {}},
    )
    assert "Latest reachability feedback" in user2
    assert "C2 latent scale" in user2


def test_build_construct_messages_describes_actual_likelihood_coefficients():
    payload = _model_definition()
    _indicator(payload, "x1").update(
        measurement_dtype="ordinal",
        aggregation="last",
        ordinal_levels=["low", "medium", "high"],
    )
    spec = _typed_structure(payload)
    state = ConstructBuildState(model=spec, data_for_model=pl.DataFrame(), order=["X", "Y", "Z"])
    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="How does X behave?",
        model=spec,
        validation_report={"indicators": {}},
    )
    assert "`obs_ordered_base_x1`" in user
    assert "`obs_ordered_gaps_x1`" in user
    assert "`obs_ordered_base`" not in user
    assert "`obs_ordered_gaps`" not in user
    assert "`manifest_mean_x1`" not in user
    assert '"cutpoint_base"' in user
    assert '"owners"' not in user


def test_ordered_threshold_parameters_bind_per_indicator_into_vectorized_sites():
    payload = _model_definition()
    x1 = _indicator(payload, "x1")
    x1.update(
        measurement_dtype="ordinal",
        aggregation="last",
        ordinal_levels=["low", "medium", "high"],
    )
    x2 = _indicator(payload, "x2")
    x2.update(
        measurement_dtype="ordinal",
        aggregation="last",
        ordinal_levels=["none", "mild", "moderate", "high", "severe"],
    )
    catalog = ParamCatalog.from_model(_typed_structure(payload))

    from nof1_causal_lab.models.ssm.compile.prior_indexing import build_semantic_prior_bindings

    bindings = build_semantic_prior_bindings(catalog.model).by_parameter
    first = bindings[catalog.parameter_ids["obs_ordered_base_x1"]]
    second = bindings[catalog.parameter_ids["obs_ordered_base_x2"]]
    assert first.site_name == second.site_name == "obs_ordered_base"
    assert (first.flat_index, second.flat_index) == (0, 1)
    assert catalog.metadata_for("obs_ordered_gaps_x1")["distribution_transform"] == "identity"
    assert catalog.metadata_for("obs_ordered_gaps_x2")["distribution_transform"] == "identity"


def test_build_construct_messages_keeps_declared_ordinal_support_with_one_observed_level():
    payload = _model_definition()
    x1 = _indicator(payload, "x1")
    x1["measurement_dtype"] = "ordinal"
    x1["aggregation"] = "last"
    x1["ordinal_levels"] = ["low", "medium", "high"]
    spec = _typed_structure(payload)
    state = ConstructBuildState(
        model=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )

    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="Does X drive Y?",
        model=spec,
        validation_report={
            "indicators": {
                fixture_entity_id("indicator", "x1"): {
                    "profile": {
                        "n_obs": 1,
                        "min": 0.0,
                        "max": 0.0,
                    }
                }
            },
        },
    )

    assert "SPARSE LEVEL COVERAGE: only one level is observed" in user
    assert "declared ordinal levels define the likelihood support" in user


def test_build_construct_messages_renders_concern_local_semantic_context():
    payload = _model_definition()
    payload["measurement_clock"] = "6h"
    y1 = _indicator(payload, "y1")
    y1.update(
        {
            "measurement_dtype": "ordinal",
            "aggregation": "last",
            "observation_window": "12h",
            "ordinal_levels": ["none", "mild", "severe"],
        }
    )
    spec = _typed_structure(payload)
    panel = pl.DataFrame(
        {
            "indicator_id": [
                "indicator:dec7b4916899d2109674",
                "indicator:dec7b4916899d2109674",
                "indicator:dec7b4916899d2109674",
                "indicator:0f93ce57e1f1d1c96f5c",
            ],
            "value": [0.0, 0.0, 0.0, 999.0],
            "anchor_time": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 10),
                datetime(2025, 1, 1),
            ],
        }
    )
    state = ConstructBuildState(model=spec, data_for_model=panel, order=["X", "Y", "Z"], cursor=1)
    state.admission = AdmissionState(model=state.model, names=("X",))
    profile = {
        "measurement_dtype": "ordinal",
        "n_obs": 3,
        "mean": 0.0,
        "std": 0.0,
        "variance": 0.0,
        "min": 0.0,
        "q25": 0.0,
        "q50": 0.0,
        "q75": 0.0,
        "max": 0.0,
        "zero_fraction": 1.0,
        "variance_to_mean_ratio": None,
        "is_nonnegative": True,
        "is_unit_interval": True,
        "looks_integer_valued": True,
        "time_coverage_ratio": 0.6,
        "max_gap_ratio": 1.4,
        "dtype_violations": 0,
        "duplicate_pct": 0.0,
        "n_unparseable_timestamps": 0,
        "arithmetic_sequence_detected": False,
    }
    validation_report = {
        "is_valid": False,
        "dataset_issues": [
            {
                "severity": "warning",
                "issue_type": "short_panel",
                "message": "Dataset-level sentinel",
            }
        ],
        "indicators": {
            fixture_entity_id("indicator", "y1"): {
                "profile": profile,
                "validation": {
                    "issues": [
                        {
                            "severity": "error",
                            "issue_type": "no_variance",
                            "message": "Zero variance (constant value = 0.0)",
                        }
                    ],
                    "checks": {"variance": "error"},
                },
            },
            fixture_entity_id("indicator", "x1"): {
                "profile": {"n_obs": 1, "mean": 999.0},
                "validation": {
                    "issues": [
                        {
                            "severity": "warning",
                            "issue_type": "sibling",
                            "message": "SIBLING_SENTINEL",
                        }
                    ],
                    "checks": {},
                },
            },
        },
    }

    _system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        model=spec,
        validation_report=validation_report,
    )

    assert "Validation report status: **INVALID**" in user
    assert "[WARNING] short_panel: Dataset-level sentinel" in user
    assert "Model clock / authored default effect interval: `6h`" in user
    assert "Estimation role: **retained latent state**" in user
    assert "Theoretical role: `endogenous`" in user
    assert "Temporal status: `time_varying`" in user
    assert "dtype=`ordinal`" in user
    assert "aggregation=`last`" in user
    assert "effective window=`12h`" in user
    assert "0=none, 1=mild, 2=severe" in user
    assert "n=3; mean=0; sd=0; variance=0" in user
    assert "zero fraction=100.0%" in user
    assert "arithmetic sequence detected=false" in user
    assert "coverage/minimum-required-span=60.0%" in user
    assert "largest-gap/allowed-threshold=1.4x" in user
    assert "Observed ordinal occupancy: 0=none (3), 1=mild (0), 2=severe (0)" in user
    assert "span=9 days; median gap=4.5 days; maximum gap=8 days" in user
    assert "[ERROR] no_variance: Zero variance (constant value = 0.0)" in user
    assert "SIBLING_SENTINEL" not in user
    assert "mean=999" not in user


def test_build_construct_messages_handles_null_empirical_profile():
    spec = _typed_structure()
    state = ConstructBuildState(
        model=spec,
        data_for_model=pl.DataFrame(),
        order=["X", "Y", "Z"],
    )

    _system, user = build_construct_messages(
        state=state,
        construct="X",
        question="Does X drive Y?",
        model=spec,
        validation_report={
            "indicators": {
                fixture_entity_id("indicator", "x1"): {
                    "profile": None,
                    "validation": {"issues": [], "checks": {}},
                }
            }
        },
    )

    assert "Raw empirical profile: unavailable (no numeric observations)" in user


def test_build_construct_messages_renders_incoming_known_input_without_hill_option():
    payload = _model_definition()
    _construct(payload, "X")["usage"] = {
        "kind": "known_input",
        "source_indicator_id": _indicator(payload, "x1")["id"],
        "scale": 10.0,
        "missing_policy": "forward_fill",
    }
    spec = _typed_structure(payload)
    state = ConstructBuildState(
        model=spec,
        data_for_model=pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:0f93ce57e1f1d1c96f5c",
                    "indicator:0f93ce57e1f1d1c96f5c",
                    "indicator:0f93ce57e1f1d1c96f5c",
                    "indicator:0f93ce57e1f1d1c96f5c",
                ],
                "value": [0.0, 10.0, 10.0, 20.0],
            }
        ),
        order=["Y", "Z"],
    )

    _system, user = build_construct_messages(
        state=state,
        construct="Y",
        question="Does X drive Y?",
        model=spec,
        validation_report={"indicators": {}},
    )

    assert "`X` — **known transition input**, lagged" in user
    assert "source indicator=`x1`" in user
    assert "scale divisor=10" in user
    assert "missing policy=`forward_fill`" in user
    assert "Source data before scaling: n=4; distinct=3; mean=10; sd=7.071" in user
    assert "Compiler input at observed source rows: mean=1; sd=0.7071; range=[0, 2]" in user
    assert "`beta_X_Y`" in user
    assert "Known-input effects are linear-only" in user
    assert "hill_emax_X_Y" not in user


def test_submit_construct_schema_is_well_formed():
    props = SUBMIT_CONSTRUCT_SCHEMA["properties"]
    assert set(SUBMIT_CONSTRUCT_SCHEMA["required"]) == {"construct", "edges", "parameters"}
    assert SUBMIT_CONSTRUCT_SCHEMA["additionalProperties"] is False
    assert props["construct"]["$ref"] == "#/$defs/Construct"
    assert props["parameters"]["items"]["$ref"] == "#/$defs/ParameterSpec"
    construct = SUBMIT_CONSTRUCT_SCHEMA["$defs"]["Construct"]
    assert {"indicators", "dynamics"} <= construct["properties"].keys()
