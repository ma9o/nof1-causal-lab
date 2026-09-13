"""Scientific identities are independent of labels and execution layout."""

from copy import deepcopy
from typing import Any

import pytest

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
from nof1_causal_lab.flows.transitions.inference.subjects import reference_posterior_findings
from nof1_causal_lab.flows.transitions.model_spec.agentic.skeleton import derive_deterministic_spec
from nof1_causal_lab.models.prior_planning import build_default_prior_plan
from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
from nof1_causal_lab.models.structural import build_structural_plan


def _design(*, rename=False, reverse=False, categorical=False):
    names = ["Renamed A", "Renamed B"] if rename else ["A", "B"]
    constructs = [
        {
            "id": f"construct:{key}",
            "name": name,
            "description": name,
            "role": "exogenous",
            "temporal_status": "time_varying",
        }
        for key, name in zip("ab", names, strict=True)
    ]
    indicators: list[dict[str, Any]] = [
        {
            "id": f"indicator:{key}",
            "construct_id": f"construct:{key}",
            "name": name + "_obs",
            "how_to_measure": name,
            "construct_polarity": "positive",
            "measurement_dtype": "continuous",
            "aggregation": "last",
        }
        for key, name in zip("ab", names, strict=True)
    ]
    if categorical:
        for indicator, levels in zip(
            indicators, (["low", "mid", "high"], ["absent", "present"]), strict=True
        ):
            indicator.update(measurement_dtype="ordinal", ordinal_levels=levels)
    if reverse:
        constructs.reverse()
        indicators.reverse()
    return CausalDesign.model_validate(
        {
            "latent": {"constructs": constructs, "edges": [], "default_outcome": None},
            "measurement": {"indicators": indicators, "model_clock": "1d"},
            "known_inputs": [],
            "scientific_only_constructs": [],
        }
    )


def _compile(design):
    plan = build_structural_plan(design)
    skeleton = derive_deterministic_spec(plan)
    from nof1_causal_lab.flows.transitions.model_spec.agentic.parameter_surfaces import (
        parameter_is_active_for_statistical_model_spec,
    )

    likelihoods = [dict(item, standardized=True) for item in skeleton.resolved_likelihoods]
    for item in skeleton.ambiguous_indicators:
        likelihoods.append(
            {
                "indicator_id": item["indicator_id"],
                "variable": item["variable"],
                "distribution": "gaussian",
                "link": "identity",
                "reasoning": "Test",
                "standardized": True,
            }
        )
    by_name: dict[str, Any] = {str(item["variable"]): item for item in likelihoods}
    parameters = [
        p
        for p in skeleton.all_params
        if parameter_is_active_for_statistical_model_spec(
            dict(p),
            by_name,
            initialization_policy="stationary",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        )
    ]
    spec = StatisticalModelSpec.model_validate(
        {"likelihoods": likelihoods, "parameters": parameters, "mechanisms": skeleton.mechanisms}
    )
    return compile_ssm_artifact(spec, build_default_prior_plan(spec), plan), spec, plan


def test_rename_preserves_parameter_and_element_identity():
    before, _, _ = _compile(_design())
    after, _, _ = _compile(_design(rename=True))
    assert {p.id: set(p.elements) for p in before.parameters} == {
        p.id: set(p.elements) for p in after.parameters
    }
    assert {p.name for p in before.parameters} != {p.name for p in after.parameters}


def test_scalar_identity_survives_reordered_execution_axes():
    before, _, _ = _compile(_design())
    after, _, _ = _compile(_design(reverse=True))
    decay = next(p for p in before.parameters if p.quantity.value == "dynamics_decay")
    newer = next(p for p in after.parameters if p.id == decay.id)
    assert newer.elements == decay.elements
    old_binding = next(b for b in before.parameter_bindings if b.parameter_id == decay.id)
    new_binding = next(b for b in after.parameter_bindings if b.parameter_id == decay.id)
    assert old_binding.coordinates != new_binding.coordinates


def test_compiler_rejects_forged_owner():
    _, spec, plan = _compile(_design())
    invalid = deepcopy(spec)
    invalid.parameters[0].owners[0] = invalid.parameters[1].owners[0]
    with pytest.raises(ValueError, match="inconsistent scientific identity"):
        compile_ssm_artifact(invalid, build_default_prior_plan(invalid), plan)


def test_posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate():
    compiled, _, _ = _compile(_design())
    binding = compiled.parameter_bindings[0]
    element, coordinate = next(iter(binding.coordinates.items()))
    row = {
        "parameter": "untrusted display alias",
        "coordinate": coordinate.model_dump(),
        "mean": 1.0,
        "lower": 0.5,
        "upper": 1.5,
        "interval_kind": "hdi",
        "interval_mass": 0.94,
        "sd": 0.2,
        "x_values": [],
        "density": [],
    }
    marginals, _, _ = reference_posterior_findings(compiled, [row], [], None)
    assert marginals[0]["subject"] == {"parameter_id": binding.parameter_id, "element_id": element}
    assert "coordinate" not in marginals[0]
    row["coordinate"] = {"site_name": "unknown", "indices": []}
    with pytest.raises(ValueError, match="unbound runtime coordinate"):
        reference_posterior_findings(compiled, [row], [], None)


def test_ordinal_components_have_label_identity_and_padding_is_explicit():
    compiled, _, _ = _compile(_design(categorical=True))
    gaps = [p for p in compiled.parameters if p.quantity.value == "obs_ordered_gaps"]
    assert len(gaps) == 1
    assert list(gaps[0].elements.values()) == ["A_obs: gap low / mid / high"]
    assert compiled.auxiliary_coordinates


def test_shared_likelihood_parameter_owns_only_active_channels():
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily, ParameterSpec
    from nof1_causal_lab.models.ssm.compile.parameter_identity import declare_parameter
    from nof1_causal_lab.models.ssm.construct_admission import AdmissionState

    _, spec, plan = _compile(_design())
    candidate = ParameterSpec.model_validate(
        declare_parameter(
            {
                "name": "obs_df",
                "quantity": "obs_df",
                "role": "observation_hyperparameter_positive",
                "constraint": "positive",
                "description": "Shared observation degrees of freedom",
                "construct_names": ["A", "B"],
                "indicator_names": ["A_obs", "B_obs"],
            },
            plan,
        )
    )
    likelihoods = list(spec.likelihoods)
    likelihoods[0] = likelihoods[0].model_copy(
        update={"distribution": DistributionFamily.STUDENT_T}
    )
    state = AdmissionState(
        likelihoods=tuple(likelihoods),
        parameters=(*spec.parameters, candidate),
        mechanisms=tuple(spec.mechanisms),
    )
    model = state.statistical_model_spec(plan)
    shared = next(parameter for parameter in model.parameters if parameter.name == "obs_df")
    assert {owner.id for owner in shared.owners} == {"construct:a", "indicator:a"}
    compiled = compile_ssm_artifact(model, build_default_prior_plan(model), plan)
    assert (
        next(parameter.id for parameter in compiled.parameters if parameter.name == "obs_df")
        == shared.id
    )
    likelihoods[1] = likelihoods[1].model_copy(
        update={"distribution": DistributionFamily.STUDENT_T}
    )
    expanded = AdmissionState(
        likelihoods=tuple(likelihoods), parameters=state.parameters, mechanisms=state.mechanisms
    ).statistical_model_spec(plan)
    shared_expanded = next(
        parameter for parameter in expanded.parameters if parameter.name == "obs_df"
    )
    assert shared_expanded.id != shared.id
    assert {owner.id for owner in shared_expanded.owners} == {
        "construct:a",
        "indicator:a",
        "construct:b",
        "indicator:b",
    }
    with pytest.raises(ValueError, match="active likelihood channels"):
        compile_ssm_artifact(
            spec.model_copy(
                update={
                    "likelihoods": list(model.likelihoods),
                    "parameters": [*spec.parameters, candidate],
                }
            ),
            build_default_prior_plan(expanded),
            plan,
        )


def test_implicit_initial_state_priors_receive_definitions_at_compile_time():
    from nof1_causal_lab.artifacts.statistical_model_spec import InitializationPolicy
    from nof1_causal_lab.flows.transitions.model_spec.prior_resolution import (
        resolve_prior_proposals,
    )

    _, spec, plan = _compile(_design())
    free = spec.model_copy(update={"initialization_policy": InitializationPolicy.FREE})
    compiled = compile_ssm_artifact(free, build_default_prior_plan(free), plan)
    initial = [
        parameter
        for parameter in compiled.parameters
        if parameter.quantity.value in {"t0_means", "t0_var_diag"}
    ]
    assert initial
    assert all(parameter.owners and parameter.elements for parameter in initial)
    proposals = resolve_prior_proposals(compiled, authored_priors={})
    assert {parameter.id for parameter in initial} <= {
        proposal["parameter_id"] for proposal in proposals
    }


def test_parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws():
    before, model, plan = _compile(_design())
    priors = build_default_prior_plan(model)
    renamed = deepcopy(model)
    for parameter in renamed.parameters:
        parameter.name = "A display label with no machine meaning"
    after = compile_ssm_artifact(renamed, priors, plan)
    assert before.spec == after.spec
    assert before.compiled_prior_semantics == after.compiled_prior_semantics
    assert before.parameter_bindings == after.parameter_bindings
    assert {parameter.id for parameter in before.parameters} == {
        parameter.id for parameter in after.parameters
    }


def test_hill_and_fixed_coefficients_survive_parameter_renaming():
    from nof1_causal_lab.artifacts.mechanism import (
        EstimatedCoefficient,
        FixedCoefficient,
        HillEdgeMechanism,
        LinearEdgeMechanism,
    )
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec
    from nof1_causal_lab.machine.equations import state_equations
    from nof1_causal_lab.models.ssm.compile.parameter_identity import declare_parameter

    design = _design().model_dump(mode="json")
    design["latent"]["constructs"][1]["role"] = "endogenous"
    design["latent"]["edges"] = [
        {
            "id": "edge:a_b",
            "cause_id": "construct:a",
            "effect_id": "construct:b",
            "description": "A changes B",
            "lagged": True,
            "sources": [],
        }
    ]
    _, model, plan = _compile(CausalDesign.model_validate(design))
    linear = next(
        mechanism for mechanism in model.mechanisms if isinstance(mechanism, LinearEdgeMechanism)
    )
    definitions = [
        ParameterSpec.model_validate(
            declare_parameter(
                {
                    "name": label,
                    "quantity": quantity,
                    "cause": "A",
                    "effect": "B",
                    "role": "dynamics_parameter_positive",
                    "constraint": "positive",
                    "description": label,
                },
                plan,
            )
        )
        for label, quantity in [("Peak response", "hill_emax"), ("Half-saturation", "hill_ec50")]
    ]
    hill = HillEdgeMechanism(
        edge_id=linear.edge_id,
        emax=EstimatedCoefficient(parameter_id=definitions[0].id),
        ec50=EstimatedCoefficient(parameter_id=definitions[1].id),
        n=FixedCoefficient(value=2),
    )
    model = model.model_copy(
        update={
            "mechanisms": [
                hill if mechanism == linear else mechanism for mechanism in model.mechanisms
            ],
            "parameters": [
                parameter
                for parameter in model.parameters
                if parameter.id != linear.weight.parameter_id
            ]
            + definitions,
        }
    )
    priors = build_default_prior_plan(model)
    before = compile_ssm_artifact(model, priors, plan)
    renamed = deepcopy(model)
    for parameter in renamed.parameters:
        parameter.name = "hill_emax_misleading_name"
    after = compile_ssm_artifact(renamed, priors, plan)
    assert before.spec == after.spec
    assert before.compiled_prior_semantics == after.compiled_prior_semantics
    assert before.parameter_bindings == after.parameter_bindings
    native_hill = before.spec.model_dump(mode="json")["dynamics_spec"]["components"][-1]
    assert native_hill["kind"] == "HillEdge"
    assert native_hill["parameters"]["n"] == {"kind": "fixed", "value": 2.0}
    assert all(parameter.quantity.value != "hill_n" for parameter in before.parameters)
    equation = next(
        row for row in state_equations(model, plan) if row.construct_id == "construct:b"
    )
    assert "Peak response" in equation.latex
    assert r"\max" in equation.latex
    assert "^{2}" in equation.latex
    assert "t-1" not in equation.latex


def test_known_input_mechanism_cannot_silently_double_its_effect():
    payload = _design().model_dump(mode="json")
    payload["latent"]["constructs"][1]["role"] = "endogenous"
    payload["latent"]["edges"] = [
        {
            "id": "edge:ab",
            "cause_id": "construct:a",
            "effect_id": "construct:b",
            "description": "A drives B",
            "lagged": True,
        }
    ]
    payload["known_inputs"] = [
        {
            "construct_id": "construct:a",
            "source_indicator_id": "indicator:a",
            "scale": 1,
            "missing_policy": "forward_fill",
        }
    ]
    _, model, plan = _compile(CausalDesign.model_validate(payload))
    edge = next(mechanism for mechanism in model.mechanisms if mechanism.kind == "linear")
    duplicated = model.model_copy(update={"mechanisms": [*model.mechanisms, edge]})
    with pytest.raises(
        ValueError, match="One parameter cannot own multiple independent runtime sites"
    ):
        compile_ssm_artifact(duplicated, build_default_prior_plan(duplicated), plan)
