"""Test-owned ModelSpec construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import Expression, coefficient, restoring_force, state
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.identity import DistributionId, ParameterId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.numpyro_json import NumPyroDistribution


def load_model_fixture(name: str) -> ModelSpec:
    """Parse one current-format canonical model without study converters."""
    return ModelSpec.model_validate_json(
        (Path(__file__).parent / "fixtures/models" / name).read_text()
    ).materialized()


def construct_named(model: ModelSpec, name: str) -> ConstructSpec:
    return next(node for node in model.constructs if node.name == name)


def indicator_named(model: ModelSpec, name: str) -> IndicatorSpec:
    return next(
        indicator for _, indicator in model.iter_indicators() if indicator.observation.name == name
    )


def likelihood_named(model: ModelSpec, name: str) -> LikelihoodSpec:
    return next(
        likelihood
        for indicator, likelihood in model.iter_likelihoods()
        if indicator.observation.name == name
    )


def parameter_named(model: ModelSpec, name: str) -> ParameterSpec:
    return next(parameter for parameter in model.parameters if parameter.name == name)


def parameter_for(model: ModelSpec, quantity: SiteKind, *owners: str) -> ParameterSpec:
    names: dict[str, str] = {node.id: node.name for node in model.constructs} | {
        indicator.observation.id: indicator.observation.name
        for _, indicator in model.iter_indicators()
    }
    (parameter,) = tuple(
        p
        for p in model.parameters
        if model.parameter_context(p.id).quantity == quantity
        and {names[ref.id] for ref in model.parameter_context(p.id).owners if ref.id in names}
        == set(owners)
    )
    return parameter


def replace_parameters(
    parameters: tuple[ParameterSpec, ...], *revisions: ParameterSpec
) -> tuple[ParameterSpec, ...]:
    updated = {parameter.id: parameter for parameter in revisions}
    return tuple(updated.get(parameter.id, parameter) for parameter in parameters)


def without_parameters(
    model: ModelSpec, *removed: ParameterSpec
) -> tuple[tuple[ParameterSpec, ...], Mapping[DistributionId, NumPyroDistribution]]:
    ids = {parameter.id for parameter in removed}
    retained = tuple(parameter for parameter in model.parameters if parameter.id not in ids)
    used = {parameter.distribution for parameter in retained} | {
        node.distribution for node in model.constructs
    }
    return retained, {
        identity: law for identity, law in model.distributions.items() if identity in used
    }


def parameter_laws(
    model: ModelSpec, updates: Mapping[ParameterId, NumPyroDistribution]
) -> Mapping[DistributionId, NumPyroDistribution]:
    replacements = {}
    for identity, law in updates.items():
        distribution = model.parameter(identity).distribution
        assert distribution is not None, "Fixture parameter must own a law"
        replacements[distribution] = law
    return {**model.distributions, **replacements}


def x_model() -> ModelSpec:
    return load_model_fixture("common/x_model.json")


def x_y_model() -> ModelSpec:
    return load_model_fixture("common/x_y_model.json")


def stress_sleep_model() -> ModelSpec:
    return load_model_fixture("common/stress_sleep_model.json")


def stress_sleep_causal_model() -> ModelSpec:
    """The two measured states with Sleep as the question's downstream outcome."""
    from nof1_causal_lab.artifacts.identity import MechanismId

    model = stress_sleep_model()
    stress = construct_named(model, "Stress")
    return model.with_entities(
        edges=(
            model.edges[0].revised(
                effect=construct_named(model, "Sleep"),
                mechanisms=(
                    DriftMechanismSpec(
                        id=MechanismId("mechanism:stress_to_sleep"),
                        expression=coefficient(0.1, role="weight") * state(stress.id),
                    ),
                ),
            ),
        )
    )


def one_state_gaussian_model() -> ModelSpec:
    return load_model_fixture("common/one_state_gaussian_model.json")


def two_state_gaussian_model() -> ModelSpec:
    return load_model_fixture("common/two_state_gaussian_model.json")


def additive_a_b_model() -> ModelSpec:
    return load_model_fixture("common/additive_a_b_model.json")


def three_state_gaussian_model() -> ModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testdynamicsmask_test_dynamics_support_zeros_non_edges__make_3latent_spec.json"
    )


def fixed_hill_model() -> ModelSpec:
    return load_model_fixture("simulation_checks/fixed_hill_model.json")


def mixed_family_model() -> ModelSpec:
    return load_model_fixture("observation_support/mixed_family_model.json")


def _make_lgss_data_model_fixture() -> ModelSpec:
    model = load_model_fixture("common/one_state_gaussian_model.json")
    latent_0 = construct_named(model, "latent_0")
    latent_0_diffusion_diag = parameter_for(model, SiteKind.DIFFUSION_DIAG, "latent_0")
    latent_0_t0_means = parameter_for(model, SiteKind.T0_MEANS, "latent_0")
    latent_0_t0_var_diag = parameter_for(model, SiteKind.T0_VAR_DIAG, "latent_0")
    latent_0_revised = latent_0.revised(
        coefficients=(
            coefficient(latent_0_diffusion_diag.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
        )
    )
    parameters, distributions = without_parameters(model, latent_0_t0_means, latent_0_t0_var_diag)
    return model.with_entities(
        edges=replace_constructs(model.edges, (latent_0_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _exact_model_model() -> ModelSpec:
    model = load_model_fixture(
        "delta_observations/authored_affine_delta_keeps_its_calibration_coefficients_complete_model.json"
    )
    setting = construct_named(model, "setting")
    setting_obs = indicator_named(model, "setting_obs")
    setting_obs_likelihood = likelihood_named(model, "setting_obs")
    manifest_mean_setting_obs = parameter_named(model, "manifest_mean_setting_obs")
    setting_obs_revised = setting_obs.revised(
        likelihood=setting_obs_likelihood.revised(
            law=DeltaLawSpec[Expression](v=state(setting.id)),
            reasoning="The recorded setting is exact at its observation anchor.",
        )
    )
    setting_revised = setting.revised(indicators=(setting_obs_revised,))
    parameters, distributions = without_parameters(model, manifest_mean_setting_obs)
    return model.with_entities(
        edges=replace_constructs(model.edges, (setting_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _two_state_fixed_drift_model() -> ModelSpec:
    model = load_model_fixture(
        "dynamics_config/scientific_model_roundtrip_preserves_derived_dynamics_model_fixture.json"
    )
    latent_0 = construct_named(model, "latent_0")
    (latent_0_potential,) = latent_0.dynamics
    latent_0_dynamics_decay = parameter_for(model, SiteKind.DYNAMICS_DECAY, "latent_0")
    latent_0_latent_1_hill_emax = parameter_for(model, SiteKind.HILL_EMAX, "latent_0", "latent_1")
    latent_0_latent_1_hill_n = parameter_for(model, SiteKind.HILL_N, "latent_0", "latent_1")
    latent_0_latent_1_hill_ec50 = parameter_for(model, SiteKind.HILL_EC50, "latent_0", "latent_1")
    latent_0_revised = latent_0.revised(
        dynamics=(
            DriftMechanismSpec(
                id=latent_0_potential.id,
                expression=restoring_force(
                    latent_0.id, center=0.0, stiffness=latent_0_dynamics_decay.id, quartic=0.0
                ),
            ),
        )
    )
    parameters, distributions = without_parameters(
        model, latent_0_latent_1_hill_emax, latent_0_latent_1_hill_n, latent_0_latent_1_hill_ec50
    )
    return model.with_entities(
        edges=replace_constructs(
            tuple(
                edge
                for edge in model.edges
                if (edge.cause.name, edge.effect.name) not in (("latent_0", "latent_1"),)
            ),
            (latent_0_revised,),
        ),
        parameters=parameters,
        distributions=distributions,
    )
