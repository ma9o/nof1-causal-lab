"""Pure authoring-prior planning before SSM compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import (
        ParameterSpec,
    )
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


def default_parameter_prior(
    parameter: ParameterSpec, model: ModelSpec, site: SiteDescriptor
) -> dist.Distribution:
    """The authored default policy, using native support and the owner's orientation."""
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind, SupportClass

    if parameter.distribution_transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
        return dist.Beta(2.0, 2.0)
    if site.site_kind == SiteKind.LOADING:
        indicator = model.indicator(
            next(
                owner.id
                for owner in model.parameter_context(parameter.id).owners
                if owner.kind == "indicator"
            )
        )
        return dist.Normal(-0.5 if indicator.construct_polarity.value == "negative" else 0.5, 0.5)
    if site.support == SupportClass.POSITIVE:
        return dist.HalfNormal(1.0)
    if site.support == SupportClass.CORRELATION:
        return dist.Uniform(-1.0, 1.0)
    return dist.Normal(0.0, 0.5)


def complete_parameter_priors(model: ModelSpec) -> ModelSpec:
    """Apply the explicit default policy using the native sites of this scientific model."""
    from nof1_causal_lab.models.model_distributions import with_parameter_distributions
    from nof1_causal_lab.models.ssm.compile.prior_indexing import build_semantic_prior_bindings
    from nof1_causal_lab.models.ssm.parameterization import build_site_registry

    model.require_execution_structure()
    bindings = build_semantic_prior_bindings(model).by_parameter
    sites = {site.name: site for site in build_site_registry(model)}
    return with_parameter_distributions(
        model,
        {
            parameter.id: default_parameter_prior(
                parameter, model, sites[bindings[parameter.id].site_name]
            )
            for parameter in model.parameters
            if parameter.distribution is None and parameter.value is None
        },
    )


def complete_model(model: ModelSpec) -> ModelSpec:
    """Explicitly complete the scientific inventory and prior policy before committing."""
    from nof1_causal_lab.models.parameter_planning import complete_component_slots

    model.require_execution_structure()
    candidate = complete_component_slots(model)
    return complete_parameter_priors(candidate)
