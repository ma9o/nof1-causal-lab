"""Pure authoring-prior planning before SSM compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.prior_distributions import distribution_from_params

if TYPE_CHECKING:
    import numpyro.distributions as dist

    from nof1_causal_lab.json_types import JsonObject


def default_parameter_prior(parameter: ParameterSpec) -> dist.Distribution:
    """Choose the explicit authoring default for one semantic parameter."""
    if parameter.role == ParameterRole.AR_COEFFICIENT:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.role == ParameterRole.LOADING:
        distribution = PriorDistributionFamily.NORMAL
        params = {
            "mu": -0.5 if parameter.constraint == ParameterConstraint.NEGATIVE else 0.5,
            "sigma": 0.5,
        }
    elif parameter.constraint == ParameterConstraint.POSITIVE:
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}
    elif parameter.constraint == ParameterConstraint.NEGATIVE:
        distribution = PriorDistributionFamily.TRUNCATED_NORMAL
        params = {"mu": -1.0, "sigma": 0.5, "lower": -5.0, "upper": 0.0}
    elif parameter.constraint == ParameterConstraint.UNIT_INTERVAL:
        distribution = PriorDistributionFamily.BETA
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.CORRELATION:
        distribution = PriorDistributionFamily.UNIFORM
        params = {"lower": -1.0, "upper": 1.0}
    else:
        distribution = PriorDistributionFamily.NORMAL
        params = {"mu": 0.0, "sigma": 0.5}

    if parameter.role in (ParameterRole.RESIDUAL_SD, ParameterRole.STATIC_STATE_SD):
        distribution = PriorDistributionFamily.HALF_NORMAL
        params = {"sigma": 1.0}

    return distribution_from_params(distribution, params)


def complete_parameter_priors(model: StatisticalModelSpec) -> StatisticalModelSpec:
    """Apply the model's explicit default policy to its still-unassigned parameters."""
    return model.model_copy(
        update={
            "parameters": [
                parameter
                if parameter.prior is not None
                else parameter.model_copy(
                    update={
                        "prior": default_parameter_prior(parameter),
                        "prior_reasoning": "Default scientific prior policy.",
                    }
                )
                for parameter in model.parameters
            ]
        }
    )


def parameter_with_prior(parameter: ParameterSpec, payload: JsonObject) -> ParameterSpec:
    """Attach a tool submission's distribution and scientific evidence to its parameter."""
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.prior import DensityPoint, PriorSource

    supplied_id = payload.get("parameter_id")
    if supplied_id is not None and supplied_id != parameter.id:
        raise ValueError(f"Prior for {parameter.name!r} references a different parameter")
    params = payload["params"]
    if not isinstance(params, dict):
        raise ValueError("Prior constructor params must be a JSON object")
    return ParameterSpec.model_validate(
        {
            **parameter.model_dump(mode="python"),
            "prior": distribution_from_params(
                PriorDistributionFamily(payload["distribution"]), params
            ),
            "reference_interval_days": payload.get("reference_interval_days"),
            "prior_reasoning": payload.get("reasoning", ""),
            "prior_sources": TypeAdapter(list[PriorSource]).validate_python(
                payload.get("sources", [])
            ),
            "prior_density_points": TypeAdapter(list[DensityPoint] | None).validate_python(
                payload.get("density_points")
            ),
        }
    )
