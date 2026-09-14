"""Compile scientific drift and measurement semantics into Dynestyx models.

Dynestyx owns state evolution, initial distributions, and numerical interpretation.
These adapters supply the application's causal vector field and heterogeneous
measurement semantics. Their parameters are explicit pytree leaves, so a model
can be carried through JAX transformations without closing over traced arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model

if TYPE_CHECKING:
    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.artifacts.likelihood import LinkFunction
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.execution.contracts import (
        LikelihoodExtraParams,
    )
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


def continuous_state_evolution(
    vector_field: VectorField,
    vf_params: tuple[dict[str, jax.Array], ...],
    diffusion_cov: jax.Array,
    input_effect: jax.Array | None = None,
    *,
    intervention: Intervention | None = None,
) -> dsx.StochasticContinuousTimeStateEvolution:
    """Declare the nonlinear SDE without selecting a numerical approximation."""
    intervention = Intervention.none() if intervention is None else intervention
    diffusion = jnp.linalg.cholesky(symmetrize(diffusion_cov))
    for clamp in intervention.variable_overrides():
        diffusion = diffusion.at[clamp.index].set(0.0)
    return vector_field.evolution(
        VectorFieldArgs(params=vf_params, intervention=intervention),
        input_effect=input_effect,
        diffusion=dsx.FullDiffusion(diffusion),
    )


class HeterogeneousObservation(dsx.ObservationModel):
    """Measurement mapping with the application's family/link and missingness rules."""

    measurement: MeasurementParams
    families: tuple[DistributionFamily, ...] = eqx.field(static=True)
    links: tuple[LinkFunction, ...] = eqx.field(static=True)
    extra_params: LikelihoodExtraParams | None = None

    @override
    def __call__(self, x, u, t):
        del u, t
        return self.at_predictor(self.linear_predictor(x))

    def linear_predictor(self, state):
        return self.measurement.lambda_mat @ state + self.measurement.manifest_means

    def at_predictor(self, predictor):
        """Use the same observation law with cached predictors and interval projections."""
        return _ObservationDistribution(predictor, self)


class _ObservationDistribution(dist.Distribution):
    """NumPyro distribution for one heterogeneous measurement row."""

    arg_constraints: ClassVar[dict[str, dist.constraints.Constraint]] = {}
    support = dist.constraints.real_vector

    def __init__(self, predictor: jax.Array, observation: HeterogeneousObservation) -> None:
        self._predictor = predictor
        self._measurement = observation.measurement
        compiled = compile_observation_model(
            observation.families,
            manifest_cov=self._measurement.manifest_cov,
            extra_params=observation.extra_params,
            manifest_links=observation.links,
        )
        self._kernel = compiled.kernel
        self._sampler = compiled.point_sampler.sample_point
        super().__init__(
            batch_shape=(),
            event_shape=(int(self._measurement.lambda_mat.shape[0]),),
            validate_args=False,
        )

    @property
    def mean(self):
        return self._kernel.response_fn(self._predictor)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        if not sample_shape:
            return self._sampler(key, self._predictor)
        from math import prod

        keys = jax.random.split(key, prod(sample_shape))
        flat = jax.vmap(lambda k: self._sampler(k, self._predictor))(keys)
        return flat.reshape((*sample_shape, *self.event_shape))

    def log_prob(self, value: jax.Array) -> jax.Array:
        observation = jnp.asarray(value)
        mask = ~jnp.isnan(observation)
        filled = jnp.nan_to_num(observation, nan=0.0)
        measurement = self._measurement
        return jnp.asarray(
            self._kernel.log_prob_fn(
                filled,
                self._predictor,
                measurement.manifest_cov,
                mask.astype(self._predictor.dtype),
            ),
            dtype=self._predictor.dtype,
        )


def assemble_likelihood_inputs(
    samples: dict[str, jnp.ndarray],
    spec: ModelSpec,
    *,
    registry: list[SiteDescriptor],
    dynamics: DynamicsSpec | None = None,
) -> tuple[
    StochasticContinuousTimeStateEvolution,
    MeasurementParams,
    MultivariateNormal,
    LikelihoodExtraParams | None,
]:
    """Consume the canonical deterministic values from NumPyro's prior replay."""
    from nof1_causal_lab.models.ssm.parameterization import assemble_extra_params_from_registry

    dynamics_spec = numeric.dynamics_components(spec) if dynamics is None else dynamics
    compiled = compile_dynamics(dynamics_spec)
    diffusion_chol = samples["diffusion"]
    dynamics = continuous_state_evolution(
        vector_field=compiled.vector_field,
        vf_params=pack_component_params_from_samples(dynamics_spec, samples),
        diffusion_cov=diffusion_chol @ diffusion_chol.T,
        input_effect=samples["input_effect"] if numeric.input_effect_block(spec).n_cols else None,
    )
    measurement = MeasurementParams(
        lambda_mat=samples["lambda"],
        manifest_means=samples["manifest_means"],
        manifest_cov=samples["manifest_cov"],
    )
    initial = MultivariateNormal(samples["t0_means"], covariance_matrix=samples["t0_cov"])
    extra = assemble_extra_params_from_registry(spec, samples, registry)
    return dynamics, measurement, initial, extra or None


def build_dynamical_model(
    model_spec: ModelSpec,
    values: dict[str, jnp.ndarray],
    *,
    t0: jax.Array,
    dynamics: DynamicsSpec | None = None,
) -> dsx.DynamicalModel:
    """Construct Dynestyx's model from the source ModelSpec and one parameter draw.

    Values include the matrices derived by assemble_model_matrices or by
    NumPyro's prior replay. An execution-only dynamics override supports paired
    edge-off admission contrasts while retaining the same measurement and initial laws.
    """
    from nof1_causal_lab.models.ssm.parameterization import build_site_registry

    evolution, measurement, initial, extra_params = assemble_likelihood_inputs(
        values, model_spec, registry=build_site_registry(model_spec), dynamics=dynamics
    )
    return dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=evolution,
        observation_model=HeterogeneousObservation(
            measurement,
            tuple(numeric.observation_families(model_spec)),
            tuple(numeric.observation_links(model_spec)),
            extra_params,
        ),
        control_dim=len(numeric.input_ids(model_spec)),
        t0=t0,
    )
