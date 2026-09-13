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

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from nof1_causal_lab.models.ssm.execution.emissions import build_heterogeneous_mean_sample_fn
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.statistical_model_spec import LinkFunction
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.execution.contracts import (
        LikelihoodExtraParams,
        MeasurementParams,
    )


class StructuralDrift(eqx.Module):
    """Causal component parameters and interventions in Dynestyx's drift signature."""

    vector_field: VectorField
    args: VectorFieldArgs
    input_effect: jax.Array | None = None

    def __call__(self, x, u, t):
        value = self.vector_field(jnp.asarray(t), x, self.args)
        if self.input_effect is not None and self.input_effect.shape[1]:
            value = value + self.input_effect @ u
        return value


def continuous_state_evolution(
    vector_field: VectorField,
    vf_params: tuple[dict[str, jax.Array], ...],
    diffusion_cov: jax.Array,
    input_effect: jax.Array | None = None,
    *,
    intervention: Intervention | None = None,
) -> dsx.StochasticContinuousTimeStateEvolution:
    """Declare the nonlinear SDE without selecting a numerical approximation."""
    return dsx.StochasticContinuousTimeStateEvolution(
        drift=StructuralDrift(
            vector_field,
            VectorFieldArgs(
                params=vf_params,
                intervention=Intervention.none() if intervention is None else intervention,
            ),
            input_effect,
        ),
        diffusion=dsx.FullDiffusion(jnp.linalg.cholesky(symmetrize(diffusion_cov))),
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
        return _ObservationDistribution(x, self)


class _ObservationDistribution(dist.Distribution):
    """NumPyro distribution for one heterogeneous measurement row."""

    arg_constraints: ClassVar[dict[str, dist.constraints.Constraint]] = {}
    support = dist.constraints.real_vector

    def __init__(self, state: jax.Array, observation: HeterogeneousObservation) -> None:
        self._state = state
        self._measurement = observation.measurement
        compiled = compile_observation_model(
            observation.families,
            manifest_cov=self._measurement.manifest_cov,
            extra_params=observation.extra_params,
            manifest_links=observation.links,
        )
        self._kernel = compiled.kernel
        self._sampler = build_heterogeneous_mean_sample_fn(
            compiled.manifest_dists, observation.extra_params
        )
        super().__init__(
            batch_shape=(),
            event_shape=(int(self._measurement.lambda_mat.shape[0]),),
            validate_args=False,
        )

    @property
    def mean(self):
        measurement = self._measurement
        return self._kernel.response_fn(
            measurement.lambda_mat @ self._state + measurement.manifest_means
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        covariance = self._measurement.manifest_cov
        if not sample_shape:
            return self._sampler(key, self.mean, covariance)
        keys = jax.random.split(key, sample_shape)
        flat = jax.vmap(lambda k: self._sampler(k, self.mean, covariance))(
            jax.random.key_data(keys).reshape((-1, 2))
        )
        return flat.reshape((*sample_shape, *self.event_shape))

    def log_prob(self, value: jax.Array) -> jax.Array:
        observation = jnp.asarray(value)
        mask = ~jnp.isnan(observation)
        filled = jnp.nan_to_num(observation, nan=0.0)
        measurement = self._measurement
        predictor = measurement.lambda_mat @ self._state + measurement.manifest_means
        return jnp.asarray(
            self._kernel.log_prob_fn(
                filled, predictor, measurement.manifest_cov, mask.astype(self._state.dtype)
            ),
            dtype=self._state.dtype,
        )


def build_dynamical_model(
    dynamics: dsx.StochasticContinuousTimeStateEvolution,
    measurement: MeasurementParams,
    initial: dist.MultivariateNormal,
    families: tuple[DistributionFamily, ...],
    links: tuple[LinkFunction, ...],
    extra_params: LikelihoodExtraParams | None,
    *,
    control_dim: int,
    t0: jax.Array,
) -> dsx.DynamicalModel:
    """The single executable model consumed by particle inference."""
    return dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=dynamics,
        observation_model=HeterogeneousObservation(measurement, families, links, extra_params),
        control_dim=control_dim,
        t0=t0,
    )
