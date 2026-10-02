"""Exact particle targets evaluated through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.models.core import DiscreteStateTransition

    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints
    from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform


# Array leaves of the continuous model and its observation grid. Static model
# metadata is retained by the discrete-model factory, outside the sampler state.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array]


@dataclass(frozen=True, eq=False)
class ParticleTarget:
    """Adapt a Dynestyx model to the application's particle sampling contract.

    Every transition operation uses the same declared distribution. The model's
    observation distribution owns missingness and the true emission density.
    """

    parameters: ParameterTransform
    context: Callable[[jax.Array, jax.Array], ParticleContext]
    model: Callable[[ParticleContext], dsx.DynamicalModel]
    observations: jax.Array
    times: jax.Array
    density_indices: tuple[int, ...] | None = None

    @property
    def initial_position(self):
        return self.parameters.initial_position

    def log_prior(self, position):
        return self.parameters.log_prior(position)

    def initial_moments(self, context):
        distribution = self.model(context).initial_condition
        assert isinstance(
            distribution, dist.MultivariateNormal
        )  # The exact model producer declares a Gaussian initial law.
        return jnp.asarray(distribution.mean), jnp.asarray(distribution.covariance_matrix)

    def initial_log_prob(self, context, state):
        return self._modeled_log_prob(self.model(context).initial_condition, state)

    def _modeled_log_prob(self, distribution, state):
        indices = jnp.asarray(
            self.density_indices
            if self.density_indices is not None
            else tuple(range(state.shape[-1]))
        )
        return dist.MultivariateNormal(
            distribution.mean[indices],
            covariance_matrix=distribution.covariance_matrix[jnp.ix_(indices, indices)],
        ).log_prob(state[indices])

    def _transition_distribution(self, context, previous, index):
        times = context[1]
        evolution = cast("DiscreteStateTransition", self.model(context).state_evolution)
        return evolution(previous, None, times[jnp.maximum(index - 1, 0)], times[index])

    def transition_log_prob(self, context, previous, current, index):
        return self._modeled_log_prob(
            self._transition_distribution(context, previous, index), current
        )

    def aligned_transition_log_prob(self, context, previous, current, index):
        return jax.vmap(lambda a, b: self.transition_log_prob(context, a, b, index))(
            previous, current
        )

    def pairwise_transition_log_prob(self, context, previous, current, index):
        return jax.vmap(
            lambda ancestor: jax.vmap(
                lambda descendant: self.transition_log_prob(context, ancestor, descendant, index)
            )(current)
        )(previous)

    def initial_path(self, context, *, exact_constraints: ExactStateConstraints | None = None):
        initial = jnp.asarray(self.model(context).initial_condition.mean)
        if exact_constraints is not None:
            initial = jnp.where(
                exact_constraints.free_mask[0], initial, exact_constraints.values[0]
            )

        def step(previous, index):
            current = self._transition_distribution(context, previous, index).mean
            if exact_constraints is not None:
                current = jnp.where(
                    exact_constraints.free_mask[index], current, exact_constraints.values[index]
                )
            return current, current

        _, tail = jax.lax.scan(step, initial, jnp.arange(1, context[1].size))
        return jnp.concatenate([initial[None], tail])

    def observation_increment(self, context, state, index, observations):
        distribution = self.model(context).observation_model(state, None, context[1][index])
        return jnp.sum(distribution.log_prob(observations[index]))

    def observation_log_probs(self, context, path, observations):
        return jax.vmap(
            lambda state, index: self.observation_increment(context, state, index, observations)
        )(path, jnp.arange(path.shape[0]))

    def path_log_prob(self, context, path, observations):
        transitions = jax.vmap(
            lambda previous, current, index: self.transition_log_prob(
                context, previous, current, index
            )
        )(path[:-1], path[1:], jnp.arange(1, path.shape[0]))
        return (
            self.initial_log_prob(context, path[0])
            + jnp.sum(transitions)
            + jnp.sum(self.observation_log_probs(context, path, observations))
        )

    def log_posterior_from_context(self, position, context, path, observations):
        path_density = self.path_log_prob(context, path, observations)
        return self.log_prior(position) + path_density, path_density

    def log_posterior(self, position, path, observations, times):
        return self.log_posterior_from_context(
            position, self.context(position, times), path, observations
        )[0]
