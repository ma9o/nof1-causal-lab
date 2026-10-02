"""Total numerical accessors over resolved compiler-owned facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.structure import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseBlockSpec,
        T0CholBlockSpec,
    )


def state_ids(model: CompiledModel) -> tuple[ConstructId, ...]:
    return tuple(state.id for state in model.states)


def state_names(model: CompiledModel) -> tuple[str, ...]:
    return tuple(state.name for state in model.states)


def observation_ids(model: CompiledModel) -> tuple[IndicatorId, ...]:
    return tuple(observation.id for observation in model.observations)


def observation_names(model: CompiledModel) -> tuple[str, ...]:
    return tuple(observation.name for observation in model.observations)


def observation_families(model: CompiledModel) -> tuple[DistributionFamily, ...]:
    return tuple(observation.law.family for observation in model.observations)


def observation_level_counts(model: CompiledModel) -> tuple[int, ...]:
    return tuple(len(observation.levels) for observation in model.observations)


def observation_standardized(model: CompiledModel) -> tuple[bool, ...]:
    return tuple(observation.standardized for observation in model.observations)


def categorical_anchors(model: CompiledModel) -> tuple[bool, ...]:
    return tuple(observation.categorical_anchor for observation in model.observations)


def time_invariant_mask(model: CompiledModel) -> np.ndarray:
    return np.asarray([state.time_invariant for state in model.states], dtype=bool)


def input_mask(model: CompiledModel) -> np.ndarray:
    return np.asarray([state.is_input for state in model.states], dtype=bool)


def diffusion_families(model: CompiledModel) -> tuple[DistributionFamily, ...]:
    return tuple(state.innovation_family for state in model.states)


def static_factor_ids(model: CompiledModel) -> tuple[ConstructId, ...]:
    return tuple(state.id for state in model.static_factors)


def static_factor_names(model: CompiledModel) -> tuple[str, ...]:
    return tuple(state.name for state in model.static_factors)


def n_states(model: CompiledModel) -> int:
    return len(model.states)


def n_observations(model: CompiledModel) -> int:
    return len(model.observations)


def parameter_blocks(
    model: CompiledModel,
) -> tuple[
    DiffusionBlockSpec,
    SparseBlockSpec[tuple[int, int]],
    SparseBlockSpec[int],
    ManifestCholBlockSpec,
    SparseBlockSpec[int],
    T0CholBlockSpec,
    SparseBlockSpec[int],
]:
    return (
        model.diffusion_block,
        model.loading_block,
        model.observation_mean_block,
        model.observation_noise_block,
        model.initial_mean_block,
        model.initial_covariance_block,
        model.static_scale_block,
    )
