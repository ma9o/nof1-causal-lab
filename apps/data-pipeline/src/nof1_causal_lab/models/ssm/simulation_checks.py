"""Reusable reductions of nonlinear simulation arrays, independent of authoring recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import NormalDist
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.construct import (
    ConstructSpec,  # noqa: TC001
)
from nof1_causal_lab.artifacts.expressions import (
    LiteralExpression,
    coefficient_key,
    expression_states,
    fold_expression,
    hill_applications,
    restoring_coefficients,
)
from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.expression import (
    SCALAR_OPERATIONS,
    apply_expression_function,
)
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    predictive_keys,
)
from nof1_causal_lab.models.ssm.reachability import (
    C1B_GROWTH_RATIO,
    C1B_MAX_EXPLOSIVE_FRAC,
    CheckResult,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    check_transmission,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


@dataclass(frozen=True)
class ConstructSimulationTarget:
    """One construct and its incoming mechanisms measured in an existing simulation."""

    construct: ConstructSpec
    edge_parents: tuple[str, ...] = ()
    hill_parents: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.construct.name


@dataclass(frozen=True)
class MeasurementTiming:
    """One measured phase of a construct-admission check."""

    phase: str
    label: str
    duration_ms: float
    checks: tuple[str, ...] = ()


def _spec_names(spec: ModelSpec) -> tuple[list[str], list[str], list[LinkFunction]]:
    """Latent names, manifest names, manifest links of a compiled spec.

    These are always populated after compilation; a missing one is a compiler
    invariant violation, not a recoverable case.
    """
    if (
        numeric.state_names(spec) is None
        or numeric.observation_names(spec) is None
        or numeric.observation_links(spec) is None
    ):
        raise ValueError("compiled ModelSpec is missing latent/manifest metadata")
    return (
        numeric.state_names(spec),
        numeric.observation_names(spec),
        numeric.observation_links(spec),
    )


def _edge_components(spec: ModelSpec, source: int, target: int):
    """Every additive native term belonging to one causal edge."""
    return [
        (i, component)
        for i, component in enumerate(numeric.dynamics_expressions(spec))
        if component.edge_owned and source in component.sources and component.target == target
    ]


def _signal_from_linear_predictor(
    link: LinkFunction,
    lp: np.ndarray,
    *,
    spec: ModelSpec,
    pred: Mapping[str, Any],
    manifest_index: int,
) -> np.ndarray:
    """Noise-free emission mean in data space from the linear predictor."""
    _lp = np.asarray(lp)
    if link == LinkFunction.IDENTITY:
        return _lp
    if link == LinkFunction.LOG:
        return np.exp(np.clip(_lp, -20.0, 20.0))
    if link == LinkFunction.LOGIT:
        return 1.0 / (1.0 + np.exp(-_lp))
    if link == LinkFunction.PROBIT:
        cdf = np.vectorize(NormalDist().cdf, otypes=[float])
        return cdf(_lp)
    if link == LinkFunction.INVERSE:
        return 1.0 / np.clip(_lp, 1e-6, None)

    if numeric.observation_level_counts(spec) is None:
        raise ValueError(f"{link.value} signal extraction requires manifest level counts")
    level_count = int(numeric.observation_level_counts(spec)[manifest_index])
    if level_count < 2:
        raise ValueError(f"{link.value} signal extraction requires at least two declared levels")

    if link == LinkFunction.CUMULATIVE_LOGIT:
        base = np.asarray(pred["obs_ordered_base"])[:, manifest_index]
        if level_count > 2:
            gaps = np.asarray(pred["obs_ordered_gaps"])[:, manifest_index, : level_count - 2]
            cutpoints = np.concatenate(
                [base[:, None], base[:, None] + np.cumsum(gaps, axis=1)],
                axis=1,
            )
        else:
            cutpoints = base[:, None]
        mid_cdf = 1.0 / (
            1.0 + np.exp(-np.clip(cutpoints[:, None, :] - _lp[:, :, None], -30.0, 30.0))
        )
        cdf = np.concatenate(
            [
                np.zeros((*_lp.shape, 1)),
                mid_cdf,
                np.ones((*_lp.shape, 1)),
            ],
            axis=2,
        )
        return np.diff(cdf, axis=2)

    if link == LinkFunction.SOFTMAX:
        intercepts = np.asarray(pred["obs_cat_intercepts"])[:, manifest_index, : level_count - 1]
        slopes = np.asarray(pred["obs_cat_slopes"])[:, manifest_index, : level_count - 1].copy()
        if (
            numeric.categorical_anchors(spec) is not None
            and numeric.categorical_anchors(spec)[manifest_index]
        ):
            slopes[:, 0] = 1.0
        nonbaseline = intercepts[:, None, :] + slopes[:, None, :] * _lp[:, :, None]
        logits = np.concatenate([np.zeros((*_lp.shape, 1)), nonbaseline], axis=2)
        logits = logits - np.max(logits, axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=2, keepdims=True)
        return probabilities
    raise ValueError(f"unsupported link for signal extraction: {link}")


def _draw_scalar_parameter(pred: Mapping[str, Any], name: str, n_draws: int) -> np.ndarray:
    """Return one scalar likelihood hyperparameter per prior draw."""
    values = np.asarray(pred[name], dtype=float)
    if values.shape[0] != n_draws or values.size != n_draws:
        raise ValueError(f"predictive parameter {name!r} must be scalar per draw")
    return values.reshape(n_draws, 1)


def _conditional_variance_for_signal(
    distribution: DistributionFamily,
    signal: np.ndarray,
    pred: Mapping[str, Any],
    manifest_index: int,
) -> np.ndarray:
    """Exact family variance around a scalar prior-predictive emission mean."""
    mean = np.asarray(signal, dtype=float)
    if mean.ndim != 2:
        raise ValueError("scalar conditional variance requires draws by observed-time means")
    n_draws = mean.shape[0]

    if distribution in {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}:
        manifest_cov = np.asarray(pred["manifest_cov"], dtype=float)
        if manifest_cov.shape[0] != n_draws or manifest_cov.ndim != 3:
            raise ValueError("manifest_cov must contain one covariance matrix per draw")
        variance = manifest_cov[:, manifest_index, manifest_index][:, None]
        if distribution == DistributionFamily.STUDENT_T:
            df = _draw_scalar_parameter(pred, "obs_df", n_draws)
            factor = np.full_like(df, np.inf)
            np.divide(df, df - 2.0, out=factor, where=df > 2.0)
            variance = variance * factor
        return np.broadcast_to(variance, mean.shape)

    if distribution == DistributionFamily.POISSON:
        return np.maximum(mean, 1e-8)
    if distribution == DistributionFamily.GAMMA:
        shape = _draw_scalar_parameter(pred, "obs_shape", n_draws)
        return np.maximum(mean, 1e-8) ** 2 / (shape + 1e-8)
    if distribution == DistributionFamily.BERNOULLI:
        probability = np.clip(mean, 1e-7, 1.0 - 1e-7)
        return probability * (1.0 - probability)
    if distribution == DistributionFamily.NEGATIVE_BINOMIAL:
        dispersion = _draw_scalar_parameter(pred, "obs_r", n_draws)
        count_mean = np.maximum(mean, 1e-8)
        return count_mean + count_mean**2 / (dispersion + 1e-8)
    if distribution == DistributionFamily.BETA:
        concentration = _draw_scalar_parameter(pred, "obs_concentration", n_draws)
        probability = np.clip(mean, 1e-7, 1.0 - 1e-7)
        return probability * (1.0 - probability) / (concentration + 1.0)
    raise ValueError(f"{distribution.value} uses probability-vector transmission")


@dataclass(frozen=True)
class DesignInfo:
    """Sampling design + observed data needed by the checks, bound once per build.

    Real longitudinal data is irregular and per-indicator, so observations are
    indexed per indicator: ``obs_index_by_indicator`` maps each indicator to the
    ``t_grid`` indices where it was actually observed, and ``values_by_indicator``
    holds the aligned observed values. ``observation_support`` is passed to the
    exact prior-predictive sampler and edge-off re-simulation when present;
    synthetic tests use a shared index across indicators.

    ``c1b_growth_ratio`` / ``c1b_max_explosive_frac`` calibrate C1b confinement: a
    draw explodes when its late-window amplitude exceeds ``c1b_growth_ratio`` times
    its own early amplitude, and the check reds when at least
    ``c1b_max_explosive_frac`` of draws do. The defaults encode the model class's
    confinement commitment (every self-dynamics component is a restoring force);
    they are design calibration, not part of the statistic — an intrinsically
    trending domain raises them here instead of accepting a standing soft-fail.
    """

    t_grid: jnp.ndarray
    manifest_ids: tuple[str, ...]
    obs_index_by_indicator: Mapping[str, np.ndarray]
    values_by_indicator: Mapping[str, np.ndarray]
    n_draws: int = 200
    seed: int = 0
    c1b_growth_ratio: float = C1B_GROWTH_RATIO
    c1b_max_explosive_frac: float = C1B_MAX_EXPLOSIVE_FRAC
    observation_support: Any = None

    @property
    def pooled_obs_index(self) -> np.ndarray:
        """Sorted union of every indicator's observation indices.

        Used by the latent-at-observation checks (edge overwhelm, saturation) that
        are indicator-agnostic — they read the latent path where any data exists.
        """
        if not self.obs_index_by_indicator:
            return np.arange(int(np.asarray(self.t_grid).shape[0]))
        return np.unique(
            np.concatenate([np.asarray(v) for v in self.obs_index_by_indicator.values()])
        )

    def observation_indices_for(self, indicators: tuple[str, ...]) -> np.ndarray:
        """Sorted union of actual observation indices for the requested indicators."""
        indices = [
            np.asarray(self.obs_index_by_indicator[name])
            for name in indicators
            if name in self.obs_index_by_indicator
            and np.asarray(self.obs_index_by_indicator[name]).size > 0
        ]
        if not indices:
            return np.asarray([], dtype=int)
        return np.unique(np.concatenate(indices))


@dataclass(frozen=True)
class _EdgeOffTarget:
    """One compiled edge coordinate to disable under the same predictive draws."""

    components: tuple[int, ...] = ()


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


def _coefficient_draws(operand, component, pred: Mapping[str, Any], prefix: str) -> np.ndarray:
    if isinstance(operand.value, (int, float)):
        return np.full(np.shape(pred["latents"])[0], operand.value)
    sites = dict(component.parameter_sites(prefix))
    return np.asarray(pred[sites[coefficient_key(operand)].name])


def measure_construct_simulation(
    spec: ModelSpec,
    pred: Mapping[str, jax.Array | np.ndarray],
    design: DesignInfo,
    target: ConstructSimulationTarget,
    *,
    edge_contrasts: bool = True,
) -> tuple[list[CheckResult], list[MeasurementTiming]]:
    """Run the reachability battery on ``target``'s latent trajectory in a compiled model.

    Generic over the construct being measured: admission runs it on the construct being
    admitted; :func:`recheck_member` runs it on an already-admitted cycle member against the
    closed-loop model, where ``target.edge_parents`` now include the just-closed feedback edge.
    """
    latent_names, _manifest_names, manifest_links = _spec_names(spec)
    d = latent_names.index(target.name)
    x = np.asarray(pred["latents"][:, :, d])
    times = np.asarray(design.t_grid, dtype=float)
    indicator_names = tuple(indicator.id for indicator in target.construct.indicators)
    target_obs = design.observation_indices_for(indicator_names)
    structural_indices = target_obs if target_obs.size else np.arange(times.size)

    results: list[CheckResult] = []
    timings: list[MeasurementTiming] = []

    started = perf_counter_ns()
    phase_results = (
        list(
            check_confinement(
                target.name,
                x,
                times,
                growth_ratio=design.c1b_growth_ratio,
                max_explosive_frac=design.c1b_max_explosive_frac,
            )
        )
        if times.size >= 4
        else [
            CheckResult(
                "C1a finiteness",
                target.name,
                f"nonfinite {float(np.mean(~np.isfinite(x))):.1%}",
                "0%",
                bool(np.isfinite(x).all()),
                "Finite-value scan; confinement requires at least four times.",
            )
        ]
    )
    results.extend(phase_results)
    timings.append(
        MeasurementTiming(
            phase="c1_confinement",
            label="C1 confinement",
            duration_ms=_elapsed_ms(started),
            checks=tuple(result.check for result in phase_results),
        )
    )

    started = perf_counter_ns()
    result = check_scale(target.name, x)
    results.append(result)
    timings.append(
        MeasurementTiming(
            phase="c2_latent_scale",
            label="C2 latent scale",
            duration_ms=_elapsed_ms(started),
            checks=(result.check,),
        )
    )

    # C3 resolvability uses the declared stiffness, whether fixed or estimated.
    potentials = [
        (i, comp, operand)
        for i, comp in enumerate(numeric.dynamics_expressions(spec))
        if comp.target == d and not comp.edge_owned
        for operand in restoring_coefficients(comp.expression, comp.state_ids[d], kind=comp.kind)
        if operand.role == "decay"
    ]
    if potentials:
        started = perf_counter_ns()
        stiffness = sum(
            _coefficient_draws(operand, comp, pred, f"vf_{i}") for i, comp, operand in potentials
        )
        tau = 1.0 / np.asarray(stiffness)
        result = check_resolvability(target.name, tau, times[target_obs])
        results.append(result)
        timings.append(
            MeasurementTiming(
                phase="c3_resolvability",
                label="C3 resolvability",
                duration_ms=_elapsed_ms(started),
                checks=(result.check,),
            )
        )

    # C4b edge overwhelm (edge-off re-simulation holds all else fixed).
    for parent in target.edge_parents if edge_contrasts else ():
        started = perf_counter_ns()
        edge_target = _incoming_edge_off_target(
            spec, replace(target, edge_parents=(parent,)), latent_names, d
        )
        x_off = _resimulate_edge_off(
            spec,
            pred,
            design.t_grid,
            edge_target,
            design.seed,
        )[:, :, d]
        edge_label = f"{parent}->{target.name}"
        phase_results = list(
            check_edge_share(
                edge_label,
                x[:, structural_indices],
                np.asarray(x_off)[:, structural_indices],
            )
        )
        results.extend(phase_results)
        timings.append(
            MeasurementTiming(
                phase=f"c4b_edge_overwhelm:{edge_label}",
                label=f"C4b edge-off resimulation: {parent} → {target.name}",
                duration_ms=_elapsed_ms(started),
                checks=tuple(result.check for result in phase_results),
            )
        )

    # C4c Hill saturation (per saturating parent).
    for parent in target.hill_parents:
        p_idx = latent_names.index(parent)
        applications = [
            (i, comp, source, ec50, exponent)
            for i, comp in _edge_components(spec, p_idx, d)
            for source, ec50, exponent in hill_applications(comp.expression)
            if comp.state_ids[p_idx] in expression_states(source)
        ]
        if not applications:
            raise ValueError(f"No Hill expression for {parent!r} -> {target.name!r}")
        for comp_idx, comp, source, ec50, exponent in applications:
            started = perf_counter_ns()
            parent_vals = fold_expression(
                source,
                literal=lambda value: np.asarray(value),
                state_value=lambda key, comp=comp: np.asarray(
                    pred["latents"][:, structural_indices, comp.state_ids.index(key)]
                ),
                coefficient_value=lambda operand, comp=comp, comp_idx=comp_idx: np.asarray(
                    _coefficient_draws(operand, comp, pred, f"vf_{comp_idx}")
                ).reshape(-1, 1),
                binary=lambda operation, left, right: SCALAR_OPERATIONS[operation](left, right),
                call=apply_expression_function,
            )
            result = check_saturation(
                f"{parent}->{target.name}",
                _coefficient_draws(ec50, comp, pred, f"vf_{comp_idx}"),
                _coefficient_draws(exponent, comp, pred, f"vf_{comp_idx}"),
                np.asarray(parent_vals),
            )
            results.append(result)
            timings.append(
                MeasurementTiming(
                    phase=f"c4c_saturation:{parent}->{target.name}",
                    label=f"C4c saturation: {parent} → {target.name}",
                    duration_ms=_elapsed_ms(started),
                    checks=(result.check,),
                )
            )

    time_invariant_mask = numeric.diffusion_block(spec).time_invariant_mask
    target_is_time_invariant = bool(
        time_invariant_mask is not None and np.asarray(time_invariant_mask, dtype=bool)[d]
    )

    # C5a/C5b coverage for every indicator; C5c transmission only for dynamic constructs.
    for indicator in target.construct.indicators:
        lik = indicator.likelihood
        assert lik is not None
        started = perf_counter_ns()
        var = indicator.id
        observed = np.asarray(design.values_by_indicator[var])
        m = design.manifest_ids.index(var)
        oi = np.asarray(design.obs_index_by_indicator[var])
        if not oi.size:
            oi = np.arange(times.size)
        pp_y = np.asarray(pred["observations"][:, oi, m])
        if manifest_links[m] in {LinkFunction.CUMULATIVE_LOGIT, LinkFunction.SOFTMAX}:
            lp = np.asarray(pred["linear_predictors"][:, oi, m])
            signal = _signal_from_linear_predictor(
                manifest_links[m],
                lp,
                spec=spec,
                pred=pred,
                manifest_index=m,
            )
        else:
            signal = np.asarray(pred["expected_observations"][:, oi, m])
        level_count = (
            int(numeric.observation_level_counts(spec)[m])
            if numeric.observation_level_counts(spec) is not None
            else None
        )
        phase_results = (
            list(
                check_coverage(
                    var,
                    pp_y,
                    observed,
                    distribution=lik.law.family.value,
                    level_count=level_count,
                )
            )
            if observed.size
            else []
        )
        if not target_is_time_invariant:
            conditional_variance = (
                None
                if signal.ndim == 3
                else _conditional_variance_for_signal(
                    lik.law.family,
                    signal,
                    pred,
                    m,
                )
            )
            phase_results.append(check_transmission(var, signal, conditional_variance))
        results.extend(phase_results)
        timings.append(
            MeasurementTiming(
                phase=f"c5_coverage:{var}",
                label=f"C5 emission reachability: {var}",
                duration_ms=_elapsed_ms(started),
                checks=tuple(result.check for result in phase_results),
            )
        )

    return results, timings


def _incoming_edge_off_target(
    spec: ModelSpec,
    contribution: ConstructSimulationTarget,
    latent_names: list[str],
    target: int,
) -> _EdgeOffTarget:
    """Compiled vector-field components for incoming edges."""
    components: set[int] = set()
    for parent in contribution.edge_parents:
        if parent in latent_names:
            p_idx = latent_names.index(parent)
            components.update(index for index, _ in _edge_components(spec, p_idx, target))
    if contribution.edge_parents and not components:
        raise ValueError(
            "Could not resolve an edge-off coordinate for incoming parents "
            f"{list(contribution.edge_parents)!r}"
        )
    return _EdgeOffTarget(tuple(sorted(components)))


def _resimulate_edge_off(
    spec: ModelSpec,
    pred: Mapping[str, jax.Array | np.ndarray],
    t_grid: jnp.ndarray,
    edge_target: _EdgeOffTarget,
    seed: int,
) -> jnp.ndarray:
    """Re-simulate the latents with the given edge contributions zeroed, all else fixed.

    Reuses the exact param draws (and Brownian path via ``seed``) from ``pred`` so
    the only difference from the edge-on trajectory is the zeroed edge — the
    same-noise contrast :func:`reachability.check_edge_share` expects.
    """
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        _simulate_vector_field_predictive_latents,
    )

    samples = {name: jnp.asarray(value) for name, value in pred.items()}
    natural = numeric.dynamics_expressions(spec)
    intervention_dynamics = DynamicsSpec(
        n_latent=numeric.n_states(spec),
        components=tuple(
            replace(component, expression=LiteralExpression(value=0))
            if index in edge_target.components
            else component
            for index, component in enumerate(natural)
        ),
    )
    latents, _linear_predictors = _simulate_vector_field_predictive_latents(
        spec,
        samples,
        t_grid,
        rng_key=predictive_keys(seed).latents,
        dynamics=intervention_dynamics,
    )
    return latents
