"""Compile constructor/layout groups for exact observation operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from nof1_causal_lab.artifacts.likelihood import (
    BernoulliLogitsLawSpec,
    BernoulliProbsLawSpec,
    BetaLawSpec,
    CategoricalLawSpec,
    DeltaLawSpec,
    GammaLawSpec,
    Law,
    LinkFunction,
    NegativeBinomial2LawSpec,
    NormalLawSpec,
    OrderedLogisticLawSpec,
    PoissonLawSpec,
    StudentTLawSpec,
    map_law,
)
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize, symmetrize_with_jitter
from nof1_causal_lab.models.ssm.execution import emissions
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    evaluate_law,
    law_response,
    masked_law_log_prob,
    with_response,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.execution.observation_dispatch import (
        PointObservationSampler,
    )
    from nof1_causal_lab.models.ssm.execution.observation_operator import ObservationOperator
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

type EmissionLogProbFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
type ResponseFn = Callable[[jax.Array], jax.Array]
type ScoreWeightFn = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
type LatentGradHessFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


@dataclass(frozen=True)
class ObservationKernel:
    log_prob_fn: EmissionLogProbFn
    response_fn: ResponseFn
    latent_grad_hess_fn: LatentGradHessFn


@dataclass(frozen=True)
class LawGroup:
    """Compatible native operands with their original channel positions."""

    indices: tuple[int, ...]
    laws: ObservationLaws
    interval: bool = False

    def evaluate(
        self,
        predictors: jax.Array,
        scales: jax.Array,
        responses: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> Law[jax.Array]:
        arrays: list[Law[jax.Array]] = []
        for index, law in zip(self.indices, self.laws, strict=True):
            eta = predictors[index]
            if responses is not None:
                eta = jnp.ones_like(eta) if isinstance(law, GammaLawSpec) else jnp.zeros_like(eta)
            observed = None if mask is None else mask[index] > 0.5
            native = evaluate_law(law, eta, scales[index], observed)
            if responses is not None:
                response = responses[index]
                if observed is not None:
                    response = jnp.where(observed, response, 1.0)
                native = with_response(native, response)
            arrays.append(native)
        operands = tuple(iter(value for _, value in law.operands()) for law in arrays)
        return map_law(
            arrays[0], lambda _value: jnp.stack(tuple(next(values) for values in operands))
        )


def compile_law_groups(
    laws: ObservationLaws,
    indices: Sequence[int] | None = None,
    *,
    interval: bool = False,
    roles: Sequence[str] | None = None,
) -> tuple[LawGroup, ...]:
    selected = range(len(laws)) if indices is None else indices
    groups: dict[tuple[type, int, str, LinkFunction], list[int]] = {}
    for index in selected:
        law = laws[index]
        constructor = type(law)
        event_size = 0
        if isinstance(law, OrderedLogisticLawSpec):
            event_size = law.cutpoints.event_size
        elif isinstance(law, CategoricalLawSpec):
            event_size = law.logits.event_size
        if interval and isinstance(law, (BernoulliLogitsLawSpec, BernoulliProbsLawSpec)):
            constructor = BernoulliProbsLawSpec
        role = "point" if roles is None else roles[index]
        if isinstance(law, NormalLawSpec):
            role = "gaussian_block"
        groups.setdefault((constructor, event_size, role, _law_link(law)), []).append(index)
    return tuple(
        LawGroup(tuple(positions), tuple(laws[index] for index in positions), interval)
        for positions in groups.values()
    )


@dataclass(frozen=True)
class CompiledObservationModel:
    kernel: ObservationKernel
    point_sampler: PointObservationSampler
    mean_log_prob_fn: EmissionLogProbFn | None
    observation_operator: ObservationOperator | None

    @property
    def requires_interval_summary_handling(self) -> bool:
        return (
            self.observation_operator is not None
            and self.observation_operator.requires_interval_summary_handling
        )


def _make_glm_grad_hess(score_weight_fn: ScoreWeightFn) -> LatentGradHessFn:
    """Build emission_grad_hess_fn for GLM families (diagonal Hessian in η-space).

    For dist with element-wise log p(y_j | η_j), the chain rule gives:
        g_z = H^T g_eta,   neg_H_z = H^T diag(w_eta) H
    which is always PSD when w_eta >= 0.
    """

    def emission_grad_hess_fn(
        y_t: jnp.ndarray,
        z_t: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        _R: jnp.ndarray,
        mask_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = H @ z_t + d
        g_eta, w_eta = score_weight_fn(y_t, eta, mask_t)
        g_z = H.T @ g_eta
        neg_H_z = H.T @ (w_eta[:, None] * H)
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


def _make_student_t_grad_hess(df: float | jnp.ndarray) -> LatentGradHessFn:
    """Build emission_grad_hess_fn for Student-t (scale extracted from diag(R))."""

    def emission_grad_hess_fn(
        y_t: jnp.ndarray,
        z_t: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
        mask_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = H @ z_t + d
        scale_diag = jnp.sqrt(jnp.diag(R))
        residual = y_t - eta
        sig2 = scale_diag**2
        denom = df * sig2 + residual**2
        g_eta = (df + 1.0) * residual / denom * mask_t
        w_eta = jnp.maximum((df + 1.0) * (df * sig2 - residual**2) / (denom**2), 0.0) * mask_t
        g_z = H.T @ g_eta
        neg_H_z = H.T @ (w_eta[:, None] * H)
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


def _delta_grad_hess(
    _y_t: jnp.ndarray,
    _z_t: jnp.ndarray,
    _H: jnp.ndarray,
    _d: jnp.ndarray,
    _R: jnp.ndarray,
    _mask_t: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    raise ValueError(
        "Delta observations impose exact constraints and have no smooth log-density "
        "for Gaussian initialization."
    )


def _make_gaussian_grad_hess() -> LatentGradHessFn:
    """Build emission_grad_hess_fn for Gaussian (full R, exact analytical form)."""
    from nof1_causal_lab.models.ssm.execution.contracts import (
        MISSING_DATA_LARGE_VAR,
    )

    def emission_grad_hess_fn(
        y_t: jnp.ndarray,
        z_t: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
        mask_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eta = H @ z_t + d
        residual = (y_t - eta) * mask_t
        R_adj = R + jnp.diag((1.0 - mask_t) * MISSING_DATA_LARGE_VAR)
        R_adj = symmetrize_with_jitter(R_adj)
        g_z = H.T @ jla.solve(R_adj, residual, assume_a="pos")
        neg_H_z = H.T @ jla.solve(R_adj, H, assume_a="pos")
        return g_z, symmetrize(neg_H_z)

    return emission_grad_hess_fn


def _law_link(law: Law[BoundExpression]) -> LinkFunction:
    """Every operand of a bound law carries the law's parsed link."""
    return law.operands()[0][1].link


def _group_initialization(group: LawGroup) -> LatentGradHessFn:
    template = group.laws[0]
    if isinstance(template, NormalLawSpec):
        return _make_gaussian_grad_hess()
    if isinstance(template, DeltaLawSpec):
        return _delta_grad_hess

    def score_weight(y: jax.Array, eta: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        size = max(group.indices) + 1
        full_eta = jnp.zeros((size,), dtype=eta.dtype).at[jnp.asarray(group.indices)].set(eta)
        law = group.evaluate(full_eta, jnp.ones((size,), dtype=eta.dtype))
        if isinstance(law, PoissonLawSpec):
            return emissions._score_weight_poisson(y, eta, mask)
        if isinstance(law, BernoulliLogitsLawSpec):
            return emissions._score_weight_bernoulli_logit(y, eta, mask)
        if isinstance(law, BernoulliProbsLawSpec):
            fn = (
                emissions._score_weight_bernoulli_probit
                if _law_link(template) == LinkFunction.PROBIT
                else emissions._score_weight_bernoulli_logit
            )
            return fn(y, eta, mask)
        if isinstance(law, NegativeBinomial2LawSpec):
            return emissions._score_weight_negative_binomial(y, eta, mask, law.concentration)
        if isinstance(law, GammaLawSpec):
            fn = (
                emissions._score_weight_gamma_inverse
                if _law_link(template) == LinkFunction.INVERSE
                else emissions._score_weight_gamma_log
            )
            return fn(y, eta, mask, law.concentration)
        if isinstance(law, BetaLawSpec):
            fn = (
                emissions._score_weight_beta_probit
                if _law_link(template) == LinkFunction.PROBIT
                else emissions._score_weight_beta_logit
            )
            return fn(y, eta, mask, law.concentration1 + law.concentration0)
        if isinstance(law, OrderedLogisticLawSpec):
            return emissions._score_weight_ordered_logistic(
                y, eta, mask, law.cutpoints, jnp.full(eta.shape, law.cutpoints.shape[-1] + 1)
            )
        if isinstance(law, CategoricalLawSpec):
            intercepts = jnp.stack(
                tuple(
                    operand.logits.evaluate(jnp.zeros(()), jnp.ones(()))[1:]
                    for operand in group.laws
                    if isinstance(operand, CategoricalLawSpec)
                )
            )
            slopes = jnp.stack(
                tuple(
                    jax.jacfwd(operand.logits.evaluate, argnums=0)(jnp.zeros(()), jnp.ones(()))[1:]
                    for operand in group.laws
                    if isinstance(operand, CategoricalLawSpec)
                )
            )
            return emissions._score_weight_categorical(
                y, eta, mask, intercepts, slopes, jnp.full(eta.shape, law.logits.shape[-1])
            )
        raise ValueError("This law has no GLM initialization score")

    if isinstance(template, StudentTLawSpec):

        def student(
            y: jax.Array,
            z: jax.Array,
            H: jax.Array,
            d: jax.Array,
            R: jax.Array,
            mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            df = jnp.stack(
                tuple(
                    law.df.evaluate(jnp.zeros(()), jnp.ones(()))
                    for law in group.laws
                    if isinstance(law, StudentTLawSpec)
                )
            )
            return _make_student_t_grad_hess(df)(y, z, H, d, R, mask)

        return student
    return _make_glm_grad_hess(score_weight)


def _group_density(
    group: LawGroup,
    y: jax.Array,
    predictors: jax.Array,
    R: jax.Array,
    mask: jax.Array,
    responses: jax.Array | None = None,
) -> jax.Array:
    indices = jnp.asarray(group.indices)
    law = group.evaluate(predictors, jnp.sqrt(jnp.diag(R)), responses, mask)
    if isinstance(law, NormalLawSpec):
        return emissions.gaussian_block_log_prob(
            y[indices], law.loc, R[jnp.ix_(indices, indices)], mask[indices]
        )
    return masked_law_log_prob(law, y[indices], mask[indices])


def compile_observation_model(
    laws: ObservationLaws,
    *,
    manifest_cov: jax.Array,
    observation_support: ObservationSupportRuntime | None = None,
) -> CompiledObservationModel:
    """Compile exact laws, shared response projection, and initialization operations."""
    from .observation_dispatch import build_point_observation_sampler

    if manifest_cov.shape != (len(laws), len(laws)):
        raise ValueError("manifest_cov must match the compiled observation law axes")
    if observation_support is not None and len(observation_support.support_kinds) != len(laws):
        raise ValueError("observation_support must match the compiled observation law axes")
    roles = (
        None
        if observation_support is None
        else tuple(str(kind) for kind in observation_support.support_kinds)
    )
    groups = compile_law_groups(laws, roles=roles)
    derivatives = tuple(_group_initialization(group) for group in groups)

    def density(y: jax.Array, eta: jax.Array, R: jax.Array, mask: jax.Array) -> jax.Array:
        return sum(
            (_group_density(group, y, eta, R, mask) for group in groups),
            jnp.zeros((), dtype=y.dtype),
        )

    def response(eta: jax.Array) -> jax.Array:
        values = jnp.zeros_like(eta)
        for group in groups:
            native = group.evaluate(eta, jnp.sqrt(jnp.diag(manifest_cov)))
            if isinstance(native, (CategoricalLawSpec, OrderedLogisticLawSpec)):
                declared = law_response(native)
            else:
                declared = jnp.stack(
                    tuple(
                        next(iter(law.operands()))[1].response_fn(eta[index])
                        for index, law in zip(group.indices, group.laws, strict=True)
                    )
                )
            values = values.at[jnp.asarray(group.indices)].set(declared)
        return values

    def grad_hess(
        y: jax.Array,
        z: jax.Array,
        H: jax.Array,
        d: jax.Array,
        R: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        gradient, hessian = jnp.zeros_like(z), jnp.zeros((z.size, z.size), dtype=z.dtype)
        for group, derivative in zip(groups, derivatives, strict=True):
            indices = jnp.asarray(group.indices)
            g, h = derivative(
                y[indices], z, H[indices], d[indices], R[jnp.ix_(indices, indices)], mask[indices]
            )
            gradient, hessian = gradient + g, hessian + h
        return gradient, hessian

    operator = compile_observation_operator(observation_support)
    kernel = ObservationKernel(density, response, grad_hess)
    point_sampler = build_point_observation_sampler(laws, manifest_cov, groups=groups)
    if operator is None or not operator.requires_interval_summary_handling:
        return CompiledObservationModel(kernel, point_sampler, None, operator)
    interval_groups = compile_law_groups(laws, operator.interval_summary_indices, interval=True)
    if any(
        isinstance(group.laws[0], (CategoricalLawSpec, OrderedLogisticLawSpec))
        for group in interval_groups
    ):
        raise ValueError("Category laws have no scalar interval-summary response")

    def mean_density(y: jax.Array, means: jax.Array, R: jax.Array, mask: jax.Array) -> jax.Array:
        return sum(
            (
                _group_density(group, y, jnp.zeros_like(means), R, mask, means)
                for group in interval_groups
            ),
            jnp.zeros((), dtype=y.dtype),
        )

    return CompiledObservationModel(
        kernel,
        point_sampler,
        mean_density,
        operator,
    )
