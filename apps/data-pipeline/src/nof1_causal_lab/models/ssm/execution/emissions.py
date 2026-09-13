"""Canonical exact predictor-space log probabilities for observation families.

Each function computes log p(y_t | eta_t) for one time step from the already
constructed linear predictor, observation parameters, and an observation mask.

Used by: MAP and blocked MCMC.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.scipy.stats as jstats

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.models.ssm.covariance_utils import (
    inflate_missing_variance,
)
from nof1_causal_lab.models.ssm.execution.contracts import (
    MISSING_DATA_LARGE_VAR,
    NUMERICAL_EPSILON,
    PROB_CLIP_MIN,
    LikelihoodExtraParams,
)
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    binary_logits_distribution,
    categorical_distribution,
    gaussian_distribution,
    mean_parameter_distribution,
    safe_observation_mean,
    sample_mean_observation,
)
from nof1_causal_lab.models.ssm.execution.observation_extra_params import (
    slice_observation_extra_params,
)
from nof1_causal_lab.models.ssm.shapes import Array, Bool, Float, FloatScalar, Int, Shaped

type MeanLogProbFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]
type MeanSampleFn = Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]


def _logistic_pdf(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    return p * (1.0 - p)


def _logistic_pdf_prime(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    p = jax.nn.sigmoid(x)
    pdf = p * (1.0 - p)
    return pdf * (1.0 - 2.0 * p)


def _normalize_discrete_observation(
    y_t: Float[Array, " M"],
    level_counts: Int[Array, " M"],
) -> tuple[Int[Array, " M"], Bool[Array, " M"]]:
    rounded = jnp.rint(y_t)
    y_idx = rounded.astype(jnp.int32)
    valid = (
        jnp.isfinite(y_t)
        & jnp.isclose(y_t, rounded, atol=1e-4)
        & (y_idx >= 0)
        & (y_idx < level_counts)
    )
    safe_idx = jnp.clip(y_idx, 0, jnp.maximum(level_counts - 1, 0))
    return safe_idx, valid


def _select_rowwise(values: Float[Array, "M C"], indices: Int[Array, " M"]) -> Float[Array, " M"]:
    return jnp.take_along_axis(values, indices[:, None], axis=1).squeeze(axis=1)


def _sum_masked_log_probs(
    log_probs: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    *,
    valid_obs: Bool[Array, " M"] | None = None,
) -> FloatScalar:
    """Sum observed-channel log-probs, returning ``-inf`` on support violations."""
    observed = obs_mask_t > 0.5
    valid = jnp.ones_like(observed, dtype=bool) if valid_obs is None else valid_obs
    invalid_observed = observed & ~valid
    total = jnp.sum(jnp.where(observed & valid, log_probs, 0.0))
    return jnp.where(jnp.any(invalid_observed), -jnp.inf, total)


def _discrete_moments_from_probs(
    probs: Float[Array, "M C"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    class_values = jnp.arange(probs.shape[1], dtype=probs.dtype)
    mean = jnp.sum(probs * class_values[None, :], axis=1)
    second_moment = jnp.sum(probs * (class_values[None, :] ** 2), axis=1)
    variance = jnp.maximum(second_moment - mean**2, NUMERICAL_EPSILON)
    return mean, variance


def ordered_logistic_probabilities(
    eta: Float[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> Float[Array, "M C"]:
    """Return per-channel ordered-logistic probabilities over encoded categories."""
    eta = jnp.asarray(eta)
    cutpoints = jnp.asarray(cutpoints)
    level_counts = jnp.asarray(level_counts, dtype=jnp.int32)

    n_manifest = eta.shape[0]
    max_levels = cutpoints.shape[1] + 1
    boundary_idx = jnp.arange(1, max_levels)
    valid_boundaries = boundary_idx[None, :] < level_counts[:, None]

    cdf_mid = jax.nn.sigmoid(cutpoints - eta[:, None])
    cdf_mid = jnp.where(valid_boundaries, cdf_mid, 1.0)

    cdf = jnp.concatenate(
        [jnp.zeros((n_manifest, 1)), cdf_mid, jnp.ones((n_manifest, 1))],
        axis=1,
    )
    probs = jnp.diff(cdf, axis=1)

    class_mask = jnp.arange(max_levels)[None, :] < level_counts[:, None]
    probs = jnp.where(class_mask, jnp.maximum(probs, 0.0), 0.0)
    norm = jnp.sum(probs, axis=1, keepdims=True)
    fallback = jax.nn.one_hot(jnp.zeros(n_manifest, dtype=jnp.int32), max_levels)
    return jnp.where(
        norm > NUMERICAL_EPSILON,
        probs / jnp.maximum(norm, NUMERICAL_EPSILON),
        fallback,
    )


def categorical_probabilities(
    eta: Float[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> Float[Array, "M C"]:
    """Return per-channel softmax probabilities over encoded categories."""
    eta = jnp.asarray(eta)
    intercepts = jnp.asarray(intercepts)
    slopes = jnp.asarray(slopes)
    level_counts = jnp.asarray(level_counts, dtype=jnp.int32)

    max_levels = intercepts.shape[1] + 1
    nonbaseline_mask = jnp.arange(1, max_levels)[None, :] < level_counts[:, None]
    logits_extra = intercepts + slopes * eta[:, None]
    logits_extra = jnp.where(nonbaseline_mask, logits_extra, -1e30)
    logits = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), logits_extra], axis=1)

    probs = jax.nn.softmax(logits, axis=1)
    class_mask = jnp.arange(max_levels)[None, :] < level_counts[:, None]
    probs = jnp.where(class_mask, probs, 0.0)
    norm = jnp.sum(probs, axis=1, keepdims=True)
    fallback = jax.nn.one_hot(jnp.zeros(eta.shape[0], dtype=jnp.int32), max_levels)
    return jnp.where(
        norm > NUMERICAL_EPSILON,
        probs / jnp.maximum(norm, NUMERICAL_EPSILON),
        fallback,
    )


def ordered_logistic_moments(
    eta: Float[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    return _discrete_moments_from_probs(
        ordered_logistic_probabilities(eta, cutpoints, level_counts)
    )


def categorical_moments(
    eta: Float[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    return _discrete_moments_from_probs(
        categorical_probabilities(eta, intercepts, slopes, level_counts)
    )


def get_ordered_logistic_extra_params(
    extra_params: LikelihoodExtraParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)
    cutpoints = jnp.asarray(extra_params["obs_ordered_cutpoints"])
    return level_counts, cutpoints


def get_categorical_extra_params(
    extra_params: LikelihoodExtraParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    level_counts = jnp.asarray(extra_params["obs_level_counts"], dtype=jnp.int32)
    intercepts = jnp.asarray(extra_params["obs_cat_intercepts"])
    slopes = jnp.asarray(extra_params["obs_cat_slopes"])
    return level_counts, intercepts, slopes


def emission_log_prob_gaussian(y_t, eta, R, obs_mask_t) -> FloatScalar:
    """Native multivariate Gaussian with the existing missing-channel marginalization."""
    residual = jnp.where(obs_mask_t > 0.5, y_t - eta, 0.0)
    n_obs = jnp.sum(obs_mask_t)
    law = gaussian_distribution(jnp.zeros_like(eta), inflate_missing_variance(R, obs_mask_t))
    n_missing = y_t.shape[0] - n_obs
    correction = 0.5 * n_missing * jnp.log(2.0 * jnp.pi * MISSING_DATA_LARGE_VAR)
    return jnp.where(n_obs > 0, law.log_prob(residual) + correction, 0.0)


def emission_log_prob_poisson(y_t, eta, R, obs_mask_t) -> FloatScalar:
    return _mean_log_prob(DistributionFamily.POISSON, y_t, jnp.exp(eta), R, obs_mask_t, {})


def emission_log_prob_student_t(y_t, eta, R, obs_mask_t, df=5.0) -> FloatScalar:
    return _mean_log_prob(DistributionFamily.STUDENT_T, y_t, eta, R, obs_mask_t, {"obs_df": df})


def emission_log_prob_gamma(y_t, eta, R, obs_mask_t, shape=1.0) -> FloatScalar:
    return _mean_log_prob(
        DistributionFamily.GAMMA, y_t, jnp.exp(eta), R, obs_mask_t, {"obs_shape": shape}
    )


def emission_log_prob_bernoulli(y_t, eta, _R, obs_mask_t) -> FloatScalar:
    return _independent_log_prob(binary_logits_distribution(logits=eta), y_t, obs_mask_t)


def emission_log_prob_negative_binomial(y_t, eta, R, obs_mask_t, r=5.0) -> FloatScalar:
    return _mean_log_prob(
        DistributionFamily.NEGATIVE_BINOMIAL, y_t, jnp.exp(eta), R, obs_mask_t, {"obs_r": r}
    )


def emission_log_prob_beta(y_t, eta, R, obs_mask_t, concentration=10.0) -> FloatScalar:
    return _mean_log_prob(
        DistributionFamily.BETA,
        y_t,
        jax.nn.sigmoid(eta),
        R,
        obs_mask_t,
        {"obs_concentration": concentration},
    )


def emission_log_prob_ordered_logistic(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | eta_t) for ordered-logistic observations."""
    probs = ordered_logistic_probabilities(eta, cutpoints, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    return _independent_log_prob(categorical_distribution(probs), y_idx, obs_mask_t, valid_obs)


def emission_log_prob_categorical(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    _R: Float[Array, "M M"],
    obs_mask_t: Shaped[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> FloatScalar:
    """Log p(y_t | eta_t) for categorical softmax observations."""
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    return _independent_log_prob(categorical_distribution(probs), y_idx, obs_mask_t, valid_obs)


def emission_log_prob_bernoulli_probit(y_t, eta, R, obs_mask_t) -> FloatScalar:
    return _mean_log_prob(
        DistributionFamily.BERNOULLI, y_t, jstats.norm.cdf(eta), R, obs_mask_t, {}
    )


def emission_log_prob_gamma_inverse(y_t, eta, R, obs_mask_t, shape=1.0) -> FloatScalar:
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    mean = jnp.where(valid_eta, 1.0 / jnp.where(valid_eta, eta, 1.0), jnp.nan)
    return _mean_log_prob(DistributionFamily.GAMMA, y_t, mean, R, obs_mask_t, {"obs_shape": shape})


def emission_log_prob_beta_probit(y_t, eta, R, obs_mask_t, concentration=10.0) -> FloatScalar:
    return _mean_log_prob(
        DistributionFamily.BETA,
        y_t,
        jstats.norm.cdf(eta),
        R,
        obs_mask_t,
        {"obs_concentration": concentration},
    )


# =============================================================================
# Analytical score (d log p/d eta_j) and neg-Hessian diag (-d^2 log p/d eta_j^2)
# for IEKS linearization. Eliminates jax.hessian from the inner loop.
# All functions: (y_t, eta, obs_mask_t) -> (g_eta, w_eta), shape (n_manifest,).
# =============================================================================


def _score_weight_poisson(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Poisson (log link): score = y - lambda, neg-Hessian = lambda."""
    lam = jnp.exp(eta)
    return (y_t - lam) * obs_mask_t, lam * obs_mask_t


def _score_weight_bernoulli_logit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Bernoulli (logit link): score = y - p, neg-Hessian = p(1-p)."""
    p = jax.nn.sigmoid(eta)
    return (y_t - p) * obs_mask_t, (p * (1.0 - p)) * obs_mask_t


def _score_weight_beta_logit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Beta (logit link): exact score and neg-Hessian via digamma/polygamma."""
    phi = concentration
    mu = jax.nn.sigmoid(eta)
    mu_c = jnp.clip(mu, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu_c
    beta_ = phi * (1.0 - mu_c)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    sig_deriv = mu_c * (1.0 - mu_c)
    psi_diff = jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha)
    score_mu = phi * (logit_y + psi_diff)
    g = sig_deriv * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w_raw = phi * sig_deriv * (phi * sig_deriv * psi1_sum - (1.0 - 2.0 * mu_c) * score_mu)
    return g, jnp.maximum(w_raw, 0.0) * obs_mask_t


def _score_weight_gamma_log(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    shape,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Gamma (log link): score = a*(y/mu - 1), neg-Hessian = a*y/mu."""
    mu = jnp.maximum(jnp.exp(eta), 1e-8)
    ratio = y_t / mu
    return shape * (ratio - 1.0) * obs_mask_t, shape * ratio * obs_mask_t


def _score_weight_gamma_inverse(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    shape,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Gamma (inverse link): score = a*(mu - y), neg-Hessian = a*mu^2."""
    valid_eta = jnp.isfinite(eta) & (eta > 0.0)
    safe_eta = jnp.where(valid_eta, eta, 1.0)
    mu = 1.0 / safe_eta
    g = shape * (mu - y_t) * obs_mask_t
    w = (shape * mu**2) * obs_mask_t
    invalid_observed = (obs_mask_t > 0.5) & ~valid_eta
    g = jnp.where(invalid_observed, jnp.nan, g)
    w = jnp.where(invalid_observed, jnp.nan, w)
    return g, w


def _score_weight_negative_binomial(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    r,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Negative Binomial (log link): exact score and neg-Hessian."""
    mu = jnp.exp(eta)
    denom = r + mu
    g = r * (y_t - mu) / denom * obs_mask_t
    w = r * (r + y_t) * mu / (denom**2) * obs_mask_t
    return g, w


def _score_weight_ordered_logistic(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    cutpoints: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Ordered-logistic score and neg-Hessian w.r.t. the linear predictor."""
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    max_cutpoints = cutpoints.shape[1]
    lower_idx = jnp.clip(y_idx - 1, 0, max_cutpoints - 1)
    upper_idx = jnp.clip(y_idx, 0, max_cutpoints - 1)

    lower_arg = _select_rowwise(cutpoints, lower_idx) - eta
    upper_arg = _select_rowwise(cutpoints, upper_idx) - eta

    has_lower = y_idx > 0
    has_upper = y_idx < (level_counts - 1)

    lower_cdf = jnp.where(has_lower, jax.nn.sigmoid(lower_arg), 0.0)
    upper_cdf = jnp.where(has_upper, jax.nn.sigmoid(upper_arg), 1.0)
    lower_pdf = jnp.where(has_lower, _logistic_pdf(lower_arg), 0.0)
    upper_pdf = jnp.where(has_upper, _logistic_pdf(upper_arg), 0.0)
    lower_pdf_prime = jnp.where(has_lower, _logistic_pdf_prime(lower_arg), 0.0)
    upper_pdf_prime = jnp.where(has_upper, _logistic_pdf_prime(upper_arg), 0.0)

    prob = jnp.maximum(upper_cdf - lower_cdf, NUMERICAL_EPSILON)
    dprob = lower_pdf - upper_pdf
    d2prob = upper_pdf_prime - lower_pdf_prime

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = dprob / prob
    weight = jnp.maximum(score**2 - d2prob / prob, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_categorical(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    intercepts: Float[Array, "M cut"],
    slopes: Float[Array, "M cut"],
    level_counts: Int[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Categorical softmax score and neg-Hessian w.r.t. the linear predictor."""
    probs = categorical_probabilities(eta, intercepts, slopes, level_counts)
    y_idx, valid_obs = _normalize_discrete_observation(y_t, level_counts)
    slope_matrix = jnp.concatenate([jnp.zeros((eta.shape[0], 1)), slopes], axis=1)
    chosen_slope = _select_rowwise(slope_matrix, y_idx)
    mean_slope = jnp.sum(probs * slope_matrix, axis=1)
    second_moment = jnp.sum(probs * (slope_matrix**2), axis=1)

    valid_mask = obs_mask_t * valid_obs.astype(obs_mask_t.dtype)
    score = chosen_slope - mean_slope
    weight = jnp.maximum(second_moment - mean_slope**2, 0.0)
    return score * valid_mask, weight * valid_mask


def _score_weight_bernoulli_probit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Bernoulli (probit link): exact score; Fisher information Hessian."""
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    var = mu * (1.0 - mu)
    return (y_t - mu) * phi_eta / var * obs_mask_t, (phi_eta**2 / var) * obs_mask_t


def _score_weight_beta_probit(
    y_t: Float[Array, " M"],
    eta: Float[Array, " M"],
    obs_mask_t: Shaped[Array, " M"],
    concentration,
) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
    """Beta (probit link): exact score; Gauss-Newton Hessian."""
    phi = concentration
    mu = jnp.clip(jstats.norm.cdf(eta), PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    alpha = phi * mu
    beta_ = phi * (1.0 - mu)
    y_c = jnp.clip(y_t, PROB_CLIP_MIN, 1.0 - PROB_CLIP_MIN)
    logit_y = jnp.log(y_c) - jnp.log(1.0 - y_c)
    phi_eta = jnp.exp(jstats.norm.logpdf(eta))
    score_mu = phi * (logit_y + jax.scipy.special.digamma(beta_) - jax.scipy.special.digamma(alpha))
    g = phi_eta * score_mu * obs_mask_t
    psi1_sum = jax.lax.polygamma(1.0, alpha) + jax.lax.polygamma(1.0, beta_)
    w = jnp.maximum(phi_eta**2 * phi**2 * psi1_sum, 0.0) * obs_mask_t
    return g, w


def _independent_log_prob(law, values, mask, valid_mean=None):
    valid = jnp.isfinite(values) & law.support(values)
    if valid_mean is not None:
        valid = valid & valid_mean
    # NumPyro supplies a support-interior value so missing/invalid observations
    # cannot inject undefined densities or gradients into the observed channels.
    safe_values = jnp.where(valid & (mask > 0.5), values, law.support.feasible_like(values))
    if law.support.is_discrete:
        safe_values = safe_values.astype(jnp.int32)
    return _sum_masked_log_probs(law.log_prob(safe_values), mask, valid_obs=valid)


def _mean_log_prob(family, values, mean, covariance, mask, extra_params):
    safe_mean, valid = safe_observation_mean(family, mean)
    law = mean_parameter_distribution(
        family, safe_mean, jnp.sqrt(jnp.diag(covariance)), extra_params
    )
    # Gamma and Beta measurements exclude endpoints even where a native density
    # has a finite limiting value. This is the authored measurement domain.
    if family == DistributionFamily.GAMMA:
        valid = valid & (values > 0.0)
    elif family == DistributionFamily.BETA:
        valid = valid & (values > 0.0) & (values < 1.0)
    return _independent_log_prob(law, values, mask, valid)


def get_mean_param_log_prob_fn(
    manifest_dist: DistributionFamily | str,
    extra_params: LikelihoodExtraParams | None = None,
) -> MeanLogProbFn:
    """Density in observation mean space, including interval-summary measurements."""
    family = DistributionFamily(manifest_dist)
    if family == DistributionFamily.GAUSSIAN:
        return emission_log_prob_gaussian
    if family in {DistributionFamily.CATEGORICAL, DistributionFamily.ORDERED_LOGISTIC}:
        raise ValueError(f"Mean-parameter log-prob is not defined for {family.value!r}")
    return lambda y, mean, R, mask: _mean_log_prob(family, y, mean, R, mask, extra_params or {})


def get_mean_param_sample_fn(
    manifest_dist: DistributionFamily | str,
    extra_params: LikelihoodExtraParams | None = None,
) -> MeanSampleFn:
    """Draw from the same native law used by the observation likelihood."""
    family = DistributionFamily(manifest_dist)
    if family == DistributionFamily.GAUSSIAN:
        return lambda key, mean, R: gaussian_distribution(mean, R).sample(key)
    if family in {DistributionFamily.CATEGORICAL, DistributionFamily.ORDERED_LOGISTIC}:
        raise ValueError(f"Mean-parameter sampler is not defined for {family.value!r}")
    return lambda key, mean, R: sample_mean_observation(
        family, key, mean, jnp.sqrt(jnp.diag(R)), extra_params or {}
    )


def build_heterogeneous_mean_log_prob_fn(
    manifest_dists: Sequence[DistributionFamily | str],
    extra_params: LikelihoodExtraParams | None = None,
) -> MeanLogProbFn:
    """Build an observation-space log-prob for heterogeneous manifest families."""
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily

    dists = [DistributionFamily(dist) for dist in manifest_dists]
    if len(set(dists)) == 1:
        return get_mean_param_log_prob_fn(dists[0], extra_params)

    from collections import defaultdict

    groups: dict[DistributionFamily, list[int]] = defaultdict(list)
    for ch_idx, dist in enumerate(dists):
        groups[dist].append(ch_idx)

    group_fns: list[tuple[list[int], MeanLogProbFn]] = []
    for dist, ch_indices in groups.items():
        group_fns.append(
            (
                ch_indices,
                get_mean_param_log_prob_fn(
                    dist,
                    slice_observation_extra_params(
                        extra_params,
                        ch_indices,
                        source_channel_count=len(dists),
                    ),
                ),
            )
        )

    def heterogeneous_mean_log_prob(y_t, mean_t, R, obs_mask_t):
        total_ll = 0.0
        for ch_indices, group_fn in group_fns:
            idx = jnp.array(ch_indices)
            y_g = y_t[idx]
            mean_g = mean_t[idx]
            R_g = R[jnp.ix_(idx, idx)]
            mask_g = obs_mask_t[idx]
            total_ll = total_ll + group_fn(y_g, mean_g, R_g, mask_g)
        return total_ll

    return heterogeneous_mean_log_prob


def build_heterogeneous_mean_sample_fn(
    manifest_dists: Sequence[DistributionFamily | str],
    extra_params: LikelihoodExtraParams | None = None,
) -> MeanSampleFn:
    """Build an observation-space sampler for heterogeneous manifest families."""
    from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily

    dists = [DistributionFamily(dist) for dist in manifest_dists]
    if len(set(dists)) == 1:
        return get_mean_param_sample_fn(dists[0], extra_params)

    from collections import defaultdict

    groups: dict[DistributionFamily, list[int]] = defaultdict(list)
    for ch_idx, dist in enumerate(dists):
        groups[dist].append(ch_idx)

    group_fns: list[tuple[list[int], MeanSampleFn]] = []
    for dist, ch_indices in groups.items():
        group_fns.append(
            (
                ch_indices,
                get_mean_param_sample_fn(
                    dist,
                    slice_observation_extra_params(
                        extra_params,
                        ch_indices,
                        source_channel_count=len(dists),
                    ),
                ),
            )
        )

    def heterogeneous_mean_sample(key, mean_t, R):
        sampled = jnp.zeros_like(mean_t)
        keys = jax.random.split(key, len(group_fns))
        for subkey, (ch_indices, group_fn) in zip(keys, group_fns, strict=False):
            idx = jnp.array(ch_indices)
            sampled = sampled.at[idx].set(group_fn(subkey, mean_t[idx], R[jnp.ix_(idx, idx)]))
        return sampled

    return heterogeneous_mean_sample
