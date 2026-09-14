"""Pure derivations of numerical support from scientific structure and policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import (
    indicator_requires_observation_intercept,
)
from nof1_causal_lab.utils.model_structure import (
    get_constructs,
    get_edges,
    get_indicators,
    get_known_inputs,
    get_model_clock,
    get_reference_indicator_lookup,
    get_reference_indicator_polarities,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


class NumericalSupportError(AggregatedCompileError):
    """Aggregate failures to derive numerical support from ModelSpec."""

    header = "Numerical derivation failed"


def get_construct_dt_days(
    model: ModelSpec,
    _construct_name: str = "",
) -> float:
    """Get the model clock interval in fractional days."""
    return parse_duration_to_hours(get_model_clock(model)) / 24.0


def get_structural_input_layout(
    model: ModelSpec,
) -> tuple[
    list[str],
    list[str],
    list[float],
    list[str],
    list[bool],
]:
    """Build canonical known-input ordering and source metadata."""
    known_inputs = get_known_inputs(model)
    estimation_edges = get_edges(model)
    input_lagged: list[bool] = []
    for item in known_inputs:
        name = str(item["construct"])
        edge_lags = {bool(edge["lagged"]) for edge in estimation_edges if edge.get("cause") == name}
        if not edge_lags:
            raise NumericalSupportError(
                [f"Known input {name!r} has no outgoing edge into a retained state"]
            )
        if len(edge_lags) > 1:
            raise NumericalSupportError(
                [
                    f"Known input {name!r} has mixed contemporaneous and lagged outgoing "
                    "edges; one input trajectory must use a consistent alignment"
                ]
            )
        input_lagged.append(edge_lags.pop())
    return (
        [str(item["construct"]) for item in known_inputs],
        [str(item["source_indicator"]) for item in known_inputs],
        [float(item.get("scale", 1.0)) for item in known_inputs],
        [str(item.get("missing_policy", "zero")) for item in known_inputs],
        input_lagged,
    )


def build_structural_support_from_model(
    latent_names: list[str],
    manifest_cols: list[str],
    n_latent: int,
    n_manifest: int,
    *,
    manifest_dists: list[DistributionFamily],
    model: ModelSpec,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], float],
]:
    """Build block/component support arrays and edge lag metadata from causal structure."""

    try:
        edges = get_edges(model)
    except ValueError as exc:
        raise NumericalSupportError([str(exc)]) from exc
    latent_construct_lookup = {construct["name"]: construct for construct in get_constructs(model)}
    indicators = get_indicators(model)
    errors: list[str] = []

    indicator_names = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name)
        for indicator in indicators
    }
    unknown_likelihoods = sorted(set(manifest_cols) - indicator_names)
    if unknown_likelihoods:
        errors.append(
            "ModelSpec likelihoods reference indicators absent from model measurement: "
            f"{unknown_likelihoods}"
        )

    latent_idx = {name: idx for idx, name in enumerate(latent_names)}
    input_names, _input_sources, _input_scales, _input_policies, _input_lagged = (
        get_structural_input_layout(model)
    )
    input_idx = {name: idx for idx, name in enumerate(input_names)}
    state_dynamics_support = np.zeros((n_latent, n_latent), dtype=bool)
    input_effect_support = np.zeros((n_latent, len(input_names)), dtype=bool)
    for latent_name, latent_idx_value in latent_idx.items():
        construct = latent_construct_lookup.get(latent_name) or {}
        if construct.get("temporal_status") != "time_invariant":
            state_dynamics_support[latent_idx_value, latent_idx_value] = True
    edge_lag_days: dict[tuple[int, int], float] = {}
    model_dt_days = get_construct_dt_days(model)

    for edge in edges:
        cause = edge["cause"]
        effect = edge["effect"]
        if effect not in latent_idx:
            continue
        if latent_construct_lookup.get(effect, {}).get("temporal_status") == "time_invariant":
            errors.append(
                "ModelSpec contains an unsupported static-target edge that should "
                f"have failed planning: {edge.get('source_id')!r} ({cause!r} -> {effect!r})."
            )
            continue
        effect_idx = latent_idx[effect]
        if cause in input_idx:
            input_effect_support[effect_idx, input_idx[cause]] = True
            continue
        if cause not in latent_idx:
            continue
        cause_idx = latent_idx[cause]
        state_dynamics_support[effect_idx, cause_idx] = True

        lagged = edge.get("lagged", True) if isinstance(edge, dict) else edge.lagged
        lag_hours = model_dt_days * 24.0 if lagged else 0.0
        if lag_hours > 0:
            edge_lag_days[(effect_idx, cause_idx)] = lag_hours / 24.0

    manifest_idx = {name: idx for idx, name in enumerate(manifest_cols)}
    lambda_mat_np = np.zeros((n_manifest, n_latent), dtype=np.float64)
    lambda_support = np.zeros((n_manifest, n_latent), dtype=bool)
    reference_indicator_lookup = get_reference_indicator_lookup(model)
    reference_indicator_polarities = get_reference_indicator_polarities(model)
    matched_manifests: set[str] = set()
    invalid_construct_manifests: set[str] = set()
    construct_channels: dict[str, list[int]] = {}

    for indicator in indicators:
        ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
        construct_name = indicator["construct_name"]
        if ind_name not in manifest_idx:
            continue
        if construct_name not in latent_idx:
            errors.append(
                "ModelSpec manifest indicator references unknown retained construct: "
                f"{ind_name!r} -> {construct_name!r}"
            )
            invalid_construct_manifests.add(ind_name)
            continue

        manifest_idx_value = manifest_idx[ind_name]
        latent_idx_value = latent_idx[construct_name]
        matched_manifests.add(ind_name)
        construct_channels.setdefault(construct_name, []).append(manifest_idx_value)

        if manifest_dists[manifest_idx_value] == DistributionFamily.CATEGORICAL:
            # Categorical slopes multiply the whole linear predictor, so a free
            # loading is exactly redundant with them (only the products enter
            # the likelihood). The loading is pinned and the slopes carry the
            # channel's discrimination; sign lives in the slopes as well, so
            # polarity is ignored.
            lambda_mat_np[manifest_idx_value, latent_idx_value] = 1.0
        elif ind_name == reference_indicator_lookup.get(construct_name):
            lambda_mat_np[manifest_idx_value, latent_idx_value] = (
                1.0 if reference_indicator_polarities[construct_name] == "positive" else -1.0
            )
        else:
            lambda_support[manifest_idx_value, latent_idx_value] = True

    lambda_mat = lambda_mat_np

    # A construct measured only by categorical channels has no fixed-link-scale
    # channel pinning its latent scale, and nominal categories break no
    # reflection symmetry. Pin the reference channel's first non-baseline slope
    # to +1 as the construct's scale and sign anchor (Bock-NRM style).
    manifest_cat_anchor = np.zeros(n_manifest, dtype=bool)
    for construct_name, channel_indices in construct_channels.items():
        if not all(
            manifest_dists[channel] == DistributionFamily.CATEGORICAL for channel in channel_indices
        ):
            continue
        reference_name = reference_indicator_lookup.get(construct_name)
        reference_channel = manifest_idx.get(reference_name or "")
        if reference_channel in channel_indices:
            manifest_cat_anchor[reference_channel] = True

    unmatched_manifests = sorted(
        set(manifest_cols)
        - matched_manifests
        - set(unknown_likelihoods)
        - invalid_construct_manifests
    )
    if unmatched_manifests:
        errors.append(
            "ModelSpec likelihoods could not be mapped to model measurement indicators: "
            f"{unmatched_manifests}"
        )

    if errors:
        raise NumericalSupportError(errors)

    return (
        state_dynamics_support,
        input_effect_support,
        lambda_mat,
        lambda_support,
        manifest_cat_anchor,
        edge_lag_days,
    )


def build_manifest_variance_from_model(
    latent_names: list[str],
    manifest_cols: list[str],
    manifest_dists: list[DistributionFamily],
    *,
    model: ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Build manifest-noise structure from the retained measurement structure.

    A diagonal manifest-noise entry is free only when both conditions hold:
    - the construct has more than one indicator (single-indicator constructs
      absorb measurement error into the structural residual, so their manifest
      channels get fixed zero observation noise), and
    - the indicator's observation family actually reads per-channel noise in
      its emission log-prob (see ``DistributionFamily.uses_manifest_noise``).
      Non-Gaussian, non-Student-t families (Poisson, Gamma, Bernoulli,
      Negative-Binomial, Beta, Ordered-Logistic, Categorical) ignore R, so a
      free noise site would be a disconnected parameter.
    """
    n_manifest = len(manifest_cols)
    empty_variance = np.zeros((n_manifest, n_manifest))
    family_noise_mask = np.array(
        [dist.uses_manifest_noise for dist in manifest_dists],
        dtype=bool,
    )

    indicators = get_indicators(model)
    latent_name_set = set(latent_names)
    manifest_idx = {name: idx for idx, name in enumerate(manifest_cols)}
    manifest_to_construct: dict[str, str] = {}
    indicators_per_construct: dict[str, int] = {}

    for indicator in indicators:
        ind_name = indicator.get("name") if isinstance(indicator, dict) else indicator.name
        construct_name = indicator["construct_name"]
        if ind_name not in manifest_idx or construct_name not in latent_name_set:
            continue
        manifest_to_construct[ind_name] = construct_name
        indicators_per_construct[construct_name] = (
            indicators_per_construct.get(construct_name, 0) + 1
        )

    if not manifest_to_construct:
        return empty_variance, family_noise_mask

    manifest_var_mask = np.zeros(n_manifest, dtype=bool)
    for manifest_name, construct_name in manifest_to_construct.items():
        if indicators_per_construct.get(construct_name, 0) <= 1:
            continue
        idx = manifest_idx[manifest_name]
        if not manifest_dists[idx].uses_manifest_noise:
            continue
        manifest_var_mask[idx] = True

    manifest_var = np.zeros((n_manifest, n_manifest), dtype=np.float64)
    return manifest_var, manifest_var_mask


def build_manifest_level_counts_from_model(
    manifest_cols: list[str],
    manifest_dists: list[DistributionFamily],
    *,
    model: ModelSpec,
) -> list[int] | None:
    """Build per-manifest discrete level counts from causal-design metadata."""

    needs_level_metadata = any(
        dist in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        for dist in manifest_dists
    )
    if not needs_level_metadata:
        return None

    indicator_lookup = {
        (indicator.get("name") if isinstance(indicator, dict) else indicator.name): indicator
        for indicator in get_indicators(model)
    }
    level_counts = [0] * len(manifest_cols)
    errors: list[str] = []

    for idx, (manifest_name, dist) in enumerate(zip(manifest_cols, manifest_dists, strict=False)):
        if dist not in {
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
        }:
            continue

        indicator = indicator_lookup.get(manifest_name)
        levels_field = (
            "ordinal_levels"
            if dist == DistributionFamily.ORDERED_LOGISTIC
            else "categorical_levels"
        )
        levels = (
            indicator.get(levels_field)
            if isinstance(indicator, dict)
            else getattr(indicator, levels_field, None)
        )
        if not levels or len(levels) < 2:
            errors.append(
                f"Indicator '{manifest_name}' uses {dist.value} but model is missing "
                f"{levels_field} with at least 2 levels"
            )
            continue
        level_counts[idx] = len(levels)

    if errors:
        raise NumericalSupportError(errors)
    return level_counts


def _build_manifest_intercept_support(
    model: ModelSpec,
    manifest_cols: list[str],
    manifest_standardized: list[bool],
) -> tuple[np.ndarray, list[str]]:
    """Bind only observation intercepts active for the locked likelihood semantics."""
    requested_ids = {
        owner.id
        for parameter in model.parameters
        if model.parameter_context(parameter.id).quantity == SiteKind.MANIFEST_MEANS
        and parameter.value is None
        for owner in model.parameter_context(parameter.id).owners
        if owner.kind == "indicator"
    }
    model_by_name = {indicator.name: indicator for indicator in model.indicators}
    requested = np.array(
        [model_by_name[name].id in requested_ids for name in manifest_cols],
        dtype=bool,
    )
    likelihood_lookup = {indicator.name: indicator.likelihood for indicator in model.indicators}
    eligible = np.zeros(len(manifest_cols), dtype=bool)
    for index, manifest_name in enumerate(manifest_cols):
        likelihood = likelihood_lookup[manifest_name]
        assert likelihood is not None
        eligible[index] = indicator_requires_observation_intercept(
            likelihood.law.family, likelihood.terms.link, standardized=manifest_standardized[index]
        )

    errors = [
        (
            f"Observation intercept 'manifest_mean_{manifest_cols[index]}' is inactive for "
            "the locked likelihood semantics; "
            "fix the channel location through its compiler-owned anchor instead."
        )
        for index in np.flatnonzero(requested & ~eligible)
    ]
    return requested & eligible, errors


def _build_static_factor_structure(
    model: ModelSpec, latent_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Derive baseline-factor incidence and fixed loadings from explicit common causes."""
    from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
    from nof1_causal_lab.artifacts.expressions import linear_coefficient
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups, coefficient_value

    groups = baseline_factor_groups(model)
    factors = [group[0] for group in groups]
    state_index = {identity: index for index, identity in enumerate(model.state_order)}
    loadings = np.zeros((len(latent_names), len(factors)))
    for index, group in enumerate(groups):
        sources = {construct.id for construct in group}
        for source in sources:
            construct = model.get_construct(source)
            assert construct.initial_state is not None
            if coefficient_value(model, construct.initial_state.mean) != 0.0:
                raise ValueError("Marginalized baseline factors require a fixed zero initial mean")
            if (
                construct.indicators
                or construct.temporal_status != "time_invariant"
                or any(edge.effect.id == source for edge in model.edges)
            ):
                raise ValueError(
                    "A marginalized baseline factor must be an unmeasured time-invariant root"
                )
            for edge in model.edges:
                if edge.cause.id != source or edge.effect.id not in state_index:
                    continue
                weight = 1.0
                if edge.mechanisms:
                    if len(edge.mechanisms) != 1:
                        raise ValueError(
                            "A marginalized factor loading requires one fixed linear coefficient"
                        )
                    coefficient = linear_coefficient(edge.mechanisms[0].expression, source)
                    if not isinstance(coefficient, FixedCoefficient):
                        raise ValueError(
                            "A marginalized factor loading requires one fixed linear coefficient"
                        )
                    weight = coefficient.value
                loadings[state_index[edge.effect.id], index] = weight
        if not np.any(loadings[:, index]):
            raise ValueError("A baseline factor must affect a retained state")
    return (
        np.ones(len(factors), dtype=bool),
        np.zeros(len(factors)),
        np.asarray(loadings),
        [factor.name for factor in factors],
    )
