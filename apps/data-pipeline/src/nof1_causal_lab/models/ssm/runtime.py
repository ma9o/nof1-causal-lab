"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import polars as pl

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.planning import (
    InferenceStructurePlan,
    plan_inference_structure,
)
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.observation_support import (
    ObservationSupportRuntime,
    augment_wide_data_with_support_boundaries,
    compile_observation_support_runtime,
    validate_discrete_manifest_metadata,
    validate_observation_support,
)
from nof1_causal_lab.models.ssm.parameterization import (
    build_prior_runtime_bundle,
)
from nof1_causal_lab.utils.data import pivot_to_wide

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
    from nof1_causal_lab.sampler_config import (
        SamplerConfig,
        SamplerConfigInput,
    )

logger = logging.getLogger(__name__)


def _standardize_manifest_columns(
    wide_data: pl.DataFrame,
    manifest_cols: list[str],
    manifest_standardized: list[bool] | None,
) -> pl.DataFrame:
    """Apply deterministic standardization to manifest columns marked standardized.

    Flagged columns become (y - mean) / sd so their link-scale spread is exactly 1,
    matching the standardized-latent convention the priors are authored under. When
    sd is 0 or undefined every centered value is already 0, so any divisor yields
    identical data; 1 is the canonical completion, not a fallback.
    """
    if manifest_standardized is None or not any(manifest_standardized):
        return wide_data

    standardized_exprs = []
    for manifest_name, standardized in zip(manifest_cols, manifest_standardized, strict=False):
        base_expr = pl.col(manifest_name).cast(pl.Float64, strict=False)
        if standardized:
            centered_expr = base_expr - base_expr.mean()
            scale_expr = base_expr.std()
            standardized_exprs.append(
                (
                    centered_expr
                    / pl.when(scale_expr > 0.0).then(scale_expr).otherwise(pl.lit(1.0))
                ).alias(manifest_name)
            )
        else:
            standardized_exprs.append(base_expr.alias(manifest_name))
    passthrough = [col_name for col_name in wide_data.columns if col_name not in set(manifest_cols)]
    return wide_data.select(*passthrough, *standardized_exprs)


@dataclass
class PreparedModelRuntime:
    """Canonical prepared runtime context shared by validation and inference."""

    model: SSMModel
    sampler_config: SamplerConfigInput
    wide_data: pl.DataFrame
    observation_data: pl.DataFrame | None
    observation_support: ObservationSupportRuntime | None
    inference_structure: InferenceStructurePlan
    observations: jnp.ndarray
    times: jnp.ndarray

    @property
    def spec(self) -> ModelSpec:
        return self.model.spec

    @property
    def parameter_layout(self) -> SSMParameterLayout:
        return self.model.parameter_layout


def get_default_sampler_config() -> SamplerConfig:
    """Return default sampler configuration from config.yaml."""
    from nof1_causal_lab.utils.config import get_config

    return get_config().inference.to_sampler_config()


def build_ssm_model(wide_data: pl.DataFrame, *, model_spec: ModelSpec) -> SSMModel:
    """Derive execution inputs directly from the scientific ModelSpec."""
    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model

    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")
    validate_discrete_manifest_metadata(model_spec, wide_data)
    validate_observation_support(model_spec, wide_data)
    priors, _, _, _, _ = compile_ssm_inputs_from_model(model_spec)
    return SSMModel(
        model_spec, priors, prior_runtime_bundle=build_prior_runtime_bundle(model_spec, priors)
    )


def prepare_fit_inputs(
    spec: ModelSpec,
    wide_data: pl.DataFrame,
) -> tuple[jnp.ndarray, jnp.ndarray, list[str], pl.DataFrame]:
    """Extract observations, times, manifest order, and standardized wide data."""
    manifest_cols = numeric.observation_names(spec)
    manifest_standardized = (
        list(numeric.observation_standardized(spec))
        if numeric.observation_standardized(spec) is not None
        else None
    )
    standardized_data = _standardize_manifest_columns(
        wide_data, manifest_cols, manifest_standardized
    )
    observations = jnp.array(standardized_data.select(manifest_cols).to_numpy(), dtype=jnp.float32)
    if "time" in standardized_data.columns:
        times = jnp.array(standardized_data["time"].to_numpy(), dtype=jnp.float32)
    else:
        times = jnp.arange(standardized_data.height, dtype=jnp.float32)
    return observations, times, manifest_cols, standardized_data


def prepare_wide_model_runtime(
    wide_data: pl.DataFrame,
    *,
    model_spec: ModelSpec,
    sampler_config: SamplerConfigInput | None = None,
    model: SSMModel | None = None,
    observation_data: pl.DataFrame | None = None,
) -> PreparedModelRuntime:
    """Build or reuse an ``SSMModel`` and extract fit-ready arrays."""
    resolved_sampler_config = sampler_config or get_default_sampler_config()
    if model is None:
        model = build_ssm_model(wide_data, model_spec=model_spec)

    spec = model.spec
    manifest_names = numeric.observation_names(spec)
    wide_data = augment_wide_data_with_support_boundaries(
        observation_data,
        wide_data,
        manifest_names,
    )
    observations, times, manifest_names, wide_data = prepare_fit_inputs(spec, wide_data)
    observation_support = compile_observation_support_runtime(
        observation_data,
        wide_data,
        manifest_names,
    )
    model.set_observation_support(observation_support)
    inference_structure = plan_inference_structure(
        spec,
        observation_support=observation_support,
        method_override=resolved_sampler_config.get("method"),
        n_timepoints=int(times.shape[0]),
    )
    if observation_support is not None and observation_support.requires_interval_summary_handling:
        interval_summary_desc = ", ".join(
            f"{name} ({operator})"
            for name, operator, support_kind in zip(
                observation_support.manifest_names,
                observation_support.summary_operators,
                observation_support.support_kinds,
                strict=False,
            )
            if support_kind == "interval" and operator is not None
        )
        logger.info(
            "Prepared runtime compiled support-aware observation semantics for %s.",
            interval_summary_desc,
        )
    return PreparedModelRuntime(
        model=model,
        sampler_config=resolved_sampler_config,
        wide_data=wide_data,
        observation_data=observation_data,
        observation_support=observation_support,
        inference_structure=inference_structure,
        observations=observations,
        times=times,
    )


def prepare_model_runtime(
    data_for_model: pl.DataFrame,
    *,
    model_spec: ModelSpec,
    sampler_config: SamplerConfigInput | None = None,
    model: SSMModel | None = None,
) -> PreparedModelRuntime:
    """Canonical entry point for preparing stage data for model work."""
    wide_data = pivot_to_wide(data_for_model)
    runtime_rows = data_for_model.rename({"indicator_id": "indicator"})
    labels = {indicator.id: indicator.name for indicator in model_spec.indicators}
    unknown = set(data_for_model["indicator_id"].unique()) - labels.keys()
    if unknown:
        raise ValueError(f"Observations reference unknown indicators: {sorted(unknown)}")
    wide_data = wide_data.rename(
        {iid: name for iid, name in labels.items() if iid in wide_data.columns}
    )
    runtime_rows = runtime_rows.with_columns(pl.col("indicator").replace_strict(labels))
    return prepare_wide_model_runtime(
        wide_data,
        model_spec=model_spec,
        sampler_config=sampler_config,
        model=model,
        observation_data=runtime_rows,
    )


def sample_prior_predictive(
    model: SSMModel,
    *,
    samples: int = 500,
    times: jnp.ndarray | None = None,
    observation_support: ObservationSupportRuntime | None = None,
    observation_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a live model and optional prepared schedule."""
    from nof1_causal_lab.models.ssm.execution.observation_families import (
        any_family_needs_level_metadata,
    )
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        sample_prior_predictive_from_runtime,
    )

    spec = model.spec
    if (
        any_family_needs_level_metadata(numeric.observation_families(spec))
        and numeric.observation_level_counts(spec) is None
    ):
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts. Build the model with data first."
        )

    if times is None:
        times = jnp.arange(10, dtype=jnp.float32)
    return sample_prior_predictive_from_runtime(
        spec,
        model.get_prior_runtime_bundle(),
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        num_samples=samples,
    )
