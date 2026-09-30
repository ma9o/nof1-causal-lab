"""Persisted posterior results and sampling metadata."""

from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from nof1_causal_lab.json_types import JsonObject

from .identity import ConstructId
from .posterior_diagnostics import (
    LOODiagnostics,
    PosteriorMarginal,
    PosteriorPair,
)


class FitSettingsSpec(BaseModel):
    """Optional numerical controls applied to the configured particle sampler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_samples: int | None = Field(default=None, ge=1)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    n_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)


class InferenceMetadata(BaseModel):
    """Inference metadata records the sampling method, sample count, and run duration."""

    model_config = ConfigDict(extra="forbid")

    method: str
    n_samples: int
    duration_seconds: float


class PosteriorDrawsInfo(BaseModel):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(ge=1)
    parameter_shapes: dict[str, list[int]]
    state_ids: tuple[ConstructId, ...] = ()
    latent_shape: tuple[int, int] | None = Field(
        default=None, description="Time and state axis lengths per retained latent draw."
    )


# Per-draw and per-time-point arrays among the engine-defined diagnostics.
_DETAIL_DIAGNOSTICS = frozenset(
    {"trace_data", "rank_histograms", "initial_latent_delta", "final_latent_delta"}
)


def _without_detail(value: JsonObject) -> JsonObject:
    return {
        key: _without_detail(item) if isinstance(item, dict) else item
        for key, item in value.items()
        if key not in _DETAIL_DIAGNOSTICS
    }


class InferenceReport(BaseModel):
    """Display findings recorded by an inference transition, separate from ModelSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    time_origin: AwareDatetime | None = Field(
        description="Known calendar instant of model day zero."
    )
    inference_metadata: InferenceMetadata
    inference_diagnostics: JsonObject = Field(
        default_factory=dict,
        description="Engine-reported telemetry for this fit; keys and values are engine-defined.",
    )
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None

    def summary(self) -> Self:
        """The report without per-draw diagnostic arrays or pair samples.

        Snapshots and diffs carry this; the inference report endpoint serves the rest.
        """
        return self.model_copy(
            update={
                "inference_diagnostics": _without_detail(self.inference_diagnostics),
                "posterior_pairs": None,
            }
        )
