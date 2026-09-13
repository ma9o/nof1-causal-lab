"""Scientific prior evidence, density display points, and validation findings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .evidence import LiteratureSource


class DensityPoint(BaseModel):
    """A density point stores one coordinate of a prior density curve for plotting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float = Field(ge=0)


class PriorSource(LiteratureSource):
    """A prior source records literature evidence used to justify a parameter's prior
    distribution.
    """

    effect_size: str | None = Field(
        default=None, description="Reported effect size if available (e.g., 'r=0.3', 'β=0.2')"
    )
    study_interval_days: float | None = Field(
        default=None,
        description="Observation/measurement interval of this study in days (daily=1, weekly=7, monthly=30)",
    )


class PriorRepairScope(BaseModel):
    """Deterministic repair scope for nonlocal prior-validation failures."""

    kind: Literal["dynamics_scc"] = Field(
        description="Repair-scope family for a nonlocal validation failure"
    )
    construct_names: list[str] = Field(
        default_factory=list,
        description="Ordered latent constructs included in the minimal repair scope",
    )


class PriorPathologyCertificate(BaseModel):
    """Comparable summary of one validation pathology."""

    kind: Literal["nonfinite_samples", "dynamics_stability", "dt_ct_approximation"] = Field(
        description="Stable certificate family for same-scope retry gating"
    )
    primary_score: float = Field(
        ge=0,
        description="Primary severity score. Lower means the pathology improved.",
    )
    secondary_score: float | None = Field(
        default=None,
        ge=0,
        description="Optional tie-break severity score. Lower means the pathology improved.",
    )


class PriorValidationResult(BaseModel):
    """Typed model-spec validation diagnostic."""

    parameter: str = Field(description="Name of the parameter that was validated")
    is_valid: bool = Field(description="Whether the prior passed validation")
    code: str = Field(default="unspecified")
    origin: Literal["compile", "prior_predictive"] = "prior_predictive"
    severity: Literal["error", "warning"] = "error"
    issue: str | None = None
    suggested_adjustment: str | None = None
    related_parameters: list[str] = Field(default_factory=list)
    compiled_site_name: str | None = None
    compiled_flat_index: int | None = None
    supporting_codes: list[str] = Field(default_factory=list)
    repair_scope: PriorRepairScope | None = None
    failure_stage: (
        Literal[
            "compiled_parameters",
            "latent_dynamics",
            "observation_mean",
            "observation_sample",
            "support_violation",
            "model_build",
            "prior_sampling",
            "unknown",
        ]
        | None
    ) = None
    bad_sample_sites: list[str] = Field(default_factory=list)
    bad_manifest_names: list[str] = Field(default_factory=list)
    failing_draw_indices: list[int] = Field(default_factory=list)
    first_bad_time_index: int | None = None
    pathology_certificate: PriorPathologyCertificate | None = None


__all__ = [
    "PriorPathologyCertificate",
    "PriorRepairScope",
    "PriorValidationResult",
]
