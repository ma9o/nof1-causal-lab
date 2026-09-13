"""Typed executable prior plans and compiler diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

from .distribution import DistributionSpec
from .identity import ParameterId  # noqa: TC001


class ExecutablePrior(DistributionSpec):
    """One authoring-scale prior consumed by the statistical-model compiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_id: ParameterId
    reference_interval_days: float | None = Field(default=None, gt=0)


class PriorPlan(BaseModel):
    """Complete typed executable priors for a StatisticalModelSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    priors: dict[ParameterId, ExecutablePrior]

    @model_validator(mode="after")
    def validate_parameter_keys(self) -> PriorPlan:
        mismatches = sorted(key for key, prior in self.priors.items() if key != prior.parameter_id)
        if mismatches:
            raise ValueError(
                f"PriorPlan keys must equal their executable prior parameter IDs: {mismatches}"
            )
        return self

    def compiler_payloads(self) -> dict[str, UncheckedJsonObject]:
        """Return compiler-owned fields without agent evidence or presentation metadata."""
        return {
            parameter: prior.model_dump(mode="json") for parameter, prior in self.priors.items()
        }


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
    "ExecutablePrior",
    "PriorPathologyCertificate",
    "PriorPlan",
    "PriorRepairScope",
    "PriorValidationResult",
]
