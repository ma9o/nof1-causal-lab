"""Scientific prior evidence and validation findings."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class PriorRepairScope(Value):
    """Deterministic repair scope for nonlocal prior-validation failures."""

    kind: Literal["dynamics_scc"] = Field(
        description="Repair-scope family for a nonlocal validation failure"
    )
    construct_names: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered latent constructs included in the minimal repair scope",
    )


class PriorPathologyCertificate(Value):
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


class PriorValidationResult(Value):
    """Typed model-spec validation diagnostic."""

    parameter: str = Field(description="Name of the parameter that was validated")
    is_valid: bool = Field(description="Whether the prior passed validation")
    code: str = Field(default="unspecified")
    origin: Literal["compile", "prior_predictive"] = "prior_predictive"
    severity: Literal["error", "warning"] = "error"
    issue: str | None = None
    suggested_adjustment: str | None = None
    related_parameters: tuple[str, ...] = Field(default_factory=tuple)
    compiled_site_name: str | None = None
    compiled_flat_index: int | None = None
    supporting_codes: tuple[str, ...] = Field(default_factory=tuple)
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
    bad_sample_sites: tuple[str, ...] = Field(default_factory=tuple)
    bad_manifest_names: tuple[str, ...] = Field(default_factory=tuple)
    failing_draw_indices: tuple[int, ...] = Field(default_factory=tuple)
    first_bad_time_index: int | None = None
    pathology_certificate: PriorPathologyCertificate | None = None

    def with_parameter_provenance(self, parameters: tuple[str, ...]) -> Self:
        """Attach resolved writer identities without mutating a diagnostic."""
        return self.model_copy(update={"related_parameters": parameters})


__all__ = [
    "PriorPathologyCertificate",
    "PriorRepairScope",
    "PriorValidationResult",
]
