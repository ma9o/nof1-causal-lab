"""Measurement validation findings and empirical profiles."""

from typing import Literal, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    computed_field,
    model_validator,
)

from .base import ArtifactPayload
from .checks import SpecificationReport
from .identity import IndicatorId


class ValidationIssue(BaseModel):
    """A validation issue explains a data problem and its severity for an indicator or the
    dataset.
    """

    model_config = ConfigDict(extra="forbid")

    indicator_id: IndicatorId | None = Field(
        default=None, description="Affected indicator; null for a dataset-wide issue."
    )
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str


class IndicatorEmpiricalProfile(BaseModel):
    """An empirical profile summarizes an indicator's observed values, coverage, and data-
    quality signals.
    """

    model_config = ConfigDict(extra="forbid")

    measurement_dtype: str | None = None
    n_obs: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    variance: float | None
    time_coverage_ratio: float | None
    max_gap_ratio: float | None
    dtype_violations: int | None = None
    duplicate_pct: float | None = None
    arithmetic_sequence_detected: bool
    n_unparseable_timestamps: int | None = None
    zero_fraction: float | None = None
    is_nonnegative: bool | None = None
    is_unit_interval: bool | None = None
    looks_integer_valued: bool | None = None
    variance_to_mean_ratio: float | None = None


class IndicatorAudit(BaseModel):
    """An indicator audit combines its empirical data profile with the results of validation
    checks.
    """

    model_config = ConfigDict(extra="forbid")

    profile: IndicatorEmpiricalProfile | None = None
    issues: list[ValidationIssue]
    checks: dict[str, Literal["ok", "warning", "error", "not_evaluated"]]


class DataProfileArtifact(ArtifactPayload):
    """Model-independent empirical measurements and data-quality findings."""

    indicators: dict[IndicatorId, IndicatorAudit]
    dataset_issues: list[ValidationIssue]

    def _has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.dataset_issues) or any(
            any(issue.severity == "error" for issue in audit.issues)
            or "error" in audit.checks.values()
            for audit in self.indicators.values()
        )

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Read-only verdict derived from the report's current findings."""
        return not self._has_errors()

    @model_validator(mode="wrap")
    @classmethod
    def validate_serialized_verdict(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Verify the derived verdict when reading a serialized report."""
        if isinstance(value, dict) and "is_valid" in value:
            payload = dict(value)
            verdict = payload.pop("is_valid")
            report = handler(payload)
            if verdict is not report.is_valid:
                raise ValueError("is_valid must match the verdict derived from the report findings")
            return report
        return handler(value)


class ValidationReportArtifact(DataProfileArtifact):
    """Measurement findings augmented with model-dependent execution checks."""

    preflight: SpecificationReport = Field(default_factory=lambda: SpecificationReport(findings=()))

    @override
    def _has_errors(self) -> bool:
        return super()._has_errors() or any(
            finding.status == "failed" for finding in self.preflight.findings
        )
