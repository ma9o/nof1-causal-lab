"""Measurement validation findings and empirical profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, Self

from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    computed_field,
    model_validator,
)

from .base import Value
from .checks import Evaluated, SpecificationAssessment
from .identity import IndicatorId


class ValidationIssue(Value):
    """A validation issue explains a data problem and its severity for an indicator or the
    dataset.
    """

    indicator_id: IndicatorId | None = Field(
        default=None, description="Affected indicator; null for a dataset-wide issue."
    )
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str


class IndicatorEmpiricalProfile(Value):
    """Count, recorded range, quartiles and mean of one indicator's usable observations."""

    n_obs: int
    min: float | None
    q25: float | None
    q50: float | None
    q75: float | None
    max: float | None
    mean: float | None


class IndicatorAudit(Value):
    """An indicator audit combines its empirical data profile with the results of validation
    checks.
    """

    profile: IndicatorEmpiricalProfile | None = None
    issues: tuple[ValidationIssue, ...]
    checks: Mapping[str, Literal["ok", "warning", "error", "not_evaluated"]]

    def with_source(self, source: Self) -> Self:
        """Compose retained empirical findings with model-dependent findings."""
        return self.revised(
            profile=source.profile,
            issues=(*source.issues, *self.issues),
            checks={**source.checks, **self.checks},
        )

    def without_data(self) -> Self:
        """Record unavailable observations without revising the published audit."""
        return self.revised(checks={**self.checks, "data_availability": "not_evaluated"})

    def with_issue(self, issue: ValidationIssue) -> Self:
        """Append a finding as a new audit, preserving the old value."""
        return self.revised(issues=(*self.issues, issue))


class DataProfileArtifact(Value):
    """Model-independent empirical measurements and data-quality findings."""

    indicators: Mapping[IndicatorId, IndicatorAudit]
    dataset_issues: tuple[ValidationIssue, ...]

    def for_indicators(self, identities: frozenset[IndicatorId]) -> Self:
        return self.revised(
            indicators={
                identity: audit
                for identity, audit in self.indicators.items()
                if identity in identities
            }
        )

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Whether the data findings contain no errors, independent of any model."""
        return not (
            any(issue.severity == "error" for issue in self.dataset_issues)
            or any(
                any(issue.severity == "error" for issue in audit.issues)
                or "error" in audit.checks.values()
                for audit in self.indicators.values()
            )
        )

    @model_validator(mode="wrap")
    @classmethod
    def validate_serialized_verdict(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Verify the derived verdict when reading a serialized report."""
        return _validate_serialized_verdict(value, handler)


class ValidationReportArtifact(Value):
    """Data findings composed with model-dependent execution checks."""

    data: DataProfileArtifact
    preflight: tuple[SpecificationAssessment, ...] = Field(default_factory=lambda: ())

    def for_indicators(self, identities: frozenset[IndicatorId]) -> Self:
        return self.revised(data=self.data.for_indicators(identities))

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Whether both the data findings and model preflight contain no failures."""
        return self.data.is_valid and not any(
            isinstance(finding, Evaluated) and finding.outcome in {"failed", "error"}
            for finding in self.preflight
        )

    @model_validator(mode="wrap")
    @classmethod
    def validate_serialized_verdict(
        cls, value: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        return _validate_serialized_verdict(value, handler)


class _ReportWithVerdict(Protocol):
    @property
    def is_valid(self) -> bool: ...


def _validate_serialized_verdict[ReportT: _ReportWithVerdict](
    value: object, handler: ModelWrapValidatorHandler[ReportT]
) -> ReportT:
    """Check serialized computed verdicts at each report's own contract boundary."""
    if isinstance(value, dict) and "is_valid" in value:
        payload = dict(value)
        verdict = payload.pop("is_valid")
        report = handler(payload)
        if verdict is not report.is_valid:
            raise ValueError("is_valid must match the verdict derived from the report findings")
        return report
    return handler(value)
