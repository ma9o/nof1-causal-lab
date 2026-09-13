"""Measurement validation findings and empirical profiles."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .base import ArtifactPayload
from .identity import EntityRef, IndicatorId


class ValidationIssue(BaseModel):
    """A validation issue explains a data problem and its severity for an indicator or the
    dataset.
    """

    model_config = ConfigDict(extra="forbid")

    subject: EntityRef | None = None
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


class IndicatorValidation(BaseModel):
    """Indicator validation records the outcomes and issues from checks on one extracted
    indicator.
    """

    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssue]
    checks: dict[str, Literal["ok", "warning", "error"]]


class IndicatorAudit(BaseModel):
    """An indicator audit combines its empirical data profile with the results of validation
    checks.
    """

    model_config = ConfigDict(extra="forbid")

    profile: IndicatorEmpiricalProfile | None = None
    validation: IndicatorValidation


class ValidationReportArtifact(ArtifactPayload):
    """A validation report summarizes whether extracted measurements satisfy the required data
    checks.
    """

    is_valid: bool
    indicators: dict[IndicatorId, IndicatorAudit]
    dataset_issues: list[ValidationIssue]
