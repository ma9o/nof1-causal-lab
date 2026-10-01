"""Submission-time findings, independent of authoring progression."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .identity import ConstructId


class SpecificationFinding(BaseModel):
    """One model-only check and its current evaluation status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: str
    status: Literal["passed", "failed", "not_evaluated"]
    message: str


class SpecificationReport(BaseModel):
    """Model-only findings; data compatibility has its own paired input references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[SpecificationFinding, ...]


class PredictiveCheckFinding(BaseModel):
    """One measured simulation check, independent of its authoring or simulation context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: str
    construct_id: ConstructId | None = None
    target: str
    value: str
    band: str
    passed: bool | None
    note: str
    reason: str | None = None
