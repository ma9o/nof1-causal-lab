"""Submission-time findings, independent of authoring progression."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SpecificationFinding(BaseModel):
    """One model-only check and its current evaluation status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: str
    status: Literal["passed", "failed", "not_evaluated"]
    message: str


class SpecificationReport(BaseModel):
    """Model-only findings; data compatibility has its own paired provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: tuple[SpecificationFinding, ...]
