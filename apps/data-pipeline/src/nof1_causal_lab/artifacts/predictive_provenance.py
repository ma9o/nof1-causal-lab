"""Conditioning provenance of current model laws."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .identity import GitOid  # noqa: TC001


class PredictiveLawProvenance(BaseModel):
    """Known conditioning history, independently of a probability law's family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["authored", "fitted", "mixed", "unknown"]
    fitted_panel_revision: GitOid | None = None
    interpretation: Literal[
        "prior_predictive",
        "in_sample_posterior_predictive",
        "posterior_predictive",
        "mixed",
        "unknown",
    ]


