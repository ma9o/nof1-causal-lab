"""Conditioning provenance of current model laws."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, computed_field

from nof1_causal_lab.artifacts.base import Value

from .identity import GitOid


class AuthoredLawProvenance(Value):
    """The current laws have authored ancestry without retained fitting."""

    kind: Literal["authored"] = "authored"

    @computed_field
    @property
    def interpretation(self) -> Literal["prior_predictive"]:
        return "prior_predictive"


class FittedLawProvenance(Value):
    """All current laws retain one committed fit's model and observation panel."""

    kind: Literal["fitted"] = "fitted"
    fitted_panel_revision: GitOid
    fitted_model_revision: GitOid
    interpretation: Literal[
        "in_sample_posterior_predictive",
        "posterior_predictive",
    ]


class MixedLawProvenance(Value):
    """Some laws retain a committed fit and others have different ancestry."""

    kind: Literal["mixed"] = "mixed"
    fitted_panel_revision: GitOid
    fitted_model_revision: GitOid

    @computed_field
    @property
    def interpretation(self) -> Literal["mixed"]:
        return "mixed"


class UnknownLawProvenance(Value):
    """Imported laws do not establish a conditioning history."""

    kind: Literal["unknown"] = "unknown"

    @computed_field
    @property
    def interpretation(self) -> Literal["unknown"]:
        return "unknown"


type PredictiveLawProvenance = Annotated[
    AuthoredLawProvenance | FittedLawProvenance | MixedLawProvenance | UnknownLawProvenance,
    Field(discriminator="kind"),
]
