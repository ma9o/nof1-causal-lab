"""Scientific dynamics declarations, independent of labels and execution axes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .identity import ConstructId, EdgeId, ParameterId  # noqa: TC001


class FixedCoefficient(BaseModel):
    """A coefficient held at a specified value on the continuous-time model scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["fixed"] = "fixed"
    value: FiniteFloat


class EstimatedCoefficient(BaseModel):
    """A free coefficient referencing its scientific parameter definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["estimated"] = "estimated"
    parameter_id: ParameterId


type MechanismCoefficient = Annotated[
    FixedCoefficient | EstimatedCoefficient, Field(discriminator="kind")
]


class NodePotentialMechanism(BaseModel):
    """Restoring drift -stiffness * (x - center) - quartic * (x - center)^3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["node_potential"] = "node_potential"
    target_id: ConstructId
    center: MechanismCoefficient
    stiffness: MechanismCoefficient
    quartic: MechanismCoefficient


class ConstantDriftMechanism(BaseModel):
    """An additive continuous-time forcing of a state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["constant_drift"] = "constant_drift"
    target_id: ConstructId
    intercept: EstimatedCoefficient


class LinearEdgeMechanism(BaseModel):
    """A directed effect proportional to the source state or known input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["linear"] = "linear"
    edge_id: EdgeId
    weight: EstimatedCoefficient


class HillEdgeMechanism(BaseModel):
    """A directed saturating effect of a state through the native Hill response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hill"] = "hill"
    edge_id: EdgeId
    emax: MechanismCoefficient
    ec50: MechanismCoefficient
    n: MechanismCoefficient


type DynamicsMechanism = Annotated[
    NodePotentialMechanism | ConstantDriftMechanism | LinearEdgeMechanism | HillEdgeMechanism,
    Field(discriminator="kind"),
]


def mechanism_coefficients(mechanism: DynamicsMechanism) -> dict[str, MechanismCoefficient]:
    """The named coefficient slots declared by a mechanism."""
    match mechanism:
        case NodePotentialMechanism():
            return {
                "center": mechanism.center,
                "stiffness": mechanism.stiffness,
                "quartic": mechanism.quartic,
            }
        case ConstantDriftMechanism():
            return {"intercept": mechanism.intercept}
        case LinearEdgeMechanism():
            return {"weight": mechanism.weight}
        case HillEdgeMechanism():
            return {"emax": mechanism.emax, "ec50": mechanism.ec50, "n": mechanism.n}
