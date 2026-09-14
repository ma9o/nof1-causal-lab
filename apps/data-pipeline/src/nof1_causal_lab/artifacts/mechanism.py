"""Persistent additive contributions to continuous-time dynamics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .expressions import (
    CallExpression,
    CoefficientExpression,
    Expression,
    walk_expression,
)
from .identity import MechanismId  # noqa: TC001


class DynamicsMechanismSpec(BaseModel):
    """A symbolic specification of an additive drift term or a node potential.

    A node potential contributes its negative gradient to the drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: MechanismId
    kind: Literal["drift", "potential"] = "drift"
    expression: Expression

    @model_validator(mode="after")
    def validate_drift_operands(self) -> DynamicsMechanismSpec:
        for node in walk_expression(self.expression):
            if isinstance(node, CoefficientExpression):
                if node.role not in {
                    "center",
                    "decay",
                    "quartic",
                    "intercept",
                    "weight",
                    "emax",
                    "ec50",
                    "exponent",
                }:
                    raise ValueError(
                        f"{node.role} is an observation coefficient, not a drift operand"
                    )
                if node.value is None:
                    raise ValueError("A declared drift contribution requires assigned coefficients")
            if isinstance(node, CallExpression) and node.function in {
                "ordered_cutpoints",
                "category_logits",
            }:
                raise ValueError(f"{node.function} requires an observation category context")
        return self
