"""Interventions, their model-time assignments, and certified readouts of generated histories."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ConfigDict, Field, FiniteFloat

from nof1_causal_lab.artifacts.base import Value

from .duration import Duration
from .identity import ConstructId


class InterventionSpec(Value):
    """Set a state some time after the design's start, then let its dynamics resume."""

    target: ConstructId
    after: Duration | None = Field(
        default=None, description="Offset from the design's start; omitted means at the start."
    )
    value: FiniteFloat


class StateAssignment(Value):
    """A state set at one model time: a resolved intervention or a replayed input reading."""

    target: ConstructId
    time: FiniteFloat = Field(description="Absolute time in model days.")
    value: FiniteFloat


class CausalEffectResult(Value):
    """Causal effects and realized trajectories under the enclosing report's design."""

    model_config = ConfigDict(allow_inf_nan=False)

    outcome: ConstructId
    labels: Mapping[ConstructId, str]
    warnings: tuple[str, ...] = Field(default_factory=tuple)
