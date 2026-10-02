"""Ephemeral bindings between scientific identities and numerical coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel

from nof1_causal_lab.artifacts.identity import ParameterElementId, ParameterId
from nof1_causal_lab.artifacts.parameter import (
    ParameterCoordinate,
    PriorAuthoringTransform,
    SiteKind,
)


class CompiledParameterBinding(Value):
    """Semantic parameter-to-runtime-site binding."""

    parameter_id: ParameterId
    parameter_name: str
    coordinates: Mapping[ParameterElementId, ParameterCoordinate] = Field(min_length=1)
    elements: Mapping[ParameterElementId, str] = Field(min_length=1)
    site_name: str
    prior_field: str | None
    flat_index: int = Field(ge=0)
    site_kind: SiteKind
    transform: PriorAuthoringTransform
    construct_names: tuple[str, ...]
    indicator_names: tuple[str, ...]
    component_index: int | None
    effect_idx: int | None
    cause_idx: int | None


def parameter_bindings(
    model: CompiledModel,
) -> tuple[tuple[CompiledParameterBinding, ...], tuple[ParameterCoordinate, ...]]:
    """Read scientific bindings already resolved by the compiler."""
    return model.bindings, model.auxiliary_coordinates
