"""Ephemeral bindings between scientific identities and numerical coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, assert_never

import numpy as np
from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ParameterElementId, ParameterId
from nof1_causal_lab.artifacts.parameter import (
    ParameterCoordinate,
    PriorAuthoringTransform,
    SiteKind,
)
from nof1_causal_lab.models.ssm.structure.sites import (
    RowSiteSelection,
    ScalarSiteSelection,
    WholeSiteSelection,
    site_size,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor, SiteSelection


class CompiledSiteCoordinate(Value):
    """A native coordinate and its resolved offset in the site's flattened array."""

    coordinate: ParameterCoordinate
    flat_index: int = Field(ge=0)


def resolve_site_selection(
    site: SiteDescriptor, selection: SiteSelection
) -> tuple[CompiledSiteCoordinate, ...]:
    """Resolve a compiler-owned selection once, before binding or prior attachment."""
    match selection:
        case ScalarSiteSelection(flat_index):
            indices = (flat_index,)
        case RowSiteSelection(row):
            indices = tuple(
                int(np.ravel_multi_index((row, column), site.shape))
                for column in range(site.shape[1])
            )
        case WholeSiteSelection():
            indices = tuple(range(site_size(site.shape)))
        case _:
            assert_never(selection)
    return tuple(
        CompiledSiteCoordinate(
            coordinate=ParameterCoordinate(
                site_name=site.name,
                indices=tuple(int(index) for index in np.unravel_index(flat_index, site.shape)),
            ),
            flat_index=flat_index,
        )
        for flat_index in indices
    )


class CompiledParameterBinding(Value):
    """Semantic parameter-to-runtime-site binding."""

    parameter_id: ParameterId
    parameter_name: str
    coordinates: Mapping[ParameterElementId, ParameterCoordinate] = Field(min_length=1)
    elements: Mapping[ParameterElementId, str] = Field(min_length=1)
    native_coordinates: tuple[CompiledSiteCoordinate, ...] = Field(min_length=1)
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
