"""Ephemeral bindings between scientific identities and numerical coordinates."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nof1_causal_lab.artifacts.identity import ParameterElementId, ParameterId  # noqa: TC001
from nof1_causal_lab.artifacts.parameter import (
    ParameterCoordinate,  # noqa: TC001
    PriorAuthoringTransform,  # noqa: TC001
    SiteKind,  # noqa: TC001
)


class CompiledParameterBinding(BaseModel):
    """Semantic parameter-to-runtime-site binding."""

    parameter_id: ParameterId
    coordinates: dict[ParameterElementId, ParameterCoordinate] = Field(min_length=1)
    elements: dict[ParameterElementId, str] = Field(min_length=1)
    site_name: str
    prior_field: str | None
    flat_index: int = Field(ge=0)
    site_kind: SiteKind
    transform: PriorAuthoringTransform
    construct_names: list[str]
    indicator_names: list[str]
    component_index: int | None
    effect_idx: int | None
    cause_idx: int | None


def parameter_bindings(model):
    """Derive scalar scientific bindings for this immutable model value."""
    from nof1_causal_lab.models.ssm.compile.prior_compilation import bind_parameters
    from nof1_causal_lab.models.ssm.compile.prior_indexing import build_semantic_prior_bindings

    return bind_parameters(build_semantic_prior_bindings(model), model, model.parameters)
