"""Derive scalar finding identities from canonical parameters and native bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import (
    scientific_id,
)
from nof1_causal_lab.artifacts.parameter import (
    SiteKind,
)
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ParameterElementId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding, SiteDescriptor


SHARED_OBSERVATION_FAMILIES = {
    SiteKind.OBS_DF: "student_t",
    SiteKind.OBS_SHAPE: "gamma",
    SiteKind.OBS_R: "negative_binomial",
    SiteKind.OBS_CONCENTRATION: "beta",
    SiteKind.OBS_CAT_INTERCEPTS: "categorical",
    SiteKind.OBS_CAT_SLOPES: "categorical",
}


def component_identity(
    parameter: ParameterSpec,
    indices: tuple[int, ...],
    binding: SemanticBinding,
    site: SiteDescriptor,
    spec: ModelSpec,
) -> tuple[ParameterElementId, str] | None:
    """Identify category components by labels and covariance components by their ordered basis.

    Padded likelihood coordinates are execution-only and have no scientific component.
    """
    kind = site.site_kind
    key: object = "scalar"
    label = parameter.name
    if kind in {
        SiteKind.OBS_ORDERED_BASE,
        SiteKind.OBS_ORDERED_GAPS,
        SiteKind.OBS_CAT_INTERCEPTS,
        SiteKind.OBS_CAT_SLOPES,
    }:
        indicators = {item.name: item for item in spec._indicators.values()}
        assert numeric.observation_names(spec) is not None
        indicator = indicators[numeric.observation_names(spec)[indices[0]]]
        levels = (
            indicator.ordinal_levels
            if kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}
            else indicator.categorical_levels
        )
        if levels is None:
            return None
        if kind == SiteKind.OBS_ORDERED_BASE:
            key = [indicator.id, "first_cutpoint", levels[:2]]
            label = f"{indicator.name}: {levels[0]} | {levels[1]}"
        elif kind == SiteKind.OBS_ORDERED_GAPS:
            column = indices[1]
            if column >= len(levels) - 2:
                return None
            key = [indicator.id, "cutpoint_gap", levels[column : column + 3]]
            label = f"{indicator.name}: gap {levels[column]} / {levels[column + 1]} / {levels[column + 2]}"
        else:
            column = indices[1]
            if column >= len(levels) - 1:
                return None
            key = [indicator.id, "category_contrast", levels[column + 1], levels[0]]
            label = f"{indicator.name}: {levels[column + 1]} vs {levels[0]}"
    elif kind in {
        SiteKind.DIFFUSION_LOWER,
        SiteKind.DIFFUSION_DIAG,
        SiteKind.T0_VAR_LOWER,
        SiteKind.T0_VAR_DIAG,
    }:
        # A Cholesky entry is conditional on the preceding ordered basis. Reordering
        # that basis changes the quantity even if the endpoint labels survive.
        constructs = {item.name: item.id for item in spec._constructs.values()}
        position = site.positions[binding.flat_index]
        row = position[0] if isinstance(position, tuple) else position
        assert numeric.state_names(spec) is not None
        basis = [constructs[name] for name in numeric.state_names(spec)[: row + 1]]
        key = ["cholesky", basis]
    elif any(n > 1 for n in site.shape) and binding.transform.value == "site_wide":
        raise ValueError(f"Parameter {parameter.name!r} needs explicit scientific component axes")
    return scientific_id("element", [parameter.id, key]), label
