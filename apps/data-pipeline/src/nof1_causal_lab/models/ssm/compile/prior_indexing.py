"""Scientific parameter IDs bound to native sample sites through explicit ownership."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm.compile import support as numeric
from nof1_causal_lab.models.ssm.structure.sites import (
    RowSiteSelection,
    ScalarSiteSelection,
    SemanticBinding,
    WholeSiteSelection,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
    from nof1_causal_lab.models.ssm.structure.sites import (
        SiteDescriptor,
        SitePosition,
        SiteSelection,
    )


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent scientific-ownership binding failures."""

    header = "Prior index binding failed"


@dataclass(frozen=True)
class SemanticBindingRegistry:
    """Parameter-ID keyed bindings; runtime aliases are display metadata only."""

    by_parameter: Mapping[ParameterId, SemanticBinding]


def _native_dynamics_bindings(
    model: ModelSpec, components: tuple[ExpressionComponentSpec, ...]
) -> dict[ParameterId, SemanticBinding]:
    """Bind coefficient references within the native component emitted by their own term."""
    state_ids = numeric.state_ids(model)
    result = {}
    for index, component in enumerate(components):
        for identity, site in component.parameter_sites(f"vf_{index}"):
            if identity in result:
                raise PriorIndexingError(
                    ["One parameter cannot own multiple independent runtime sites"]
                )
            parameter = model.parameter(identity)
            result[identity] = SemanticBinding(
                parameter_name=parameter.name,
                site_name=site.name,
                selection=ScalarSiteSelection(0),
                site_kind=site.site_kind,
                prior_field=site.priors_field,
                construct_names=tuple(
                    model.get_construct(key).name
                    for key in state_ids
                    if key
                    in {component.state_ids[i] for i in component.sources | {component.target}}
                ),
                component_index=index,
                effect_idx=component.target if component.edge_owned else None,
                cause_idx=component.source,
            )
    return result


def build_semantic_prior_bindings(
    model: ModelSpec,
    sites: tuple[SiteDescriptor, ...],
    components: tuple[ExpressionComponentSpec, ...],
    parameters: tuple[ParameterSpec, ...],
) -> SemanticBindingRegistry:
    """Bind by mechanism coefficient references, quantities, and scientific owner IDs."""
    from nof1_causal_lab.models.ssm.compile.parameter_identity import SHARED_OBSERVATION_FAMILIES

    bindings = _native_dynamics_bindings(model, components)
    latent = {identity: index for index, identity in enumerate(numeric.state_ids(model))}
    manifest = {identity: index for index, identity in enumerate(numeric.observation_ids(model))}
    errors: list[str] = []
    latent_names = numeric.state_names(model)
    manifest_names = numeric.observation_names(model)

    for parameter in parameters:
        if parameter.id in bindings:
            continue
        kind = model.parameter_context(parameter.id).quantity
        matches: list[tuple[SiteDescriptor, SiteSelection]] = []
        owners = model.parameter_context(parameter.id).owners
        construct_ids = {owner.id for owner in owners if owner.kind == "construct"}
        indicator_ids = {owner.id for owner in owners if owner.kind == "indicator"}
        state_indices = {latent[key] for key in construct_ids if key in latent}
        indicator_indices = {manifest[key] for key in indicator_ids if key in manifest}
        position: SitePosition | None = None
        if kind in SHARED_OBSERVATION_FAMILIES or kind == SiteKind.PROC_DF:
            matches = [(site, WholeSiteSelection()) for site in sites if site.site_kind == kind]
        elif kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}:
            if len(indicator_indices) == 1:
                matches = [
                    (
                        site,
                        RowSiteSelection(next(iter(indicator_indices)))
                        if kind == SiteKind.OBS_ORDERED_GAPS
                        else ScalarSiteSelection(next(iter(indicator_indices))),
                    )
                    for site in sites
                    if site.site_kind == kind
                ]
        else:
            if kind == SiteKind.STATIC_STATE_SD:
                factor_ids = construct_ids & set(numeric.static_factor_ids(model))
                if len(factor_ids) == 1:
                    position = {
                        identity: index
                        for index, identity in enumerate(numeric.static_factor_ids(model))
                    }[next(iter(factor_ids))]
            elif kind == SiteKind.LOADING:
                if len(indicator_indices) == 1 and len(state_indices) == 1:
                    position = (next(iter(indicator_indices)), next(iter(state_indices)))
            elif kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
                if len(state_indices) == 2:
                    position = (max(state_indices), min(state_indices))
            elif kind in {SiteKind.MANIFEST_MEANS, SiteKind.MANIFEST_VAR_DIAG}:
                if len(indicator_indices) == 1:
                    position = next(iter(indicator_indices))
            elif (
                kind in {SiteKind.DIFFUSION_DIAG, SiteKind.T0_MEANS, SiteKind.T0_VAR_DIAG}
                and len(state_indices) == 1
            ):
                position = next(iter(state_indices))
            if position is not None:
                matches = [
                    (site, ScalarSiteSelection(index))
                    for site in sites
                    if site.site_kind == kind
                    for index, candidate in enumerate(site.positions)
                    if candidate == position
                ]
        if len(matches) != 1:
            errors.append(
                f"Parameter {parameter.name!r} ({parameter.id}, {kind.value}) must bind to "
                f"one active site through its scientific owners; found {len(matches)}"
            )
            continue
        site, selection = matches[0]
        bindings[parameter.id] = SemanticBinding(
            parameter_name=parameter.name,
            site_name=site.name,
            prior_field=site.priors_field,
            selection=selection,
            site_kind=kind,
            construct_names=tuple(latent_names[index] for index in sorted(state_indices)),
            indicator_names=tuple(manifest_names[index] for index in sorted(indicator_indices)),
        )
    if errors:
        raise PriorIndexingError(errors)
    return SemanticBindingRegistry(MappingProxyType(bindings))


__all__ = [
    "PriorIndexingError",
    "SemanticBinding",
    "SemanticBindingRegistry",
    "build_semantic_prior_bindings",
]
