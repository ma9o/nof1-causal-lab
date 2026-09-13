"""Scientific parameter IDs bound to native sample sites through explicit ownership."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.ssm.parameterization import build_site_registry
from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec, StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor, SitePosition


class PriorIndexingError(AggregatedCompileError):
    """Aggregate independent scientific-ownership binding failures."""

    header = "Prior index binding failed"


@dataclass(frozen=True)
class SemanticBindingRegistry:
    """Parameter-ID keyed bindings; runtime aliases are display metadata only."""

    by_parameter: dict[ParameterId, SemanticBinding] = field(default_factory=dict)


def empty_prior_bindings() -> SemanticBindingRegistry:
    return SemanticBindingRegistry()


def _axis(ids: list[str] | None, size: int, label: str) -> dict[str, int]:
    if ids is None or len(ids) != size or len(set(ids)) != size:
        raise ValueError(f"Scientific prior binding requires {size} explicit unique {label} IDs")
    return {key: index for index, key in enumerate(ids)}


def _owners(parameter: ParameterSpec, kind: str) -> set[str]:
    return {owner.id for owner in parameter.owners if owner.kind == kind}


def _native_dynamics_bindings(
    spec: SSMSpec, model: StatisticalModelSpec, plan: StructuralPlan | None
) -> dict[str, SemanticBinding]:
    """Bind an explicitly supplied native component composition by quantity and axis IDs."""
    from nof1_causal_lab.models.ssm.dynamics.spec import iter_dynamics_semantic_bindings

    axis = _axis(spec.latent_ids, spec.n_latent, "latent")
    if spec.latent_names is None:
        raise ValueError("Native semantic bindings require latent labels alongside their IDs")
    axis_ids = list(axis)
    sites = {site.name: site for site in build_site_registry(spec)}

    def owner_ids(binding: SemanticBinding) -> set[str]:
        position = sites[binding.site_name].positions[binding.flat_index]
        indices = (position,) if isinstance(position, int) else position
        return {axis_ids[index] for index in indices}

    def matches_edge(binding: SemanticBinding, parameter: ParameterSpec) -> bool:
        if plan is None or not _owners(parameter, "edge"):
            return True
        if binding.cause_idx is None or binding.effect_idx is None:
            return False
        edge_ids = {
            edge.id
            for edge in plan.semantics.edges.values()
            if edge.cause_id == axis_ids[binding.cause_idx]
            and edge.effect_id == axis_ids[binding.effect_idx]
        }
        return _owners(parameter, "edge") == edge_ids

    candidates = list(
        iter_dynamics_semantic_bindings(spec.dynamics_spec, latent_names=tuple(spec.latent_names))
    )
    result = {}
    for parameter in model.parameters:
        matches = {
            (binding.site_name, binding.flat_index): binding
            for binding in candidates
            if binding.site_kind == parameter.quantity
            and owner_ids(binding) == _owners(parameter, "construct")
            and matches_edge(binding, parameter)
        }
        if len(matches) > 1:
            raise ValueError(f"Parameter {parameter.id!r} targets multiple native dynamics sites")
        if matches:
            result[parameter.id] = replace(
                next(iter(matches.values())),
                parameter_name=parameter.name,
                transform=parameter.prior_transform,
            )
    return result


def build_semantic_prior_bindings(
    ssm_spec: SSMSpec,
    statistical_model_spec: StatisticalModelSpec,
    *,
    structural_plan: StructuralPlan | None = None,
) -> SemanticBindingRegistry:
    """Bind by mechanism coefficient references, quantities, and scientific owner IDs."""
    from nof1_causal_lab.models.ssm.compile.parameter_identity import SHARED_OBSERVATION_FAMILIES

    bindings = _native_dynamics_bindings(ssm_spec, statistical_model_spec, structural_plan)
    latent = _axis(ssm_spec.latent_ids, ssm_spec.n_latent, "latent")
    manifest = _axis(ssm_spec.manifest_ids, ssm_spec.n_manifest, "manifest")
    inputs = _axis(ssm_spec.input_ids, ssm_spec.input_effect_block.n_cols, "input")
    sites = build_site_registry(ssm_spec)
    errors: list[str] = []
    latent_names = ssm_spec.latent_names
    manifest_names = ssm_spec.manifest_names
    assert latent_names is not None
    assert manifest_names is not None

    for parameter in statistical_model_spec.parameters:
        if parameter.id in bindings:
            continue
        kind = parameter.quantity
        matches: list[tuple[SiteDescriptor, int]] = []
        construct_ids = _owners(parameter, "construct")
        indicator_ids = _owners(parameter, "indicator")
        state_indices = {latent[key] for key in construct_ids if key in latent}
        indicator_indices = {manifest[key] for key in indicator_ids if key in manifest}
        input_indices = {inputs[key] for key in construct_ids if key in inputs}
        position: SitePosition | None = None
        transform = parameter.prior_transform
        if kind in SHARED_OBSERVATION_FAMILIES:
            matches = [(site, 0) for site in sites if site.site_kind == kind]
            transform = PriorAuthoringTransform.SITE_WIDE
        elif kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}:
            if len(indicator_indices) == 1:
                matches = [
                    (site, next(iter(indicator_indices)))
                    for site in sites
                    if site.site_kind == kind
                ]
            transform = (
                PriorAuthoringTransform.SITE_ROW
                if kind == SiteKind.OBS_ORDERED_GAPS
                else PriorAuthoringTransform.IDENTITY
            )
        else:
            if kind == SiteKind.STATIC_STATE_SD:
                if (
                    ssm_spec.static_factor_ids is not None
                    and parameter.id in ssm_spec.static_factor_ids
                ):
                    position = ssm_spec.static_factor_ids.index(parameter.id)
            elif kind == SiteKind.LOADING:
                if len(indicator_indices) == 1 and len(state_indices) == 1:
                    position = (next(iter(indicator_indices)), next(iter(state_indices)))
            elif kind == SiteKind.INPUT_EFFECT:
                if len(state_indices) == 1 and len(input_indices) == 1:
                    position = (next(iter(state_indices)), next(iter(input_indices)))
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
                    (site, index)
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
        site, flat_index = matches[0]
        bindings[parameter.id] = SemanticBinding(
            parameter_name=parameter.name,
            site_name=site.name,
            prior_field=site.priors_field,
            flat_index=flat_index,
            site_kind=kind,
            transform=transform,
            construct_names=tuple(latent_names[index] for index in sorted(state_indices))
            + tuple((ssm_spec.input_names or [])[index] for index in sorted(input_indices)),
            indicator_names=tuple(manifest_names[index] for index in sorted(indicator_indices)),
            effect_idx=next(iter(state_indices)) if kind == SiteKind.INPUT_EFFECT else None,
        )
    if errors:
        raise PriorIndexingError(errors)
    return SemanticBindingRegistry(bindings)


__all__ = [
    "PriorAuthoringTransform",
    "PriorIndexingError",
    "SemanticBinding",
    "SemanticBindingRegistry",
    "build_semantic_prior_bindings",
    "empty_prior_bindings",
]
