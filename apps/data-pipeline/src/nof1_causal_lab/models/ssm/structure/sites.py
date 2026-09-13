"""Canonical sample-site descriptor and support enums.

Lives in ``structure`` so block specs can return them from
``iter_sites()`` without circular imports with ``parameterization.py``.

The descriptor is the canonical identity of a sample site: name, shape,
support class, semantic kind, assembly grouping, and the various prior
binding keys used by compile-time and runtime layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from nof1_causal_lab.artifacts.parameter import (
    PriorAuthoringTransform,
    SiteKind,
    SupportClass,
)

type SitePosition = int | tuple[int, int] | tuple[int, int, int]


@dataclass(frozen=True)
class SiteDescriptor:
    """Metadata for a single sample site.

    ``positions`` is the free-entry index list owned by the originating
    block: ``list[int]`` for vector-shaped sites, ``list[tuple[int, int]]``
    for matrix-shaped sites. Compile-time consumers use it to translate
    flat-vector parameter indices back to structural ``(i, j)`` positions.
    """

    name: str
    shape: tuple[int, ...]
    support: SupportClass
    assembly_group: str
    site_kind: SiteKind
    positions: tuple[SitePosition, ...] = ()
    deterministic_name: str | None = None
    fixed_spec_field: str | None = None
    priors_field: str | None = None
    runtime_prior_key: str | None = None
    is_runtime_prior_controlled: bool = True


@dataclass(frozen=True)
class SemanticBinding:
    """One semantic model parameter bound to a runtime sample-site component."""

    parameter_name: str
    site_name: str
    flat_index: int
    site_kind: SiteKind
    transform: PriorAuthoringTransform = PriorAuthoringTransform.IDENTITY
    prior_field: str | None = None
    construct_names: tuple[str, ...] = ()
    indicator_names: tuple[str, ...] = ()
    component_index: int | None = None
    effect_idx: int | None = None
    cause_idx: int | None = None


def make_site(
    name: str,
    shape: tuple[int, ...],
    support: SupportClass,
    assembly_group: str,
    site_kind: SiteKind,
    *,
    positions: tuple[SitePosition, ...] = (),
    deterministic_name: str | None = None,
    fixed_spec_field: str | None = None,
    priors_field: str | None = None,
    runtime_prior_key: str | None = None,
) -> SiteDescriptor:
    """Construct the scientific metadata for one authored sample site."""
    return SiteDescriptor(
        name=name,
        shape=shape,
        support=support,
        assembly_group=assembly_group,
        site_kind=site_kind,
        positions=positions,
        deterministic_name=deterministic_name,
        fixed_spec_field=fixed_spec_field,
        priors_field=priors_field,
        runtime_prior_key=runtime_prior_key or name,
        is_runtime_prior_controlled=True,
    )


def site_size(shape: tuple[int, ...]) -> int:
    """Number of scalar elements in a site shape."""
    size = 1
    for d in shape:
        size *= d
    return size
