"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.model_structure import dependency_id

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def get_state_ids(model: ModelSpec) -> list[ConstructId]:
    return list(model.state_order)


def get_state_names(model: ModelSpec) -> list[str]:
    return [model._constructs[source_id].name for source_id in model.state_order]


def get_constructs(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {"source_id": source_id, **construct.model_dump(mode="json")}
        for source_id, construct in model._constructs.items()
    ]


def get_manifest_indicators(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": source_id,
            **model.indicator(source_id).model_dump(mode="json", exclude={"likelihood"}),
            "construct_id": model.indicator_owner(source_id).id,
            "construct_name": model.indicator_owner(source_id).name,
        }
        for source_id in model.manifest_indicator_order
    ]


def get_reference_indicator_lookup(model: ModelSpec) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    return {
        model._constructs[construct_id].name: (model._indicators[indicator_id].name)
        for construct_id, indicator_id in model.reference_indicator_ids.items()
    }


def get_reference_indicator_polarities(model: ModelSpec) -> dict[str, str]:
    """Return retained construct name to its planned reference polarity."""
    return {
        model._constructs[construct_id].name: (model._indicators[indicator_id].construct_polarity)
        for construct_id, indicator_id in model.reference_indicator_ids.items()
    }


def get_marginalized_scales(  # noqa: V103 - scientific scale projection consumed by offline model templates
    model: ModelSpec,
) -> list[UncheckedJsonObject]:
    """Return identifiable marginalized-confounder scale equivalence classes."""
    footprint_by_confounder: dict[ConstructId, set[str]] = defaultdict(set)
    kind_by_confounder: dict[ConstructId, str] = {}
    directions_by_confounder: dict[ConstructId, list[tuple[str, str]]] = defaultdict(list)
    dependency_ids_by_confounder: dict[ConstructId, list[str]] = defaultdict(list)

    for key, sources in model.induced_dependencies.items():
        first, second, kind = key
        between = model.get_construct(first).name, model.get_construct(second).name
        identity = dependency_id(key, sources)
        for source_id in sources:
            footprint_by_confounder[source_id].update(between)
            directions_by_confounder[source_id].append(between)
            dependency_ids_by_confounder[source_id].append(identity)
            kind_by_confounder[source_id] = kind

    members_by_footprint: dict[tuple[str, frozenset[str]], list[ConstructId]] = defaultdict(list)
    for source_id, footprint in footprint_by_confounder.items():
        members_by_footprint[(kind_by_confounder[source_id], frozenset(footprint))].append(
            source_id
        )

    scales: list[UncheckedJsonObject] = []
    for (kind, footprint), source_ids in sorted(
        members_by_footprint.items(),
        key=lambda item: (
            item[0][0],
            sorted(item[0][1]),
            sorted(item[1]),
        ),
    ):
        source_ids = sorted(source_ids)
        source_names = sorted(model.get_construct(source_id).name for source_id in source_ids)
        directions: set[tuple[str, str]] = set()
        dependency_ids: set[str] = set()
        for source_id in source_ids:
            directions.update(directions_by_confounder[source_id])
            dependency_ids.update(dependency_ids_by_confounder[source_id])
        scales.append(
            {
                "parameter": "tau_" + "__".join(source_names),
                "kind": kind,
                "source_ids": source_ids,
                "sources": source_names,
                "affected_states": sorted(footprint),
                "directions": sorted(directions),
                "dependency_ids": sorted(dependency_ids),
            }
        )
    return scales


def get_model_clock(model: ModelSpec) -> str:
    model.require_measurements()
    assert model.measurement_clock is not None
    return model.measurement_clock
