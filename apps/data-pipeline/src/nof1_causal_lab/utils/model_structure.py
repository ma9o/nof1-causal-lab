"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.model_structure import dependency_id

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def get_state_ids(model: ModelSpec) -> list[str]:
    return list(model.state_order)


def get_state_names(model: ModelSpec) -> list[str]:
    return [model._constructs[source_id].name for source_id in model.state_order]


def get_constructs(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {"source_id": source_id, **construct.model_dump(mode="json")}
        for source_id, construct in model._constructs.items()
    ]


def get_indicators(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": source_id,
            **indicator.model_dump(mode="json", exclude={"likelihood"}),
            "construct_id": model.indicator_owner(indicator.id).id,
            "construct_name": model.indicator_owner(indicator.id).name,
        }
        for source_id, indicator in model._indicators.items()
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


def get_edges(model: ModelSpec) -> list[UncheckedJsonObject]:
    result: list[UncheckedJsonObject] = []
    for edge in model.execution_edges:
        result.append(
            {
                "source_id": edge.id,
                "cause_id": edge.cause.id,
                "effect_id": edge.effect.id,
                "cause": model._constructs[edge.cause.id].name,
                "effect": model._constructs[edge.effect.id].name,
                "description": edge.description,
                "lagged": edge.lagged,
                "sources": [source.model_dump(mode="json") for source in edge.sources],
            }
        )
    return result


def get_known_inputs(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {
            "construct_id": identity,
            "source_indicator_id": usage.source_indicator_id,
            "construct": model.get_construct(identity).name,
            "source_indicator": model.indicator(usage.source_indicator_id).name,
            "scale": usage.scale,
            "missing_policy": usage.missing_policy,
        }
        for identity, usage in model.known_inputs.items()
    ]


def get_induced_dependencies(model: ModelSpec) -> list[UncheckedJsonObject]:
    return [
        {
            "source_id": dependency_id(key, sources),
            "between_ids": list(key[:2]),
            "between": [model.get_construct(identity).name for identity in key[:2]],
            "kind": key[2],
            "source_confounder_ids": list(sources),
            "source_confounders": [model.get_construct(identity).name for identity in sources],
        }
        for key, sources in model.induced_dependencies.items()
    ]


def get_marginalized_scales(
    model: ModelSpec,
) -> list[UncheckedJsonObject]:
    """Return identifiable marginalized-confounder scale equivalence classes."""
    dependencies = get_induced_dependencies(model)
    footprint_by_confounder: dict[str, set[str]] = defaultdict(set)
    source_name_by_id: dict[str, str] = {}
    kind_by_confounder: dict[str, str] = {}
    directions_by_confounder: dict[str, list[tuple[str, str]]] = defaultdict(list)
    dependency_ids_by_confounder: dict[str, list[str]] = defaultdict(list)

    for dependency in dependencies:
        kind = str(dependency["kind"])
        between = tuple(str(name) for name in dependency["between"])
        if len(between) != 2:
            raise ValueError(
                f"Malformed induced dependency {dependency['source_id']!r}: {between!r}"
            )
        for source_id, source_name in zip(
            dependency["source_confounder_ids"],
            dependency["source_confounders"],
            strict=True,
        ):
            source_id = str(source_id)
            source_name_by_id[source_id] = str(source_name)
            footprint_by_confounder[source_id].update(between)
            directions_by_confounder[source_id].append(between)
            dependency_ids_by_confounder[source_id].append(str(dependency["source_id"]))
            prior_kind = kind_by_confounder.setdefault(source_id, kind)
            if prior_kind != kind:
                raise ValueError(
                    f"Confounder source {source_id!r} has inconsistent dependency kinds"
                )

    members_by_footprint: dict[tuple[str, frozenset[str]], list[str]] = defaultdict(list)
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
        source_names = sorted(source_name_by_id[source_id] for source_id in source_ids)
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
