"""Graph and observation helpers for canonical operation input projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from nof1_causal_lab.utils.observation_semantics import (
    get_observation_semantics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.json_types import UncheckedJsonObject


def get_indicator_polarity(indicator: UncheckedJsonObject) -> str:
    """Return the declared indicator polarity, failing loudly when absent."""
    polarity = indicator.get("construct_polarity")
    if polarity not in {"positive", "negative"}:
        raise ValueError(
            f"Indicator {indicator.get('name')!r} is missing a valid construct_polarity"
        )
    return str(polarity)


# How strongly a pinned reference loading anchors the latent scale, by dtype:
# continuous channels standardize (data units pin scale and location), ordinal
# channels pin scale through the fixed logistic link, binary/count pin scale
# through their link but carry less information, and a pinned categorical
# loading anchors nothing because the free class slopes absorb it.
_REFERENCE_DTYPE_TIERS = {
    "continuous": 0,
    "ordinal": 1,
    "binary": 2,
    "count": 2,
    "categorical": 3,
}


def choose_reference_indicator(
    indicators: list[UncheckedJsonObject],
) -> UncheckedJsonObject | None:
    """Choose a deterministic marker indicator for one construct.

    Prefer the dtype whose fixed loading anchors the latent scale most strongly
    (see ``_REFERENCE_DTYPE_TIERS``); within a tier prefer positive polarity so
    the latent orientation matches the construct name, then declaration order.
    """
    if not indicators:
        return None

    def _rank(item: tuple[int, UncheckedJsonObject]) -> tuple[int, int, int]:
        declaration_index, indicator = item
        dtype = str(indicator.get("measurement_dtype") or "")
        tier = _REFERENCE_DTYPE_TIERS.get(dtype, 2)
        polarity_rank = 0 if get_indicator_polarity(indicator) == "positive" else 1
        return (tier, polarity_rank, declaration_index)

    return min(enumerate(indicators), key=_rank)[1]


def get_effective_observation_window(
    indicator: UncheckedJsonObject,
    model_clock: str | None,
) -> str | None:
    """Return the effective support window for an indicator."""
    return indicator.get("observation_window") or model_clock


def get_measurement_indicator_info(
    measurement_structure: UncheckedJsonObject,
) -> dict[str, UncheckedJsonObject]:
    """Extract indicator info from a MeasurementStructure dict."""
    result: dict[str, UncheckedJsonObject] = {}
    model_clock = measurement_structure.get("model_clock")
    for ind in measurement_structure.get("indicators", []):
        sem = get_observation_semantics(ind)
        result[ind["id"]] = {
            "dtype": ind.get("measurement_dtype"),
            "ordinal_levels": ind.get("ordinal_levels"),
            "categorical_levels": ind.get("categorical_levels"),
            "support_kind": sem.support_kind.value,
            "summary_operator": sem.summary_operator.value,
            "anchor_policy": sem.anchor_policy.value,
            "observation_window": get_effective_observation_window(ind, model_clock),
        }
    return result


_WORKER_INDICATOR_KEYS = (
    "id",
    "name",
    "measurement_dtype",
    "how_to_measure",
    "source_columns",
    "aggregation",
    "recording",
    "observation_window",
    "ordinal_levels",
    "categorical_levels",
)


def make_measurement_extraction_context(
    measurement_structure: UncheckedJsonObject,
) -> UncheckedJsonObject:
    """Build minimal context needed by extraction extraction workers.

    Workers need:
    - indicators: name, measurement_dtype, how_to_measure, source_columns,
      aggregation, support_kind, summary_operator, anchor_policy, observation_window

    Does not include: construct_name, latent edges, or non-outcome constructs.
    Includes ordinal_levels only for ordinal indicators so workers can use a
    stable numeric codebook.
    """
    model_clock = measurement_structure.get("model_clock")
    slim_indicators = []
    for ind in measurement_structure.get("indicators", []):
        sem = get_observation_semantics(ind)
        entry = {
            **{k: ind[k] for k in _WORKER_INDICATOR_KEYS if k in ind},
            "support_kind": sem.support_kind.value,
            "summary_operator": sem.summary_operator.value,
            "anchor_policy": sem.anchor_policy.value,
        }
        effective_window = get_effective_observation_window(ind, model_clock)
        if effective_window:
            entry["observation_window"] = effective_window
        slim_indicators.append(entry)
    return {
        "model_clock": model_clock,
        "indicators": slim_indicators,
    }


def get_outcome_construct(
    constructs: Sequence[ConstructSpec],
    default_outcome: ConstructId | None,
) -> ConstructSpec | None:
    """Resolve the selected outcome from its canonical construct identity."""
    if default_outcome is None:
        return None
    return next((item for item in constructs if item.id == default_outcome), None)


def get_outcome_name(
    constructs: Sequence[ConstructSpec], default_outcome: ConstructId | None
) -> str | None:
    """Resolve the selected outcome display name from its construct."""
    outcome = get_outcome_construct(constructs, default_outcome)
    return outcome.name if outcome is not None else None


# ---------------------------------------------------------------------------
# Graph utilities (merged from effects.py)
# ---------------------------------------------------------------------------


def build_digraph(
    constructs: Sequence[ConstructSpec], edges: Sequence[CausalEdgeSpec]
) -> nx.DiGraph:
    """Build a directed graph including constructs with no incident edges."""
    graph = nx.DiGraph()
    graph.add_nodes_from(construct.name for construct in constructs)
    graph.add_edges_from((edge.cause.name, edge.effect.name) for edge in edges)
    return graph


def build_digraph_from_edges(edges: Sequence[CausalEdgeSpec]) -> nx.DiGraph:
    """Build a simple DiGraph from canonical causal edge endpoints."""
    return build_digraph((), edges)


def get_all_treatments(
    constructs: Sequence[ConstructSpec],
    edges: Sequence[CausalEdgeSpec],
    default_outcome: ConstructId | None,
) -> list[str]:
    """Return construct names with a directed path to the selected outcome."""
    outcome = get_outcome_name(constructs, default_outcome)
    if outcome is None:
        return []
    graph = build_digraph(constructs, edges)
    return sorted(nx.ancestors(graph, outcome))
