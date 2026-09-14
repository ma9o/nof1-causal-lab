"""Graph and observation helpers for canonical operation input projections."""

import networkx as nx

from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.utils.observation_semantics import (
    get_observation_semantics,
)


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
            "construct_id": ind["construct_id"],
            "ordinal_levels": ind.get("ordinal_levels"),
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
    graph_input: UncheckedJsonObject,
) -> UncheckedJsonObject | None:
    """Resolve the outcome in a canonical graph input projection."""
    latent = graph_input
    target = latent.get("default_outcome")
    if target is None:
        return None
    return next((item for item in latent["constructs"] if item["id"] == target), None)


def get_outcome_name(graph_input: UncheckedJsonObject) -> str | None:
    """Resolve the selected outcome display name from the graph input."""
    outcome = get_outcome_construct(graph_input)
    return outcome["name"] if outcome else None


# ---------------------------------------------------------------------------
# Graph utilities (merged from effects.py)
# ---------------------------------------------------------------------------


def build_digraph(latent_structure: UncheckedJsonObject) -> nx.DiGraph:
    """Build a simple DiGraph from a latent structure's edge list.

    Args:
        latent_structure: Dict with 'edges' list of {cause, effect} dicts

    Returns:
        nx.DiGraph with one node per referenced construct
    """
    names = {item["id"]: item["name"] for item in latent_structure["constructs"]}
    graph = nx.DiGraph()
    graph.add_nodes_from(names.values())
    graph.add_edges_from(
        (names[edge["cause_id"]], names[edge["effect_id"]]) for edge in latent_structure["edges"]
    )
    return graph


def build_digraph_from_edges(edges: list[UncheckedJsonObject]) -> nx.DiGraph:
    """Build a simple DiGraph from an edge list."""
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["cause"], edge["effect"])
    return G


def _get_treatments_from_graph(
    *,
    node_names: list[str],
    edges: list[UncheckedJsonObject],
    outcome: str | None,
) -> list[str]:
    """Return nodes with a directed path to the outcome within the given graph."""
    if not outcome:
        return []

    G = build_digraph_from_edges(edges)
    G.add_nodes_from(node_names)
    if outcome not in G:
        return []

    return sorted(
        node
        for node in node_names
        if node != outcome and G.has_node(node) and nx.has_path(G, node, outcome)
    )


def get_all_treatments(latent_structure: UncheckedJsonObject) -> list[str]:
    """Get all potential treatments from latent structure.

    A treatment is any construct that has a causal path to the outcome.

    Args:
        latent_structure: Dict with 'constructs' and 'edges'

    Returns:
        Sorted list of treatment construct names
    """
    return _get_treatments_from_graph(
        node_names=[
            construct["name"]
            for construct in latent_structure.get("constructs", [])
            if construct.get("name")
        ],
        edges=[
            {"cause": cause, "effect": effect}
            for cause, effect in build_digraph(latent_structure).edges
        ],
        outcome=get_outcome_name(latent_structure),
    )
