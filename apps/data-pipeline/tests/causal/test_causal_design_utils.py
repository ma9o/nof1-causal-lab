"""Tests for ``utils.causal_design`` helpers.

Trivial accessors are exercised through higher-level tests. This file covers
the helpers with real transformation or graph logic:
- ``make_measurement_extraction_context``
- ``build_digraph``
- ``get_outcome_name``
- ``get_all_treatments``
- ModelSpec state and marginalized-scale accessors
"""

import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.utils.causal_design import (
    build_digraph,
    get_all_treatments,
    get_outcome_name,
    make_measurement_extraction_context,
)
from nof1_causal_lab.utils.model_structure import get_state_names
from tests.causal.graph_fixtures import make_graph
from tests.helpers import fixture_entity_id, make_model

pytestmark = pytest.mark.contract


def _full_spec():
    """Minimal valid CausalDesign dict."""
    return {
        "latent": {
            "default_outcome": "construct:bbc87212909e45b9e6c3",
            "constructs": [
                {"id": "construct:6b04dc42c531e7091eb8", "name": "stress", "role": "endogenous"},
                {
                    "id": "construct:bbc87212909e45b9e6c3",
                    "name": "mood",
                    "role": "endogenous",
                },
            ],
            "edges": [
                {
                    "cause_id": "construct:6b04dc42c531e7091eb8",
                    "effect_id": "construct:bbc87212909e45b9e6c3",
                    "id": "edge:923689028b6b177617c2",
                    "description": "Stress affects mood",
                },
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": "indicator:6bde869aba53fb51e0f4",
                    "construct_id": "construct:6b04dc42c531e7091eb8",
                    "name": "pss_score",
                    "construct_polarity": "positive",
                    "measurement_dtype": "continuous",
                    "how_to_measure": "Extract PSS score",
                    "aggregation": "mean",
                },
                {
                    "id": "indicator:e05e217de7f4442abdc5",
                    "construct_id": "construct:bbc87212909e45b9e6c3",
                    "name": "mood_rating",
                    "construct_polarity": "positive",
                    "measurement_dtype": "ordinal",
                    "how_to_measure": "Rate mood 1-5",
                    "aggregation": "last",
                    "ordinal_levels": ["low", "medium", "high"],
                },
            ],
        },
        "estimation": {
            "state_order": ["stress", "mood"],
            "edges": [
                {
                    "cause_id": "construct:6b04dc42c531e7091eb8",
                    "effect_id": "construct:bbc87212909e45b9e6c3",
                    "id": "edge:923689028b6b177617c2",
                    "description": "Stress affects mood",
                }
            ],
            "induced_dependencies": [],
        },
    }


# =============================================================================
# make_measurement_extraction_context
# =============================================================================


class TestMakeMeasurementExtractionContext:
    def test_strips_to_worker_fields(self):
        spec = _full_spec()
        # Add extra fields that workers don't need
        spec["measurement"]["indicators"][0]["aggregation"] = "mean"
        spec["measurement"]["indicators"][0]["construct_id"] = "construct:stress"
        spec["measurement"]["indicators"][0]["source_columns"] = ["pss_col"]
        ctx = make_measurement_extraction_context(spec["measurement"])
        ind = ctx["indicators"][0]
        assert set(ind.keys()) == {
            "id",
            "name",
            "measurement_dtype",
            "how_to_measure",
            "source_columns",
            "aggregation",
            "support_kind",
            "summary_operator",
            "anchor_policy",
            "observation_window",
        }
        assert "construct_id" not in ind
        assert "ordinal_levels" not in ind

    def test_source_columns_included_when_present(self):
        spec = _full_spec()
        spec["measurement"]["indicators"][0]["source_columns"] = ["col_a", "col_b"]
        ctx = make_measurement_extraction_context(spec["measurement"])
        assert ctx["indicators"][0]["source_columns"] == ["col_a", "col_b"]

    def test_ordinal_levels_included_for_worker_codebook(self):
        spec = _full_spec()
        ctx = make_measurement_extraction_context(spec["measurement"])
        assert ctx["indicators"][1]["ordinal_levels"] == ["low", "medium", "high"]


class TestBuildDigraph:
    @pytest.mark.parametrize(
        ("names", "pairs", "expected_edges"),
        [
            (["A", "B", "C"], [("A", "B"), ("B", "C")], {("A", "B"), ("B", "C")}),
            ([], [], set()),
            (
                ["A", "B", "C", "D"],
                [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
                {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")},
            ),
            (["A"], [("A", "A")], {("A", "A")}),
            (["A", "B"], [("A", "B"), ("A", "B")], {("A", "B")}),
        ],
        ids=["chain", "empty", "diamond", "self_loop", "duplicate_edges"],
    )
    def test_edge_topologies(self, names, pairs, expected_edges):
        _, edges = make_graph(
            [{"name": name} for name in names],
            [{"cause": cause, "effect": effect} for cause, effect in pairs],
        )
        graph = build_digraph((), edges)
        assert set(graph.nodes) == set(names)
        assert set(graph.edges) == expected_edges


class TestGetOutcomeName:
    def test_finds_outcome(self):
        constructs, _ = make_graph([{"name": "X"}, {"name": "Y"}], [])
        assert get_outcome_name(constructs, fixture_entity_id("construct", "Y")) == "Y"

    def test_no_outcome(self):
        constructs, _ = make_graph([{"name": "X"}, {"name": "Z"}], [])
        assert get_outcome_name(constructs, None) is None

    def test_empty_constructs(self):
        assert get_outcome_name((), None) is None


class TestGetAllTreatments:
    @pytest.mark.parametrize(
        ("names", "pairs", "outcome", "expected"),
        [
            (["A", "B", "Y"], [("A", "B"), ("B", "Y")], "Y", ["A", "B"]),
            (["X", "Y", "Z"], [("X", "Y")], "Y", ["X"]),
            (["A", "B"], [("A", "B")], None, []),
            (
                ["Zebra", "Apple", "Outcome"],
                [("Zebra", "Outcome"), ("Apple", "Outcome")],
                "Outcome",
                ["Apple", "Zebra"],
            ),
            (["X", "Y", "Z"], [("X", "Y"), ("X", "Z")], "Y", ["X"]),
            (
                ["A", "B", "C", "D"],
                [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
                "D",
                ["A", "B", "C"],
            ),
            ([], [], None, []),
            (["Y"], [], "Y", []),
        ],
        ids=[
            "chain",
            "disconnected",
            "no_outcome",
            "sorted",
            "fork",
            "diamond",
            "empty",
            "outcome_only",
        ],
    )
    def test_treatments(self, names, pairs, outcome, expected):
        constructs, edges = make_graph(
            [{"name": name} for name in names],
            [{"cause": cause, "effect": effect} for cause, effect in pairs],
        )
        default_outcome = fixture_entity_id("construct", outcome) if outcome is not None else None
        assert get_all_treatments(constructs, edges, default_outcome) == expected


class TestModelSpecAccessors:
    def test_get_state_names_preserves_compiled_order(self):
        plan = make_model(["stress", "mood"], [("stress", "mood")])
        assert get_state_names(ModelSpec.model_validate(plan)) == ["stress", "mood"]
