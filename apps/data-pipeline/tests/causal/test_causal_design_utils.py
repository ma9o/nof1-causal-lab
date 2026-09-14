"""Tests for ``utils.causal_design`` helpers.

Trivial accessors are exercised through higher-level tests. This file covers
the helpers with real transformation or graph logic:
- ``make_measurement_extraction_context``
- ``build_digraph``
- ``get_outcome_name``
- ``get_all_treatments``
- ModelSpec state and marginalized-scale accessors
"""

from typing import Any

import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.utils.causal_design import (
    build_digraph_from_edges,
    get_all_treatments,
    get_outcome_name,
    make_measurement_extraction_context,
)
from nof1_causal_lab.utils.model_structure import (
    get_marginalized_scales,
    get_state_names,
)
from tests.helpers import make_model


def _full_spec():
    """Minimal valid CausalDesign dict."""
    return {
        "latent": {
            "default_outcome": "construct:bbc87212909e45b9e6c3",
            "constructs": [
                {"id": "construct:6b04dc42c531e7091eb8", "name": "stress", "role": "exogenous"},
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
    def test_simple_chain(self):
        model = {
            "edges": [
                {"cause": "A", "effect": "B"},
                {"cause": "B", "effect": "C"},
            ]
        }
        graph = build_digraph_from_edges((model)["edges"])
        assert set(graph.nodes()) == {"A", "B", "C"}
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert not graph.has_edge("A", "C")

    def test_empty_edges(self):
        assert len(build_digraph_from_edges(({"edges": []})["edges"]).nodes()) == 0

    def test_diamond_topology(self):
        graph = build_digraph_from_edges(
            (
                {
                    "edges": [
                        {"cause": "A", "effect": "B"},
                        {"cause": "A", "effect": "C"},
                        {"cause": "B", "effect": "D"},
                        {"cause": "C", "effect": "D"},
                    ]
                }
            )["edges"]
        )
        assert set(graph.nodes()) == {"A", "B", "C", "D"}
        assert len(graph.edges()) == 4

    def test_self_loop(self):
        graph = build_digraph_from_edges(({"edges": [{"cause": "A", "effect": "A"}]})["edges"])
        assert set(graph.nodes()) == {"A"}
        assert graph.has_edge("A", "A")

    def test_duplicate_edges(self):
        graph = build_digraph_from_edges(
            (
                {
                    "edges": [
                        {"cause": "A", "effect": "B"},
                        {"cause": "A", "effect": "B"},
                    ]
                }
            )["edges"]
        )
        assert len(graph.edges()) == 1


class TestGetOutcomeName:
    def test_finds_outcome(self):
        assert (
            get_outcome_name(
                {
                    "default_outcome": "construct:Y",
                    "constructs": [
                        {"id": "construct:X", "name": "X"},
                        {"id": "construct:Y", "name": "Y"},
                    ],
                }
            )
            == "Y"
        )

    def test_no_outcome(self):
        assert (
            get_outcome_name(
                {
                    "constructs": [
                        {"name": "X"},
                        {"name": "Z"},
                    ]
                }
            )
            is None
        )

    def test_empty_constructs(self):
        assert get_outcome_name({"constructs": []}) is None

    def test_missing_constructs_key(self):
        assert get_outcome_name({}) is None


class TestGetAllTreatments:
    def test_chain_treatments(self):
        treatments = get_all_treatments(
            {
                "default_outcome": "construct:Y",
                "constructs": [
                    {"id": "construct:A", "name": "A"},
                    {"id": "construct:B", "name": "B"},
                    {"id": "construct:Y", "name": "Y"},
                ],
                "edges": [
                    {"cause_id": "construct:A", "effect_id": "construct:B"},
                    {"cause_id": "construct:B", "effect_id": "construct:Y"},
                ],
            }
        )
        assert treatments == ["A", "B"]

    def test_disconnected_not_treatment(self):
        treatments = get_all_treatments(
            {
                "default_outcome": "construct:Y",
                "constructs": [
                    {"id": "construct:X", "name": "X"},
                    {"id": "construct:Y", "name": "Y"},
                    {"id": "construct:Z", "name": "Z"},
                ],
                "edges": [{"cause_id": "construct:X", "effect_id": "construct:Y"}],
            }
        )
        assert treatments == ["X"]

    def test_no_outcome_returns_empty(self):
        assert (
            get_all_treatments(
                {
                    "constructs": [
                        {"id": "construct:A", "name": "A"},
                        {"id": "construct:B", "name": "B"},
                    ],
                    "edges": [{"cause_id": "construct:A", "effect_id": "construct:B"}],
                }
            )
            == []
        )

    def test_sorted_output(self):
        treatments = get_all_treatments(
            {
                "default_outcome": "construct:Outcome",
                "constructs": [
                    {"id": "construct:Zebra", "name": "Zebra"},
                    {"id": "construct:Apple", "name": "Apple"},
                    {"id": "construct:Outcome", "name": "Outcome"},
                ],
                "edges": [
                    {"cause_id": "construct:Zebra", "effect_id": "construct:Outcome"},
                    {"cause_id": "construct:Apple", "effect_id": "construct:Outcome"},
                ],
            }
        )
        assert treatments == ["Apple", "Zebra"]

    def test_fork_topology(self):
        treatments = get_all_treatments(
            {
                "default_outcome": "construct:Y",
                "constructs": [
                    {"id": "construct:X", "name": "X"},
                    {"id": "construct:Y", "name": "Y"},
                    {"id": "construct:Z", "name": "Z"},
                ],
                "edges": [
                    {"cause_id": "construct:X", "effect_id": "construct:Y"},
                    {"cause_id": "construct:X", "effect_id": "construct:Z"},
                ],
            }
        )
        assert treatments == ["X"]

    def test_diamond_all_treatments(self):
        treatments = get_all_treatments(
            {
                "default_outcome": "construct:D",
                "constructs": [
                    {"id": "construct:A", "name": "A"},
                    {"id": "construct:B", "name": "B"},
                    {"id": "construct:C", "name": "C"},
                    {"id": "construct:D", "name": "D"},
                ],
                "edges": [
                    {"cause_id": "construct:A", "effect_id": "construct:B"},
                    {"cause_id": "construct:A", "effect_id": "construct:C"},
                    {"cause_id": "construct:B", "effect_id": "construct:D"},
                    {"cause_id": "construct:C", "effect_id": "construct:D"},
                ],
            }
        )
        assert treatments == ["A", "B", "C"]

    def test_empty_model(self):
        assert get_all_treatments({"constructs": [], "edges": []}) == []

    def test_outcome_only(self):
        assert (
            get_all_treatments(
                {
                    "default_outcome": "construct:Y",
                    "constructs": [{"id": "construct:Y", "name": "Y"}],
                    "edges": [],
                }
            )
            == []
        )


class TestModelSpecAccessors:
    def test_get_state_names_preserves_compiled_order(self):
        plan = make_model(["stress", "mood"], [("stress", "mood")])
        assert get_state_names(ModelSpec.model_validate(plan)) == ["stress", "mood"]


class TestGetMarginalizedScales:
    @staticmethod
    def _spec(induced_dependencies: list[dict[str, Any]]) -> ModelSpec:
        state_names = sorted(
            {str(state) for dependency in induced_dependencies for state in dependency["between"]}
        )
        source_names = sorted(
            {
                str(source)
                for dependency in induced_dependencies
                for source in dependency["source_confounders"]
            }
        )
        from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
        from tests.helpers import fixture_entity_id, make_model

        model = make_model(state_names or ["observed"])
        confounders = tuple(
            ConstructSpec(
                id=fixture_entity_id("construct", name),
                name=name,
                description="Latent root",
                role="exogenous",
                temporal_status=(
                    "time_invariant"
                    if next(
                        dep["kind"]
                        for dep in induced_dependencies
                        if name in dep["source_confounders"]
                    )
                    == "initial_state_correlation"
                    else "time_varying"
                ),
            )
            for name in source_names
        )
        pairs = {
            (source, child)
            for dep in induced_dependencies
            for source in dep["source_confounders"]
            for child in dep["between"]
        }
        model = model.revised(
            edges=replace_constructs(
                model.edges
                + tuple(
                    CausalEdgeSpec(
                        id=fixture_entity_id("edge", source + "->" + child),
                        cause=next(item for item in confounders if item.name == source),
                        effect=model.get_construct(fixture_entity_id("construct", child)),
                        description="Explicit confounding",
                        lagged=False,
                    )
                    for source, child in sorted(pairs)
                ),
                tuple(c for c in model.constructs if c.indicators) + confounders,
            )
        )
        # Seed the derived dependency cache to exercise grouping independently,
        # including inconsistent kinds that a valid model would never derive.
        return model.model_copy(
            update={
                "induced_dependencies": {
                    (
                        fixture_entity_id("construct", dep["between"][0]),
                        fixture_entity_id("construct", dep["between"][1]),
                        dep["kind"],
                    ): tuple(
                        fixture_entity_id("construct", name) for name in dep["source_confounders"]
                    )
                    for dep in induced_dependencies
                }
            }
        )

    def test_golden_like_three_plus_one_confounders_yield_two_scales(self):
        spec = self._spec(
            [
                {
                    "between": ["screen_time", "sleep_quality"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["age", "living_situation", "personality_traits"],
                },
                {
                    "between": ["screen_time", "stress"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["occupation_demands"],
                },
            ]
        )
        scales = get_marginalized_scales(spec)

        assert [scale["parameter"] for scale in scales] == [
            "tau_age__living_situation__personality_traits",
            "tau_occupation_demands",
        ]
        merged, solo = scales
        assert merged["sources"] == ["age", "living_situation", "personality_traits"]
        assert merged["affected_states"] == ["screen_time", "sleep_quality"]
        assert merged["directions"] == [("screen_time", "sleep_quality")]
        assert merged["kind"] == "initial_state_correlation"
        assert solo["sources"] == ["occupation_demands"]
        assert solo["affected_states"] == ["screen_time", "stress"]

    def test_multi_scale_per_dep_when_footprints_differ(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c1", "c2"],
                },
                {
                    "between": ["x", "z"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c2"],
                },
                {
                    "between": ["y", "z"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c2"],
                },
            ]
        )
        scales = get_marginalized_scales(spec)

        assert len(scales) == 2
        by_name = {scale["parameter"]: scale for scale in scales}
        assert by_name["tau_c1"]["affected_states"] == ["x", "y"]
        assert by_name["tau_c1"]["directions"] == [("x", "y")]
        assert by_name["tau_c2"]["affected_states"] == ["x", "y", "z"]
        assert by_name["tau_c2"]["directions"] == [
            ("x", "y"),
            ("x", "z"),
            ("y", "z"),
        ]

    def test_confounder_with_inconsistent_kind_raises(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["c"],
                },
                {
                    "between": ["y", "z"],
                    "kind": "innovation_correlation",
                    "source_confounders": ["c"],
                },
            ]
        )
        with pytest.raises(ValueError, match="inconsistent dependency kinds"):
            get_marginalized_scales(spec)

    def test_empty_dependencies_yield_empty_scales(self):
        assert get_marginalized_scales(self._spec([])) == []

    def test_canonical_name_is_sorted(self):
        spec = self._spec(
            [
                {
                    "between": ["x", "y"],
                    "kind": "initial_state_correlation",
                    "source_confounders": ["zebra", "apple", "mango"],
                }
            ]
        )
        (scale,) = get_marginalized_scales(spec)
        assert scale["parameter"] == "tau_apple__mango__zebra"
