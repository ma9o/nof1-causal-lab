"""Canonical model values projected into the inputs consumed by numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


# Serialized scientific inputs at the boundary to graph and observation operations.
type ScientificInput = dict[str, Any]


def graph_input(model: ModelSpec) -> ScientificInput:
    """Directed assumptions consumed by temporal unrolling and identification."""
    return {
        "constructs": [
            item.model_dump(
                mode="json", include={"id", "name", "description", "role", "temporal_status"}
            )
            for item in model.constructs
        ],
        "edges": [
            {
                **item.model_dump(mode="json", include={"id", "description", "lagged", "sources"}),
                "cause_id": item.cause.id,
                "effect_id": item.effect.id,
            }
            for item in model.edges
        ],
        "default_outcome": model.default_outcome,
    }


def indicator_rows(model: ModelSpec) -> list[ScientificInput]:
    """Measurement inputs with ownership derived from canonical containment."""
    return [
        {**indicator.model_dump(mode="json", exclude={"likelihood"}), "construct_id": construct.id}
        for construct, indicator in model.iter_indicators()
    ]


def observation_input(model: ModelSpec) -> ScientificInput:
    return {"model_clock": model.measurement_clock, "indicators": indicator_rows(model)}


def identification_input(model: ModelSpec) -> ScientificInput:
    return {"graph": graph_input(model), "observations": observation_input(model)}


def compilation_input(model: ModelSpec) -> ScientificInput:
    """Structure and constants needed by execution, independent of the current law."""
    return {
        **model.model_dump(
            mode="json",
            exclude={
                "question": True,
                "edges": True,
                "distributions": True,
                "time_points": True,
                "parameters": {
                    "__all__": {"distribution", "distribution_transform", "reference_interval_days"}
                },
            },
        ),
        "constructs": [
            construct.model_dump(mode="json", exclude={"distribution"})
            for construct in model.constructs
        ],
        "edges": [
            {
                **edge.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause_id": edge.cause.id,
                "effect_id": edge.effect.id,
            }
            for edge in model.edges
        ],
    }


def input_fingerprints(model: ModelSpec) -> dict[str, str]:
    """Content identities of the values supplied to each numerical boundary."""
    from nof1_causal_lab.artifacts.identity import scientific_id

    values = {
        "extraction": {"question": model.question, **observation_input(model)},
        "identification": identification_input(model),
        "compilation": compilation_input(model),
        "belief": model.model_dump(mode="json", exclude={"question"}),
    }
    return {
        purpose: scientific_id("input", ["additive-model-v1", value])
        for purpose, value in values.items()
    }
