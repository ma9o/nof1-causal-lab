"""Canonical model values projected into the inputs consumed by numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.json_types import JsonObject, JsonValue


def graph_input(model: ModelSpec) -> JsonObject:
    """Serialized directed assumptions used to fingerprint identification inputs."""
    result: dict[str, JsonValue] = {
        "constructs": [
            item.model_dump(
                mode="json", include={"id", "name", "description", "role", "temporal_status"}
            )
            for item in model.constructs
        ],
        "edges": [
            {
                **item.model_dump(mode="json", include={"id", "description", "sources"}),
                "cause_id": item.cause.id,
                "effect_id": item.effect.id,
            }
            for item in model.edges
        ],
        "default_outcome": model.default_outcome,
    }
    return result


def indicator_rows(model: ModelSpec) -> list[JsonObject]:
    """Measurement inputs with ownership derived from canonical containment."""
    return [
        {**indicator.model_dump(mode="json", exclude={"likelihood"}), "construct_id": construct.id}
        for construct, indicator in model.iter_indicators()
    ]


def observation_input(model: ModelSpec) -> JsonObject:
    return {
        "model_clock": model.measurement_clock.source
        if model.measurement_clock is not None
        else None,
        "indicators": indicator_rows(model),
    }


def identification_input(model: ModelSpec) -> JsonObject:
    return {"graph": graph_input(model), "observations": observation_input(model)}


def compilation_input(model: ModelSpec) -> JsonObject:
    """Structure and constants needed by execution, independent of the current law."""
    result: dict[str, JsonValue] = {
        **model.model_dump(
            mode="json",
            exclude={
                "question": True,
                "edges": True,
                "distributions": True,
                "time_points": True,
                "parameters": {"__all__": {"distribution", "transform"}},
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
    return result


def input_fingerprints(model: ModelSpec) -> dict[str, str]:
    """Content identities of the values supplied to each numerical boundary."""
    from nof1_causal_lab.artifacts.identity import scientific_id

    values = {
        "observations": observation_input(model),
        "identification": identification_input(model),
        "compilation": compilation_input(model),
        "belief": model.model_dump(mode="json", exclude={"question"}),
    }
    return {
        purpose: scientific_id("input", ["additive-model-v1", value])
        for purpose, value in values.items()
    }


def data_binding_issues(model: ModelSpec, metadata: PreparedDataMetadata) -> list[str]:
    """Compare scientific observation semantics, independently of generating-model ancestry."""
    variables = {item.id: item for item in metadata.variables}
    issues = []
    for indicator in model.indicators:
        variable = variables.get(indicator.observation.id)
        if variable is None:
            issues.append(f"No prepared variable for model indicator {indicator.observation.id}")
            continue
        for field, expected, actual in (
            (
                "measurement_dtype",
                indicator.observation.measurement_dtype,
                variable.measurement_dtype,
            ),
            ("aggregation", indicator.observation.aggregation, variable.aggregation),
            ("ordinal_levels", indicator.observation.ordinal_levels, variable.ordinal_levels),
            (
                "categorical_levels",
                indicator.observation.categorical_levels,
                variable.categorical_levels,
            ),
        ):
            if expected != actual:
                issues.append(f"Observation {indicator.observation.id} has incompatible {field}")
        window = indicator.observation.observation_window or model.measurement_clock
        assert (
            variable.observation_window is not None
        )  # PreparedDataMetadata resolves every window.
        if window is None or window.seconds != variable.observation_window.seconds:
            issues.append(
                f"Observation {indicator.observation.id} has an incompatible observation window"
            )
    return issues
