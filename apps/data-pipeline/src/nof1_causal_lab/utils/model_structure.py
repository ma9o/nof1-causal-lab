"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.model_structure import (
    reference_indicators,
    selected_state_ids,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.duration import Duration
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def get_state_names(model: ModelSpec) -> list[str]:
    return [model._constructs[source_id].name for source_id in selected_state_ids(model)]


def get_constructs(model: ModelSpec) -> list[ConstructSpec]:
    return list(model.constructs)


def get_reference_indicator_lookup(model: ModelSpec) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    return {
        model._constructs[construct_id].name: (model._indicators[indicator_id].name)
        for construct_id, indicator_id in reference_indicators(model).items()
    }


def get_model_clock(model: ModelSpec) -> Duration:
    model.require_measurements()
    assert model.measurement_clock is not None
    return model.measurement_clock
