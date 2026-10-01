"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.workers.context import MeasurementIndicator


def get_state_names(model: ModelSpec) -> list[str]:
    return [model._constructs[source_id].name for source_id in model.state_order]


def get_constructs(model: ModelSpec) -> list[ConstructSpec]:
    return list(model.constructs)


def get_manifest_indicators(model: ModelSpec) -> list[MeasurementIndicator]:
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


def get_model_clock(model: ModelSpec) -> str:
    model.require_measurements()
    assert model.measurement_clock is not None
    return model.measurement_clock
