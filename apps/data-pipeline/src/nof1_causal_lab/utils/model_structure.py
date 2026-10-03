"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.model_structure import reference_indicators

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.duration import Duration
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


def get_constructs(model: ModelSpec) -> list[ConstructSpec]:
    return list(model.constructs)


def get_reference_indicator_lookup(selection: StructuralSelection) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    model = selection.model
    return {
        model._constructs[construct_id].name: (model._indicators[indicator_id].observation.name)
        for construct_id, indicator_id in reference_indicators(selection).items()
    }


def get_model_clock(model: ModelSpec) -> Duration:
    model.require_measurements()
    assert model.measurement_clock is not None
    return model.measurement_clock
