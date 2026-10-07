"""Presentation and compiler input rows derived from canonical model entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.model_structure import reference_indicators

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.duration import Duration
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


def get_reference_indicator_lookup(selection: StructuralSelection) -> dict[str, str]:
    """Return retained construct name to its planned reference indicator name."""
    dynamical_model_spec = selection.dynamical_model_spec
    return {
        dynamical_model_spec._constructs[construct_id].name: (
            dynamical_model_spec.indicator(indicator_id).observation.name
        )
        for construct_id, indicator_id in reference_indicators(selection).items()
    }


def get_model_clock(dynamical_model_spec: DynamicalModelSpec) -> Duration:
    """Require a measurement-ready model and return its authored clock duration."""
    dynamical_model_spec.require_measurements()
    assert dynamical_model_spec.measurement_clock is not None
    return dynamical_model_spec.measurement_clock
