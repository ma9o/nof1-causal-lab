"""Factory fixtures for ConstructSpec/IndicatorSpec schema tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from nof1_causal_lab.artifacts.construct import ConstructSpec, Role, TemporalStatus
from nof1_causal_lab.artifacts.indicator import IndicatorPolarity, IndicatorSpec
from tests.helpers import fixture_entity_id

if TYPE_CHECKING:
    from nof1_causal_lab.measurement_types import AggregationFunction, MeasurementDtype


@pytest.fixture
def construct_factory():
    """Factory for creating ConstructSpec objects.

    Usage:
        def test_something(construct_factory):
            stress = construct_factory("stress", Role.EXOGENOUS)
            mood = construct_factory("mood", Role.ENDOGENOUS)
    """

    def _make(
        name: str,
        role: Role = Role.ENDOGENOUS,
        temporal_status: TemporalStatus = TemporalStatus.TIME_VARYING,
    ) -> ConstructSpec:
        return ConstructSpec(
            id=fixture_entity_id("construct", name),
            name=name,
            description=f"{name} description",
            role=role,
            temporal_status=temporal_status,
        )

    return _make


@pytest.fixture
def indicator_factory():
    """Factory for creating IndicatorSpec objects.

    Usage:
        def test_something(indicator_factory):
            ind = indicator_factory("mood_rating")
    """

    def _make(
        name: str,
        dtype: MeasurementDtype = "continuous",
        aggregation: AggregationFunction = "mean",
        construct_polarity: IndicatorPolarity = IndicatorPolarity.POSITIVE,
        ordinal_levels: list[str] | None = None,
        source_columns: list[str] | None = None,
        extraction_mode: Literal["computed", "semantic"] = "semantic",
    ) -> IndicatorSpec:
        # Auto-provide ordinal_levels for ordinal dtype if not specified
        if dtype == "ordinal" and ordinal_levels is None:
            ordinal_levels = ["low", "medium", "high"]
        return IndicatorSpec(
            id=fixture_entity_id("indicator", name),
            name=name,
            construct_polarity=construct_polarity,
            how_to_measure=f"Extract {name}",
            measurement_dtype=dtype,
            aggregation=aggregation,
            ordinal_levels=ordinal_levels,
            source_columns=source_columns or [name],
            extraction_mode=extraction_mode,
        )

    return _make
