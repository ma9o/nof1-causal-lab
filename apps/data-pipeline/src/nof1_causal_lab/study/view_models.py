"""Server-composed artifact views at a selected model revision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, FiniteFloat

from nof1_causal_lab.artifacts.availability import Evaluation, NotApplicable, Unavailable
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.identity import (
    GitOid,
    IndicatorId,
)
from nof1_causal_lab.artifacts.measurements import ObservationRecord
from nof1_causal_lab.artifacts.observations import ResolvedObservationSpec
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks


class RawDataDateRange(Value):
    """Observed date bounds of the uploaded table, when it contains a date column."""

    start: str
    end: str


class RawDataColumnDescription(Value):
    """A stored column's physical type and authored interpretation."""

    name: str
    dtype: str
    description: str | None


class RawDataData(Value):
    """Profile and representative rows from one uploaded table revision."""

    n_records: int
    n_columns: int
    date_range: RawDataDateRange | None
    sample: tuple[Mapping[str, str | None], ...]
    column_descriptions: tuple[RawDataColumnDescription, ...]


class MeasurementsData(Value):
    """Counts and representative observations read directly from one panel revision."""

    n_observations: int
    per_indicator_counts: Mapping[IndicatorId, int]
    combined_extractions_sample: tuple[ObservationRecord, ...]


class Added[PayloadT](Value):
    """The after payload exists; a null payload is still a present value."""

    kind: Literal["added"] = "added"
    after: PayloadT


class Removed[PayloadT](Value):
    """The before payload exists and has no successor in this comparison."""

    kind: Literal["removed"] = "removed"
    before: PayloadT


class Revised[PayloadT](Value):
    """Both payloads exist and differ by the comparison's stated criterion."""

    kind: Literal["revised"] = "revised"
    before: PayloadT
    after: PayloadT


class Unchanged[PayloadT](Value):
    """Both payloads satisfy the criterion; other attributes may still differ."""

    kind: Literal["unchanged"] = "unchanged"
    before: PayloadT
    after: PayloadT


type Change[PayloadT] = Annotated[
    Added[PayloadT] | Removed[PayloadT] | Revised[PayloadT], Field(discriminator="kind")
]


class DataPoint(Value):
    """An observed anchor and support; dates are synthetic for a calendar-free series."""

    anchor_time: AwareDatetime
    support_start: AwareDatetime | None
    support_end: AwareDatetime | None
    value: FiniteFloat | None


class DataSeries(Value):
    """One variable's recorded measurements in one history; no pooling across replicas."""

    variable: ResolvedObservationSpec | None
    time_origin: AwareDatetime | None = Field(
        description="Recorded calendar binding; null means the point dates are serialization coordinates, not real dates."
    )
    points: tuple[DataPoint, ...]


type DataStatistic = Literal[
    "observed_count", "missing_count", "mean", "sd", "min", "max", "proportion"
]


class DataStatisticComparison(Value):
    """The same descriptive statistic measured independently in every selected history."""

    statistic: DataStatistic
    level: str | None = None
    left: tuple[FiniteFloat | None, ...]
    right: tuple[FiniteFloat | None, ...]
    left_histogram: tuple[HistogramBin, ...]
    right_histogram: tuple[HistogramBin, ...]


class PredictiveComparison(Value):
    """A selected reference history retains its role even when checks are unavailable."""

    kind: Literal["comparison"] = "comparison"
    reference_side: Literal["left", "right"]
    evaluation: Evaluation[PosteriorPredictiveChecks]


type PredictiveComparisonResult = Annotated[
    PredictiveComparison | Unavailable | NotApplicable, Field(discriminator="kind")
]


class DataVariableDiff(Value):
    """Definitions, histories and comparisons for one persistent observation identity."""

    indicator_id: IndicatorId
    left: tuple[DataSeries, ...]
    right: tuple[DataSeries, ...]
    changes: tuple[Change[DataPoint], ...]
    statistics: tuple[DataStatisticComparison, ...]
    comparison_issues: tuple[str, ...]
    predictive: PredictiveComparisonResult


class Dataset(Value):
    """One parsed observation history, identified by its immutable source."""

    source: DataRef[GitOid, int]
    series: Mapping[IndicatorId, DataSeries]
