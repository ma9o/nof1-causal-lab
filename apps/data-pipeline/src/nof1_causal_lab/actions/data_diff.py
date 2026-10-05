"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts import call_identity
from nof1_causal_lab.models.posterior_predictive import data_diff
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.view_models import (
    DataDiffReport,
    DataDiffRequest,
    DataRef,
    DataSelection,
    Dataset,
    PanelRef,
    SimulationRef,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import IndicatorId


def _compute_data_diff(workspace_id: str, request: DataDiffRequest) -> DataDiffReport:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from nof1_causal_lab.actions.prepare_data import read_simulation_observations
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.lineage import read_data_metadata
    from nof1_causal_lab.study.store import ArtifactStore, read_dataset, read_model

    store = ArtifactStore(workspace_id)
    read_array = cache(store.read_array)
    input_indicators: set[IndicatorId] = set()

    @cache
    def load(source: DataRef) -> tuple[Dataset, ...]:
        if source.kind == "panel":
            metadata = read_data_metadata(store, source.revision)
            return (
                read_dataset(
                    source,
                    metadata.variables,
                    store.read_parquet_file(
                        "panel",
                        source.revision,
                        "panel.parquet",
                    ),
                    metadata.time_origin,
                ),
            )
        record = StudyRepository(workspace_id).record(source.revision)
        if record.record.attempt.action != "simulate" or not isinstance(
            record.record.attempt.outcome, Applied
        ):
            raise StudyLookupError("Simulation data must select an applied simulation commit")
        report = record.record.attempt.outcome.result.evidence
        if report.model.workspace_id != workspace_id:
            raise StudyLookupError("The simulation must belong to the selected study")
        model = read_model(store, report.model.revision)
        input_indicators.update(
            indicator.observation.id
            for construct in model.constructs
            if construct.role == "exogenous"
            for indicator in construct.indicators
        )
        indices = range(report.draws) if source.replicate is None else (source.replicate,)
        result = []
        for replicate in indices:
            panel = read_simulation_observations(report, replicate, read_array=read_array)
            result.append(
                read_dataset(
                    SimulationRef(revision=source.revision, replicate=replicate),
                    report.observation_layout.variables,
                    panel,
                    report.time_origin,
                )
            )
        return tuple(result)

    def selection(value: DataSelection) -> tuple[Dataset, ...]:
        refs = (value,) if isinstance(value, (PanelRef, SimulationRef)) else value
        return tuple(dataset for source in refs for dataset in load(source))  # pyright: ignore[reportArgumentType] -- Value freezes DataRef; Pydantic's stub declares __hash__=None before its frozen metaclass installs the hash.

    left, right = selection(request.left), selection(request.right)
    return data_diff(left, right, input_indicators=input_indicators)


def read_data_diff(workspace_id: str, request: DataDiffRequest) -> DataDiffReport:
    """Current-code comparison of the exact selections retained by its action leaf."""
    from nof1_causal_lab.study.store import cached_value

    value, _ = cached_value(
        workspace_id,
        ("data-diff", call_identity(request)),
        TypeAdapter(DataDiffReport),
        lambda: _compute_data_diff(workspace_id, request),
    )
    return value
