"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.io import DataDiffOutput
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.simulation import SimulationEvidence
from nof1_causal_lab.models.posterior_predictive import compare_data_variables

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts import DataDiffRequest
    from nof1_causal_lab.artifacts.data_ref import DataSelection
    from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId
    from nof1_causal_lab.study.view_models import Dataset


def _compute_data_diff(workspace_id: str, request: DataDiffRequest[GitOid]) -> DataDiffOutput:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from nof1_causal_lab.study.data import read_data_histories, read_data_source
    from nof1_causal_lab.study.store import ArtifactStore, read_dataset, read_model

    store = ArtifactStore(workspace_id)
    input_indicators: set[IndicatorId] = set()

    @cache
    def load(source: DataRef[GitOid, int | None]) -> tuple[Dataset, ...]:
        producer = read_data_source(store, source.revision)
        if isinstance(producer, SimulationEvidence):
            model = read_model(store, producer.model.revision)
            input_indicators.update(
                indicator.observation.id
                for construct in model.constructs
                if construct.role == "exogenous"
                for indicator in construct.indicators
            )
        return tuple(
            read_dataset(history.source, history.observations)
            for history in read_data_histories(store, source)
        )

    def selection(value: DataSelection[GitOid]) -> tuple[Dataset, ...]:
        refs = (value,) if isinstance(value, DataRef) else value
        return tuple(dataset for source in refs for dataset in load(source))  # pyright: ignore[reportArgumentType] -- Value freezes DataRef; Pydantic's stub declares __hash__=None before its frozen metaclass installs the hash.

    left, right = selection(request.input.left_ref), selection(request.input.right_ref)
    return DataDiffOutput(
        left=tuple(item.source for item in left),
        right=tuple(item.source for item in right),
        variables=compare_data_variables(left, right, input_indicators=input_indicators),
    )


def read_data_diff(workspace_id: str, request: DataDiffRequest[GitOid]) -> DataDiffOutput:
    """Compute the comparison once during its owning action."""
    return _compute_data_diff(workspace_id, request)
