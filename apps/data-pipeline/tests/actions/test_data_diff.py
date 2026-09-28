"""Stored histories compare symmetrically without generation or scientific state changes."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nof1_causal_lab.actions.data_diff import (
    DataDiffRequest,
    DataRef,
    Dataset,
    data_diff,
    read_data_diff,
)
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.artifacts.simulation import (
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.read_facade import create_read_facade_app
from tests.git_fixtures import git_oid


def _dataset(values, *, number=1, times=None, variable=None):
    variable = variable or ObservationSpec(
        id="indicator:y",
        name="Y",
        measurement_dtype="continuous",
        aggregation="last",
        observation_window="1d",
    )
    times = list(range(len(values))) if times is None else times
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    anchors = [origin + timedelta(days=day) for day in times]
    width = timedelta(days=1) if variable.support_kind.value == "interval" else timedelta()
    frame = pl.DataFrame(
        {
            "indicator_id": [variable.id] * len(values),
            "value": values,
            "anchor_time": anchors,
            "support_start": [time - width for time in anchors],
            "support_end": anchors,
            "support_kind": [variable.support_kind.value] * len(values),
            "summary_operator": [variable.summary_operator.value] * len(values),
            "anchor_policy": [variable.anchor_policy.value] * len(values),
            "observation_window": [variable.observation_window] * len(values),
        },
        schema_overrides={"value": pl.Float64},
    )
    return Dataset(DataRef(kind="panel", revision=git_oid(number)), (variable,), frame)


@pytest.mark.contract
def test_selection_contracts_require_nonempty_explicit_sources():
    source = {"kind": "panel", "revision": git_oid(1)}
    request = DataDiffRequest(left=source, right=[source])
    assert DataDiffRequest.model_validate_json(request.model_dump_json()) == request
    for invalid in (
        [],
        {**source, "replicate": 0},
        {**source, "time_origin": "2026-01-01T00:00:00Z"},
    ):
        with pytest.raises(ValidationError):
            DataDiffRequest(left=invalid, right=source)
    for invalid in ({"replicate": -1}, {"time_origin": "2026-01-01"}):
        with pytest.raises(ValidationError):
            DataRef(kind="simulation", revision=git_oid(1), **invalid)


@pytest.mark.inference(concern="predictive")
def test_single_histories_report_values_missingness_and_schedule_changes():
    left = _dataset([1, None, 3], times=[0, 1, 2])
    right = _dataset([1, 4, 8], times=[0, 1, 3], number=2)
    result = data_diff(left, right)
    variable = result.variables[0]
    assert [item.change for item in variable.changes] == ["revised", "removed", "added"]
    assert variable.changes[0].left is not None
    assert variable.changes[0].right is not None
    assert variable.changes[0].left.value is None
    assert variable.changes[0].right.value == 4
    assert variable.predictive_checks is None
    assert variable.reference_side is None
    assert variable.comparison_issues == ("Observation schedules or measurement windows differ",)
    missing = next(item for item in variable.statistics if item.statistic == "missing_count")
    assert (missing.left, missing.right) == ((1,), (0,))
    reverse = data_diff(right, left).variables[0]
    assert [item.change for item in reverse.changes] == ["revised", "added", "removed"]
    assert data_diff(left, left).variables[0].changes == ()


@pytest.mark.inference(concern="predictive")
def test_replica_checks_are_symmetric_and_preserve_whole_history_statistics():
    observed = _dataset([0, 1, None, 3])
    replicas = [
        _dataset([offset, 1 + offset, 1000, 3 + offset], number=offset + 2) for offset in range(3)
    ]
    forward = data_diff(replicas, observed).variables[0]
    backward = data_diff(observed, replicas).variables[0]
    assert forward.reference_side == "right"
    assert backward.reference_side == "left"
    assert forward.predictive_checks == backward.predictive_checks
    assert forward.predictive_checks is not None
    stats = {item.stat_name: item for item in forward.predictive_checks.test_stats}
    assert stats["mean"].observed_value == pytest.approx(4 / 3)
    assert stats["mean"].rep_values == pytest.approx([4 / 3, 7 / 3, 10 / 3])
    assert forward.predictive_checks.n_subsample == 3
    assert len(forward.left) == 3
    # Many-to-many retains one summary per history, without implying paired draws.
    many = data_diff(replicas, [observed, _dataset([0, 1, 2, 3], number=9)]).variables[0]
    assert many.predictive_checks is None
    means = next(item for item in many.statistics if item.statistic == "mean")
    assert len(means.left) == 3
    assert len(means.right) == 2


@pytest.mark.inference(concern="predictive")
def test_measurement_mismatches_and_uncovered_times_do_not_produce_predictive_checks():
    observed = _dataset([1, 2, 3])
    replicas = [_dataset([1, 2, 3], number=number, times=[0, 1, 2.5]) for number in (2, 3)]
    mismatch = data_diff(observed, replicas).variables[0]
    assert mismatch.predictive_checks is None
    assert mismatch.predictive_unavailable_reason is not None
    assert "anchors" in mismatch.predictive_unavailable_reason
    interval = observed.variables[0].model_copy(update={"aggregation": "mean"})
    replicas = [_dataset([1, 2, 3], number=number, variable=interval) for number in (2, 3)]
    mismatch = data_diff(observed, replicas).variables[0]
    assert mismatch.predictive_unavailable_reason == "Measurement definitions differ"
    assert mismatch.left[0].variable is not None
    assert mismatch.right[0].variable is not None
    assert mismatch.left[0].variable.aggregation == "last"
    assert mismatch.right[0].variable.aggregation == "mean"
    renamed = replace(
        observed, variables=(observed.variables[0].model_copy(update={"name": "Renamed"}),)
    )
    assert not data_diff(observed, renamed).variables[0].comparison_issues
    # A first-in-window observation is anchored at support_start, not support_end.
    first = _dataset(
        [1, 2, 3], variable=observed.variables[0].model_copy(update={"aggregation": "first"})
    )
    first = replace(
        first,
        observations=first.observations.with_columns(
            pl.col("support_end") + timedelta(days=1),
        ),
    )
    assert data_diff(first, first).variables[0].changes == ()


@pytest.mark.inference(concern="predictive")
def test_discrete_codebooks_compare_frequencies_and_keep_absent_variables_explicit():
    variable = ObservationSpec(
        id="indicator:category",
        name="Category",
        measurement_dtype="categorical",
        aggregation="last",
        observation_window="1d",
        categorical_levels=("a", "b", "c"),
    )
    a = _dataset([0, 1, 1], variable=variable)
    b = [_dataset([2, 2, 1], number=n, variable=variable) for n in (2, 3)]
    result = data_diff(a, b).variables[0]
    proportions = {item.level: item for item in result.statistics if item.statistic == "proportion"}
    assert proportions["c"].left == (0,)
    assert proportions["c"].right == pytest.approx((2 / 3, 2 / 3))
    assert not any(item.statistic == "mean" for item in result.statistics)
    assert result.predictive_checks is None
    result = data_diff(a, _dataset([1, 2, 3], number=4))
    assert len(result.variables) == 2
    assert all("absent" in item.comparison_issues[0] for item in result.variables)


@pytest.mark.contract
def test_invalid_histories_and_duplicate_sources_are_rejected_before_comparison():
    data = _dataset([1, 2, 3])
    with pytest.raises(ValueError, match="at least one"):
        data_diff([], data)
    with pytest.raises(ValueError, match="counted twice"):
        data_diff([data, data], data)
    with pytest.raises(ValueError, match="duplicate anchors"):
        data_diff(_dataset([1, 2], times=[0, 0]), data)
    with pytest.raises(ValueError, match="finite"):
        data_diff(_dataset([1, np.inf]), data)


@pytest.mark.inference(concern="predictive")
def test_read_only_api_compares_saved_draws_without_generation_or_model_access(
    tmp_path, monkeypatch
):
    from nof1_causal_lab.artifacts.data_preparation import (
        PreparedDataMetadata,
        SimulationReplicateRef,
    )
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    store, history = ArtifactStore("DIFF"), StudyRepository("DIFF")
    observed = _dataset([1, 2, 3])
    values = np.asarray([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]], dtype=float)
    report = SimulationReport(
        model=GitRef(workspace_id="DIFF", revision=git_oid(99), path="model.json"),
        design=SimulationSpec(end=2),
        times=(0, 1, 2),
        draws=3,
        seed=0,
        state_ids=(),
        parameter_draws={},
        latent_paths="unreadable-latent-array",
        observations=store.write_array(values),
        observation_layout=SimulationObservationLayout(
            variables=observed.variables,
            support_start_times=store.write_array(np.arange(3.0)[:, None]),
            support_end_times=store.write_array(np.arange(3.0)[:, None]),
            mask=store.write_array(np.ones_like(values, dtype=bool)),
        ),
    )
    commit = history.append(
        TransitionRecord(
            seq=1,
            ts="2026-09-28T00:00:00Z",
            action="simulate",
            operation_id="simulate",
            inputs={},
            status="applied",
            trace_ids=[],
            resume=None,
            diagnostics={"report": report.model_dump(mode="json")},
        )
    )
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    shifted = observed.observations.with_columns(
        pl.col(name) + (origin - datetime(1970, 1, 1, tzinfo=UTC))
        for name in ("anchor_time", "support_start", "support_end")
    )
    panel = store.write_artifact(
        "panel",
        produced_by="run:simulated_measurements",
        derived_from={},
        json_files={
            "metadata.json": PreparedDataMetadata(
                source=SimulationReplicateRef(revision=commit, replicate=1),
                variables=observed.variables,
            ).model_dump(mode="json")
        },
        parquet_files={"panel.parquet": shifted},
    )
    request = DataDiffRequest(
        left=DataRef(kind="simulation", revision=commit, time_origin=origin),
        right=DataRef(kind="panel", revision=panel.revision),
    )
    refs_before = sorted(store.repo.references)
    client = TestClient(create_read_facade_app())
    response = client.post("/api/episodes/DIFF/data-diff", json=request.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["left"]) == 3
    assert result["variables"][0]["predictive_checks"]["n_subsample"] == 3
    assert history.head() == commit
    assert sorted(store.repo.references) == refs_before
    assert store.list_revisions("model") == []
    assert isinstance(request.left, DataRef)
    one = request.model_copy(update={"left": request.left.model_copy(update={"replicate": 1})})
    assert read_data_diff("DIFF", one).variables[0].changes == ()
    bad = request.model_copy(
        update={"left": DataRef(kind="simulation", revision=commit, replicate=3)}
    )
    assert (
        client.post("/api/episodes/DIFF/data-diff", json=bad.model_dump(mode="json")).status_code
        == 422
    )
    bad = request.model_copy(update={"right": DataRef(kind="panel", revision=git_oid(98))})
    assert (
        client.post("/api/episodes/DIFF/data-diff", json=bad.model_dump(mode="json")).status_code
        == 404
    )
