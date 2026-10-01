"""Stored histories compare symmetrically without generation or scientific state changes."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
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
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.store import ArtifactStore
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
    return Dataset(DataRef(kind="panel", revision=git_oid(number)), (variable,), frame, origin)


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
    assert variable.predictive_unavailable_reason is None
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
    assert mismatch.predictive_unavailable_reason is None
    assert mismatch.comparison_issues == ("Observation schedules or measurement windows differ",)
    missing = data_diff(observed, [_dataset([1, None, 3], number=n) for n in (2, 3)]).variables[0]
    assert missing.predictive_checks is None
    assert (
        missing.predictive_unavailable_reason
        == "Replicas contain missing values at observed anchors"
    )
    assert missing.comparison_issues == ()
    interval = type(observed.variables[0]).model_validate(
        {**observed.variables[0].model_dump(), "aggregation": "mean"}
    )
    replicas = [_dataset([1, 2, 3], number=number, variable=interval) for number in (2, 3)]
    mismatch = data_diff(observed, replicas).variables[0]
    assert mismatch.predictive_unavailable_reason is None
    assert (
        "Measurement definitions differ; statistics describe each side separately"
        in mismatch.comparison_issues
    )
    assert mismatch.left[0].variable is not None
    assert mismatch.right[0].variable is not None
    assert mismatch.left[0].variable.aggregation == "last"
    assert mismatch.right[0].variable.aggregation == "mean"
    renamed = replace(
        observed,
        variables=(
            type(observed.variables[0]).model_validate(
                {**observed.variables[0].model_dump(), "name": "Renamed"}
            ),
        ),
    )
    assert not data_diff(observed, renamed).variables[0].comparison_issues
    # A first-in-window observation is anchored at support_start, not support_end.
    first = _dataset(
        [1, 2, 3],
        variable=type(observed.variables[0]).model_validate(
            {**observed.variables[0].model_dump(), "aggregation": "first"}
        ),
    )
    first = replace(
        first,
        observations=first.observations.with_columns(
            pl.col("support_end") + timedelta(days=1),
        ),
    )
    assert data_diff(first, first).variables[0].changes == ()
    floating = [replace(_dataset([1, 2, 3], number=n), time_origin=None) for n in (2, 3)]
    mismatch = data_diff(observed, floating).variables[0]
    assert mismatch.predictive_checks is None
    assert mismatch.right[0].time_origin is None
    assert mismatch.predictive_unavailable_reason is None
    assert mismatch.comparison_issues == (
        "Calendar-free histories cannot be aligned to calendar-bound histories",
    )


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
def test_reads_saved_draws_without_generation_or_writing_models(tmp_path, monkeypatch):
    from nof1_causal_lab.artifacts.data_preparation import (
        PreparedDataMetadata,
        SimulationReplicateRef,
    )
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("DIFF"), StudyRepository("DIFF")
    model = store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={},
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    observed = _dataset([1, 2, 3])
    values = np.asarray([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]], dtype=float)
    report = SimulationReport(
        model=GitRef(workspace_id="DIFF", revision=model.revision, path="model.json"),
        design=SimulationSpec(end=2),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        predictive={
            "states": {},
            "indicators": {
                "indicator:y": {
                    "label": "Y",
                    "action": {
                        "kind": "numeric",
                        "mean": [1, 2, 3],
                        "lower": [0.05, 1.05, 2.05],
                        "upper": [1.95, 2.95, 3.95],
                        "n_draws": [3, 3, 3],
                    },
                }
            },
            "fit_reliability": "not_fitted",
        },
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
        AttemptRecord(
            seq=1,
            ts="2026-09-28T00:00:00Z",
            action="simulate",
            inputs={},
            status="applied",
            trace_ids=[],
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
        produced_by="prepare_data",
        derived_from={},
        json_files={
            "metadata.json": PreparedDataMetadata(
                source=SimulationReplicateRef(revision=commit, replicate=1),
                variables=observed.variables,
                time_origin=origin,
            ).model_dump(mode="json")
        },
        parquet_files={"panel.parquet": shifted},
    )
    request = DataDiffRequest(
        left=DataRef(kind="simulation", revision=commit),
        right=DataRef(kind="panel", revision=panel.revision),
    )
    refs_before = sorted(store.repo.references)
    result = read_data_diff("DIFF", request).model_dump(mode="json")
    assert len(result["left"]) == 3
    assert result["variables"][0]["left"][0]["time_origin"] == "2026-01-01T00:00:00Z"
    assert result["variables"][0]["predictive_checks"]["n_subsample"] == 3
    assert history.head() == commit
    assert sorted(store.repo.references) == refs_before
    assert store.list_revisions("model") == [model.revision]
    assert isinstance(request.left, DataRef)
    one = type(request).model_validate(
        {
            **request.model_dump(),
            "left": type(request.left).model_validate(
                {**request.left.model_dump(), "replicate": 1}
            ),
        }
    )
    assert read_data_diff("DIFF", one).variables[0].changes == ()
    bad = type(request).model_validate(
        {**request.model_dump(), "left": DataRef(kind="simulation", revision=commit, replicate=3)}
    )
    with pytest.raises(ValueError, match="replicate"):
        read_data_diff("DIFF", bad)
    bad = type(request).model_validate(
        {**request.model_dump(), "right": DataRef(kind="panel", revision=git_oid(98))}
    )
    with pytest.raises((KeyError, FileNotFoundError)):
        read_data_diff("DIFF", bad)

    # Saved histories keep absolute model days. Materialization alone resets the
    # origin of a calendar-free replicate; calendar-bound histories keep dates.
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel

    for index, origin in enumerate((None, datetime(2026, 1, 1, tzinfo=UTC))):
        commits = []
        for start in (5, 6):
            times = np.arange(start, start + 3.0)
            saved = type(report).model_validate(
                {
                    **report.model_dump(),
                    "time_origin": origin,
                    "design": SimulationSpec(start=start, end=start + 2),
                    "times": tuple(times),
                    "observation_layout": type(report.observation_layout).model_validate(
                        {
                            **report.observation_layout.model_dump(),
                            "support_start_times": store.write_array(times[:, None]),
                            "support_end_times": store.write_array(times[:, None]),
                        }
                    ),
                }
            )
            commits.append(
                history.append(
                    AttemptRecord(
                        seq=2 + index * 2 + start - 5,
                        ts="2026-09-28T00:00:00Z",
                        action="simulate",
                        status="applied",
                        trace_ids=[],
                        diagnostics={"report": saved.model_dump(mode="json")},
                    )
                )
            )
        aligned = read_data_diff(
            "DIFF",
            DataDiffRequest(
                left=DataRef(kind="simulation", revision=commits[0], replicate=1),
                right=DataRef(kind="simulation", revision=commits[1], replicate=1),
            ),
        ).variables[0]
        epoch = origin or datetime(1970, 1, 1, tzinfo=UTC)
        assert [(change.anchor_time, change.change) for change in aligned.changes] == [
            (epoch + timedelta(days=5), "removed"),
            (epoch + timedelta(days=6), "revised"),
            (epoch + timedelta(days=7), "revised"),
            (epoch + timedelta(days=8), "added"),
        ]
        assert aligned.left[0].time_origin == aligned.right[0].time_origin == origin
        materialized = prepare_simulation_panel(saved, 1, read_array=store.read_array)
        assert materialized["anchor_time"][0] == (
            origin + timedelta(days=6) if origin is not None else epoch
        ).replace(tzinfo=None)
