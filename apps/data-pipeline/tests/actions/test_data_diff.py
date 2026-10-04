"""Stored histories compare symmetrically without generation or scientific state changes."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.data_diff import read_data_diff
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.data_preparation import FilePreparedDataMetadata
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.observations import ResolvedObservationSpec
from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.simulation import (
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.models.posterior_predictive import data_diff
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, ModelSimulationResult
from nof1_causal_lab.study.store import ArtifactStore, read_dataset
from nof1_causal_lab.study.view_models import DataDiffRequest, PanelRef, SimulationRef
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revisions, git_oid


def _dataset(values, *, number=1, times=None, variable=None):
    variable = variable or ResolvedObservationSpec(
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
            "observation_window": [variable.observation_window.source] * len(values),
        },
        schema_overrides={"value": pl.Float64},
    )
    return read_dataset(PanelRef(revision=git_oid(number)), (variable,), frame, origin)


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
            SimulationRef.model_validate({"revision": git_oid(1), **invalid})


@pytest.mark.inference(concern="predictive")
def test_single_histories_report_values_missingness_and_schedule_changes():
    left = _dataset([1, None, 3], times=[0, 1, 2])
    right = _dataset([1, 4, 8], times=[0, 1, 3], number=2)
    result = data_diff(left, right)
    variable = result.variables[0]
    assert [item.kind for item in variable.changes] == ["revised", "removed", "added"]
    revised = variable.changes[0]
    assert revised.kind == "revised"
    assert revised.before.value is None
    assert revised.after.value == 4
    assert variable.predictive.kind == "not_applicable"
    assert variable.comparison_issues == ("Observation schedules or measurement windows differ",)
    missing = next(item for item in variable.statistics if item.statistic == "missing_count")
    assert (missing.left, missing.right) == ((1,), (0,))
    reverse = data_diff(right, left).variables[0]
    assert [item.kind for item in reverse.changes] == ["revised", "added", "removed"]
    assert data_diff(left, left).variables[0].changes == ()


@pytest.mark.inference(concern="predictive")
def test_replica_checks_are_symmetric_and_preserve_whole_history_statistics():
    observed = _dataset([0, 1, None, 3])
    replicas = [
        _dataset([offset, 1 + offset, 1000, 3 + offset], number=offset + 2) for offset in range(3)
    ]
    forward = data_diff(replicas, observed).variables[0]
    backward = data_diff(observed, replicas).variables[0]
    assert forward.predictive.kind == "comparison"
    assert backward.predictive.kind == "comparison"
    assert forward.predictive.reference_side == "right"
    assert backward.predictive.reference_side == "left"
    assert forward.predictive.evaluation == backward.predictive.evaluation
    assert forward.predictive.evaluation.kind == "available"
    stats = {item.stat_name: item for item in forward.predictive.evaluation.value.test_stats}
    assert stats["mean"].observed_value == pytest.approx(4 / 3)
    assert stats["mean"].rep_values == pytest.approx([4 / 3, 7 / 3, 10 / 3])
    assert forward.predictive.evaluation.value.n_subsample == 3
    assert len(forward.left) == 3
    # Many-to-many retains one summary per history, without implying paired draws.
    many = data_diff(replicas, [observed, _dataset([0, 1, 2, 3], number=9)]).variables[0]
    assert many.predictive.kind == "unavailable"
    means = next(item for item in many.statistics if item.statistic == "mean")
    assert len(means.left) == 3
    assert len(means.right) == 2


@pytest.mark.inference(concern="predictive")
def test_measurement_mismatches_and_uncovered_times_do_not_produce_predictive_checks():
    observed = _dataset([1, 2, 3])
    replicas = [_dataset([1, 2, 3], number=number, times=[0, 1, 2.5]) for number in (2, 3)]
    mismatch = data_diff(observed, replicas).variables[0]
    assert mismatch.predictive.kind == "comparison"
    assert mismatch.predictive.reference_side == "left"
    assert mismatch.predictive.evaluation.kind == "not_applicable"
    assert mismatch.comparison_issues == ("Observation schedules or measurement windows differ",)
    missing = data_diff(observed, [_dataset([1, None, 3], number=n) for n in (2, 3)]).variables[0]
    assert missing.predictive.kind == "comparison"
    assert missing.predictive.reference_side == "left"
    assert missing.predictive.evaluation.kind == "unavailable"
    assert (
        missing.predictive.evaluation.reason
        == "Replicas contain missing values at observed anchors"
    )
    assert missing.comparison_issues == ()
    original_variable = next(iter(observed.series.values())).variable
    assert original_variable is not None
    interval = original_variable.revised(aggregation="mean")
    replicas = [_dataset([1, 2, 3], number=number, variable=interval) for number in (2, 3)]
    mismatch = data_diff(observed, replicas).variables[0]
    assert mismatch.predictive.kind == "comparison"
    assert mismatch.predictive.evaluation.kind == "not_applicable"
    assert (
        "Measurement definitions differ; statistics describe each side separately"
        in mismatch.comparison_issues
    )
    assert mismatch.left[0].variable is not None
    assert mismatch.right[0].variable is not None
    assert mismatch.left[0].variable.aggregation == "last"
    assert mismatch.right[0].variable.aggregation == "mean"
    renamed = _dataset(
        [1, 2, 3],
        variable=original_variable.revised(name="Renamed"),
    )
    assert not data_diff(observed, renamed).variables[0].comparison_issues
    # A first-in-window observation is anchored at support_start, not support_end.
    first = _dataset(
        [1, 2, 3],
        variable=original_variable.revised(aggregation="first"),
    )
    # Interval coordinates were parsed once by read_dataset; comparisons own no frame.
    assert data_diff(first, first).variables[0].changes == ()
    floating = [
        type(item)(
            source=item.source,
            series={
                key: type(series)(variable=series.variable, time_origin=None, points=series.points)
                for key, series in item.series.items()
            },
        )
        for item in [_dataset([1, 2, 3], number=n) for n in (2, 3)]
    ]
    mismatch = data_diff(observed, floating).variables[0]
    assert mismatch.predictive.kind == "comparison"
    assert mismatch.predictive.evaluation.kind == "not_applicable"
    assert mismatch.right[0].time_origin is None
    assert mismatch.comparison_issues == (
        "Calendar-free histories cannot be aligned to calendar-bound histories",
    )


@pytest.mark.inference(concern="predictive")
def test_discrete_codebooks_compare_frequencies_and_keep_absent_variables_explicit():
    variable = ResolvedObservationSpec(
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
    assert result.predictive.kind == "comparison"
    assert result.predictive.evaluation.kind == "unavailable"
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
        causal=NotApplicable(reason="No intervention was requested."),
        fit_reliability="not_fitted",
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            model=GitRef(workspace_id="DIFF", revision=model.revision, path="model.json"),
            design=SimulationSpec(start=date(2026, 1, 1), horizon="2d"),
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            times=(0, 1, 2),
            draws=3,
            seed=0,
            state_ids=(),
            parameter_draws={},
            latent_paths="unreadable-latent-array",
            observations=store.write_array(values),
            observation_layout=SimulationObservationLayout(
                variables=tuple(series.variable for series in observed.series.values()),
                support_start_times=store.write_array(np.arange(3.0)[:, None]),
                support_end_times=store.write_array(np.arange(3.0)[:, None]),
                mask=store.write_array(np.ones_like(values, dtype=bool)),
            ),
        ),
    )
    commit = history.append(
        applied_record(
            Applied(
                result=ModelSimulationResult(evidence=(report).evidence), effects=ActionEffects()
            ),
            seq=1,
            ts="2026-09-28T00:00:00Z",
            trace_ids=[],
        )
    ).commit_id
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    shifted = pl.DataFrame(
        {
            "indicator_id": ["indicator:y"] * 3,
            "value": [1.0, 2.0, 3.0],
            "anchor_time": [origin + timedelta(days=n) for n in range(3)],
            "support_start": [origin + timedelta(days=n) for n in range(3)],
            "support_end": [origin + timedelta(days=n) for n in range(3)],
            "support_kind": ["point"] * 3,
            "summary_operator": ["last"] * 3,
            "anchor_policy": ["support_end"] * 3,
            "observation_window": ["1d"] * 3,
        }
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={
            "metadata.json": FilePreparedDataMetadata(
                source=SimulationReplicateRef(revision=commit, replicate=1), time_origin=origin
            ).model_dump(mode="json")
        },
        parquet_files={"panel.parquet": shifted},
    )
    request = DataDiffRequest(
        left=SimulationRef(revision=commit),
        right=PanelRef(revision=panel.revision),
    )
    refs_before = sorted(store.repo.references)
    result = read_data_diff("DIFF", request).model_dump(mode="json")
    assert len(result["left"]) == 3
    assert result["variables"][0]["left"][0]["time_origin"] == "2026-01-01T00:00:00Z"
    assert result["variables"][0]["predictive"]["evaluation"]["value"]["n_subsample"] == 3
    assert history.head() == commit
    assert sorted(store.repo.references) == refs_before
    assert artifact_revisions(store, "model") == [model.revision]
    assert isinstance(request.left, SimulationRef)
    one = request.revised(left=request.left.revised(replicate=1))
    assert read_data_diff("DIFF", one).variables[0].changes == ()
    bad = request.revised(left=SimulationRef(revision=commit, replicate=3))
    with pytest.raises(StudyLookupError, match="replicate"):
        read_data_diff("DIFF", bad)
    bad = request.revised(right=PanelRef(revision=git_oid(98)))
    with pytest.raises(StudyLookupError):
        read_data_diff("DIFF", bad)

    # Saved histories keep absolute model days, and materialization keeps their dates.
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel

    origin = datetime(2026, 1, 1, tzinfo=UTC)
    commits = []
    for start in (5, 6):
        times = np.arange(start, start + 3.0)
        saved = report.revised(
            time_origin=origin,
            design=SimulationSpec(start=date(2026, 1, 1 + start), horizon="2d"),
            times=tuple(times),
            observation_layout=report.observation_layout.revised(
                support_start_times=store.write_array(times[:, None]),
                support_end_times=store.write_array(times[:, None]),
            ),
        )
        commits.append(
            history.append(
                applied_record(
                    Applied(
                        result=ModelSimulationResult(evidence=(saved).evidence),
                        effects=ActionEffects(),
                    ),
                    seq=2 + start - 5,
                    ts="2026-09-28T00:00:00Z",
                    trace_ids=[],
                )
            ).commit_id
        )
    aligned = read_data_diff(
        "DIFF",
        DataDiffRequest(
            left=SimulationRef(revision=commits[0], replicate=1),
            right=SimulationRef(revision=commits[1], replicate=1),
        ),
    ).variables[0]
    assert [
        (
            (change.before.anchor_time if change.kind == "removed" else change.after.anchor_time),
            change.kind,
        )
        for change in aligned.changes
    ] == [
        (origin + timedelta(days=5), "removed"),
        (origin + timedelta(days=6), "revised"),
        (origin + timedelta(days=7), "revised"),
        (origin + timedelta(days=8), "added"),
    ]
    assert aligned.left[0].time_origin == aligned.right[0].time_origin == origin
    materialized = prepare_simulation_panel(saved, 1, read_array=store.read_array)
    assert materialized["anchor_time"][0] == (origin + timedelta(days=6)).replace(tzinfo=None)
