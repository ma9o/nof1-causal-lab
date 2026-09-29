"""Pre-extracted observations preserve their schema and bypass extraction atomically."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts import PrepareDataRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks, read_data_metadata
from nof1_causal_lab.actions.execution import plan_execution
from nof1_causal_lab.artifacts.data_preparation import ObservationTableSpec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import PrepareObservationTableOperation
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.store import ArtifactStore
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("text_timestamps", [False, True])
def test_observation_table_selection_validation_and_publication(
    tmp_path, monkeypatch, text_timestamps
):
    from nof1_causal_lab.machine import store as store_module
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))

    def forbidden_model(*_args, **_kwargs):
        raise AssertionError("Import must not load a model")

    monkeypatch.setattr(store_module, "read_model", forbidden_model)
    variables = [
        {
            "id": f"indicator:{dtype}",
            "name": dtype,
            "measurement_dtype": dtype,
            "aggregation": "mean" if dtype == "continuous" else "last",
            "observation_window": "1d",
            **(
                {f"{dtype}_levels": ["none", "mild", "severe"]}
                if dtype in {"ordinal", "categorical"}
                else {}
            ),
        }
        for dtype in ("continuous", "ordinal", "categorical", "count", "binary")
    ]
    source = {"file": "panel.parquet", "start": "2026-01-02", "end": "2026-01-04"}
    payload = {"action": "prepare_data", "input": {"source": source, "variables": variables}}
    request = PrepareDataRequest.model_validate(payload)
    assert isinstance(request.input, ObservationTableSpec)
    command = plan_execution(request)
    assert isinstance(command.operation, PrepareObservationTableOperation)
    assert command.input_revisions == {}
    rows = [
        {
            "indicator_id": variable["id"],
            "value": None if day == 3 else 1.0,
            "anchor_time": datetime(2026, 1, day),
            "support_start": datetime(2026, 1, day) - timedelta(days=1),
            "support_end": datetime(2026, 1, day),
            "support_kind": "interval"
            if variable["measurement_dtype"] == "continuous"
            else "point",
            "summary_operator": variable["aggregation"],
            "anchor_policy": "support_end",
            "observation_window": "1d",
        }
        for variable in variables
        for day in range(1, 5)
    ]
    # Invalid undeclared rows must not leak into validation or into the selected panel.
    rows.append(
        {
            **rows[0],
            "indicator_id": "indicator:ignored",
            "value": float("inf"),
            "support_kind": "invalid",
        }
    )
    frame = pl.DataFrame(rows)
    if text_timestamps:
        frame = frame.with_columns(
            pl.col("anchor_time", "support_start", "support_end").dt.to_string("%Y-%m-%dT%H:%M:%S")
        )
    upload = tmp_path / "TEST" / "input" / "panel.parquet"
    upload.parent.mkdir(parents=True)
    frame.write_parquet(upload)
    state = EpisodeState()
    effects = run_async(execute_transition("TEST", command.operation, state))
    assert [item.artifact_id for item in effects.produced] == ["panel"]
    effects = evaluate_data_checks("TEST", state, effects)
    assert {item.artifact_id for item in effects.produced} == {"panel", "data_profile"}
    store = ArtifactStore("TEST")
    panel_info = effects.produced[0]
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    assert panel.height == 10
    assert panel["value"].null_count() == 5
    assert panel["anchor_time"].min() == datetime(2026, 1, 2)
    assert panel["anchor_time"].max() == datetime(2026, 1, 3)
    assert panel["support_start"].min() == datetime(2026, 1, 1)
    metadata = read_data_metadata(store, panel_info.revision)
    assert metadata.source.model_dump(mode="json") == source
    assert metadata.variables == request.input.variables
    assert metadata.preparation is None
    assert panel_info.derived_from == {}
    assert effects.diagnostics["n_observations"] == 5

    # Each rejected upload must fail before any artifact is staged, even with an existing panel.
    state = state.with_artifacts(effects.produced)

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("Invalid observations must not stage an artifact")

    monkeypatch.setattr(ArtifactStore, "write_artifact", forbidden_write)
    for dtype, field, value, message in (
        ("ordinal", "value", 3.0, "codes"),
        ("categorical", "value", -1.0, "codes"),
        ("ordinal", "value", 0.5, "codes"),
        ("binary", "value", 2.0, "0 or 1"),
        ("count", "value", -1.0, "non-negative integers"),
        ("count", "value", 0.5, "non-negative integers"),
        ("continuous", "value", float("nan"), "finite"),
        ("continuous", "support_kind", "point", "support_kind"),
        ("continuous", "summary_operator", "sum", "summary_operator"),
        ("continuous", "anchor_policy", "support_start", "anchor_policy"),
        ("continuous", "observation_window", "2d", "observation window"),
    ):
        bad = frame.with_columns(
            pl.when(pl.col("indicator_id") == f"indicator:{dtype}")
            .then(pl.lit(value))
            .otherwise(pl.col(field))
            .alias(field)
        )
        bad.write_parquet(upload)
        with pytest.raises(ValueError, match=message):
            run_async(execute_transition("TEST", command.operation, state))
    frame.filter(pl.col("indicator_id") != "indicator:ordinal").write_parquet(upload)
    with pytest.raises(ValueError, match="no rows"):
        run_async(execute_transition("TEST", command.operation, state))
    frame.drop("support_kind").write_parquet(upload)
    with pytest.raises(ValueError, match="missing canonical columns"):
        run_async(execute_transition("TEST", command.operation, state))
    for column in ("anchor_time", "support_start", "support_end"):
        frame.with_columns(pl.lit(None, dtype=frame.schema[column]).alias(column)).write_parquet(
            upload
        )
        with pytest.raises(ValueError, match=r"anchor_time|support boundaries"):
            run_async(execute_transition("TEST", command.operation, state))

    for changes in ({"file": "../panel.parquet"}, {"end": source["start"]}):
        with pytest.raises(ValidationError):
            PrepareDataRequest.model_validate(
                {"input": {"source": {**source, **changes}, "variables": variables}}
            )
    with pytest.raises(ValidationError, match="resolved observation windows"):
        PrepareDataRequest.model_validate(
            {
                "input": {
                    "source": source,
                    "variables": [{**variables[0], "observation_window": None}],
                }
            }
        )


def test_observation_table_fill_null_is_bounded_and_retained(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    variables = [
        {
            "id": f"indicator:{name}",
            "name": name,
            "measurement_dtype": "count" if name == "zero" else "continuous",
            "aggregation": "sum" if name == "zero" else "last",
            "observation_window": "1d",
            "fill_null": fill_null,
        }
        for name, fill_null in (
            ("unfilled", None),
            ("forward", "forward"),
            ("zero", 0),
        )
    ]
    source = {"file": "panel.parquet", "start": "2026-01-02", "end": "2026-01-08"}
    request = PrepareDataRequest.model_validate(
        {"input": {"source": source, "variables": variables}}
    )
    assert isinstance(request.input, ObservationTableSpec)
    frame = pl.DataFrame(
        [
            {
                "indicator_id": variable.id,
                "value": value,
                "anchor_time": datetime(2026, 1, day),
                "support_start": datetime(2026, 1, day) - timedelta(days=1),
                "support_end": datetime(2026, 1, day),
                "support_kind": variable.support_kind.value,
                "summary_operator": variable.summary_operator.value,
                "anchor_policy": variable.anchor_policy.value,
                "observation_window": "1d",
            }
            for variable in request.input.variables
            for day, value in ((1, 99.0), (2, None), (3, 10.0), (4, None), (6, 20.0), (8, 40.0))
        ]
    )
    upload = tmp_path / "TEST" / "input" / "panel.parquet"
    upload.parent.mkdir(parents=True)
    frame.write_parquet(upload)
    command = plan_execution(request)
    assert isinstance(command.operation, PrepareObservationTableOperation)
    effects = run_async(execute_transition("TEST", command.operation, EpisodeState()))
    panel_info = effects.produced[0]
    store = ArtifactStore("TEST")
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    for name, days, values in (
        ("unfilled", [2, 3, 4, 6], [None, 10.0, None, 20.0]),
        ("forward", [2, 3, 4, 5, 6, 7], [None, 10.0, 10.0, 10.0, 20.0, 20.0]),
        ("zero", [2, 3, 4, 5, 6, 7], [0.0, 10.0, 0.0, 0.0, 20.0, 0.0]),
    ):
        rows = panel.filter(pl.col("indicator_id") == f"indicator:{name}")
        assert rows["anchor_time"].dt.day().to_list() == days
        assert rows["value"].to_list() == values
        assert rows["support_end"].equals(rows["anchor_time"])
        assert rows["support_start"].equals(rows["anchor_time"] - timedelta(days=1))
    metadata = read_data_metadata(store, panel_info.revision)
    assert metadata.source.model_dump(mode="json") == source
    assert metadata.variables == request.input.variables
    assert [item.model_dump(exclude_none=True).get("fill_null") for item in metadata.variables] == [
        None,
        "forward",
        0,
    ]
    assert metadata.preparation is None
    checked = evaluate_data_checks("TEST", EpisodeState(), effects)
    assert checked.diagnostics["n_observations"] == 13

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("Ambiguous null filling must not stage an artifact")

    monkeypatch.setattr(ArtifactStore, "write_artifact", forbidden_write)
    pl.concat([frame, frame.filter(pl.col("indicator_id") == "indicator:forward")]).write_parquet(
        upload
    )
    with pytest.raises(ValueError, match="one row per indicator"):
        run_async(execute_transition("TEST", command.operation, EpisodeState()))

    # Filling cannot bypass the declared value domain.
    frame.write_parquet(upload)
    invalid = PrepareDataRequest.model_validate(
        {"input": {"source": source, "variables": [{**variables[2], "fill_null": 0.5}]}}
    )
    invalid_operation = plan_execution(invalid).operation
    assert isinstance(invalid_operation, PrepareObservationTableOperation)
    with pytest.raises(ValueError, match="non-negative integers"):
        run_async(execute_transition("TEST", invalid_operation, EpisodeState()))


@pytest.mark.parametrize(
    "fields",
    [
        {"recording": "changes"},
        {"fill_null": {}},
        {"fill_null": {"strategy": "forward"}},
        {"fill_null": "interpolate"},
        {"fill_null": 0, "fill_null_limit": 1},
        {"fill_null": "mean", "fill_null_limit": 1},
        {"fill_null": "forward", "fill_null_limit": -1},
        {"fill_null": float("inf")},
        {"fill_null": "0"},
        {"fill_null": False},
        {"fill_null_limit": 1},
    ],
)
def test_fill_null_contract_rejects_non_polars_arguments(fields):
    with pytest.raises(ValidationError):
        ObservationTableSpec.model_validate(
            {
                "source": {"file": "panel.parquet"},
                "variables": [
                    {
                        "id": "indicator:dose",
                        "name": "dose",
                        "measurement_dtype": "continuous",
                        "aggregation": "last",
                        "observation_window": "1d",
                        **fields,
                    }
                ],
            }
        )
