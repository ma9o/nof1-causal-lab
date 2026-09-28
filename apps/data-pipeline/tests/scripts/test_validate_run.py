"""Lineage validation uses canonical IDs and original artifact pins."""

from dataclasses import replace

import polars as pl
import pytest

from nof1_causal_lab.artifacts.raw_data import with_column_descriptions
from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.model_checks import check_execution
from scripts import validate_run
from tests.data_fixtures import metadata_for_model
from tests.git_fixtures import artifact_revision, git_oid
from tests.helpers import complete_test_model, make_model

pytestmark = pytest.mark.contract


def _context(artifacts, *, pins=None, model_indicators=None, raw_columns=None):
    state = EpisodeState().with_artifacts(
        [
            ArtifactRecord(
                artifact_id=name,
                revision=git_oid(1),
                derived_from=(pins or {}).get(name, {}),
            )
            for name in artifacts
        ]
    )
    return validate_run.RunContext("ws", state, artifacts, {}, model_indicators, raw_columns)


def test_source_columns_use_raw_parquet_columns():
    model = make_model(["observed_value"])
    payload = metadata_for_model(model).model_dump(mode="json")
    payload["preparation"]["variables"][0]["source_columns"] = ["value"]
    ctx = _context(
        {"panel": payload},
        raw_columns={"timestamp", "value"},
    )
    assert validate_run.rule_source_columns_in_raw_data(ctx) == []
    bad = replace(ctx, raw_input_columns={"timestamp"})
    [issue] = validate_run.rule_source_columns_in_raw_data(bad)
    assert issue.rule == "source-columns-in-raw-data"
    assert "value" in issue.message


def test_panel_coverage_uses_its_own_schema_without_a_model():
    model = make_model(["declared"])
    ctx = _context({"model": model.model_dump(mode="json")}, model_indicators=set())
    assert validate_run.rule_indicators_in_panel(ctx) == []
    extracted = _context(
        {"panel": metadata_for_model(model).model_dump(mode="json")},
        model_indicators=set(),
    )
    [issue] = validate_run.rule_indicators_in_panel(extracted)
    assert model.indicators[0].id in issue.message


def test_loading_cutoff_does_not_open_downstream_panels(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    store = ArtifactStore("ws")
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="run:raw_data",
        parquet_files={
            "raw.parquet": with_column_descriptions(
                pl.DataFrame({"timestamp": [0], "value": [1]}).to_arrow(),
                {"timestamp": "Observation time", "value": "Observed value"},
            )
        },
    )
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": make_model(["value"]).model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"indicator_id": ["indicator:downstream"]})},
    )
    monkeypatch.setattr(
        validate_run,
        "read_current_state",
        lambda _: EpisodeState().with_artifacts([raw, model, panel]),
    )
    ctx = validate_run.load_run_context("ws", up_to="model")
    assert set(ctx.artifacts) == {"model"}
    assert ctx.model_indicators is None
    assert ctx.raw_input_columns == {"timestamp", "value"}
    assert validate_run.rule_contract_conformance(ctx) == []
    undescribed = replace(ctx, raw_table=pl.DataFrame({"value": [1]}).to_arrow())
    [issue] = validate_run.rule_contract_conformance(undescribed)
    assert issue.artifacts == ("raw_data",)
    assert "Missing description" in issue.message


def test_current_model_contract_allows_incomplete_scientific_revisions(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    store = ArtifactStore("ws")
    model = complete_test_model(make_model(["mood"]))
    check_execution(model)
    store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    latest = store.write_artifact(
        "model",
        derived_from={"model": artifact_revision("ws", "model", 1)},
        produced_by="write:model",
        json_files={"model.json": make_model(["other"]).model_dump(mode="json")},
    )
    ctx = validate_run.RunContext(
        "ws",
        EpisodeState().with_artifacts([latest]),
        {
            "model": make_model(["other"]).model_dump(mode="json"),
        },
        {},
        None,
        None,
    )
    assert validate_run.rule_pinned_model_contracts(ctx) == []
