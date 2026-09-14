"""Lineage validation uses canonical IDs and original artifact pins."""

from dataclasses import replace

import polars as pl

from nof1_causal_lab.artifacts.raw_data import with_column_descriptions
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.model_checks import check_execution
from scripts import validate_run
from tests.helpers import complete_test_model, graph_constructs, make_model


def _context(artifacts, *, pins=None, model_indicators=None, raw_columns=None):
    state = EpisodeState().with_versions(
        [
            ArtifactVersionInfo(
                artifact_id=name,
                version=1,
                provenance="computed",
                derived_from=(pins or {}).get(name, {}),
            )
            for name in artifacts
        ]
    )
    return validate_run.RunContext("ws", state, artifacts, {}, model_indicators, raw_columns)


def test_source_columns_use_raw_parquet_columns():
    model = make_model(["observed_value"])
    payload = model.model_dump(mode="json")
    graph_constructs(payload)[0]["indicators"][0]["source_columns"] = ["value"]
    ctx = _context(
        {"model": payload},
        raw_columns={"timestamp", "value"},
    )
    assert validate_run.rule_source_columns_in_raw_data(ctx) == []
    bad = replace(ctx, raw_input_columns={"timestamp"})
    [issue] = validate_run.rule_source_columns_in_raw_data(bad)
    assert issue.rule == "source-columns-in-raw-data"
    assert "value" in issue.message


def test_panel_coverage_requires_a_panel_and_uses_its_model_pin():
    model = make_model(["declared"])
    ctx = _context({"model": model.model_dump(mode="json")}, model_indicators=set())
    assert validate_run.rule_indicators_in_panel(ctx) == []
    extracted = _context(
        {**ctx.artifacts, "panel": {}},
        model_indicators=set(),
        pins={"panel": {"model": 1}},
    )
    [issue] = validate_run.rule_indicators_in_panel(extracted)
    assert model.indicators[0].id in issue.message


def test_loading_cutoff_does_not_open_downstream_panels(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    store = ArtifactStore("ws")
    raw = store.write_version(
        "raw_data",
        provenance="computed",
        derived_from={},
        produced_by="run:raw_data",
        parquet_files={
            "raw.parquet": with_column_descriptions(
                pl.DataFrame({"timestamp": [0], "value": [1]}).to_arrow(),
                {"timestamp": "Observation time", "value": "Observed value"},
            )
        },
    )
    model = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": make_model(["value"]).model_dump(mode="json")},
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"indicator_id": ["indicator:downstream"]})},
    )
    monkeypatch.setattr(
        validate_run,
        "derive_current_state",
        lambda _: EpisodeState().with_versions([raw, model, panel]),
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
    store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    latest = store.write_version(
        "model",
        provenance="human",
        derived_from={"model": 1},
        produced_by="write:model",
        json_files={"model.json": make_model(["other"]).model_dump(mode="json")},
    )
    ctx = validate_run.RunContext(
        "ws",
        EpisodeState().with_versions([latest]),
        {
            "model": make_model(["other"]).model_dump(mode="json"),
        },
        {},
        None,
        None,
    )
    assert validate_run.rule_pinned_model_contracts(ctx) == []
