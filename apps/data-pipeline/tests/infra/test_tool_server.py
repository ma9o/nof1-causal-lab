from __future__ import annotations

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from pathlib import Path

from typing import Any

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from pydantic.json_schema import JsonSchemaValue

import nof1_causal_lab.tool_server as tool_server
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
)
from nof1_causal_lab.artifacts.posterior import InferenceMetadata, InferenceReport
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.tool_contracts import GetModelInfoInput
from tests.git_fixtures import artifact_revision
from tests.helpers import fixture_entity_id
from tests.inference_fixtures import inference_log
from tests.model_fixtures import (
    compile_fit_fixture,
    parameter_draws,
)

pytestmark = pytest.mark.contract


def _identification(treatment: str, outcome: str) -> IdentificationReport:
    return IdentificationReport(
        outcome=fixture_entity_id("construct", outcome),
        treatments={
            fixture_entity_id("construct", treatment): IdentifiedTreatmentStatus(
                method="do_calculus",
                estimand=f"E[{outcome} | do({treatment})]",
            )
        },
    )


def test_execute_tool_rejects_invalid_input_before_invoking_tool(monkeypatch):
    client = TestClient(tool_server.app)
    called = False

    def fake_impl(_ctx, _args):
        nonlocal called
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("literature", "search_literature"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/literature/search_literature",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_execute_tool_surfaces_unexpected_exception_detail(monkeypatch):
    client = TestClient(tool_server.app)

    monkeypatch.setattr(tool_server, "_build_context", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("literature", "search_literature"),
        lambda _ctx, _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/api/tools/literature/search_literature",
        json={"workspace_id": "user-123", "input": {"query": "sleep"}},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "message": "boom",
            "exception_type": "RuntimeError",
            "context_id": "literature",
            "tool_name": "search_literature",
        }
    }


def test_build_analysis_context_loads_joint_laws_without_fit_compilation(monkeypatch, tmp_path):
    import polars as pl

    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import AttemptRecord
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    design = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'tool_server/build_analysis_context_loads_joint_laws_without_fit_compilation_complete_test_model.json').read_text())
    design = design.revised(default_outcome=fixture_entity_id("construct", "sleep_quality"))
    check_execution(design)
    conditioned = condition_model(
        compile_fit_fixture(design),
        ParticleMCMCPosterior(
            JointPosteriorDraws(parameter_draws(design, 1), jnp.zeros((1, 1, 2)))
        ),
        times=jnp.array([0.0]),
    )
    report = InferenceReport.model_validate(
        inference_log(conditioned).diagnostics["report"]
    ).model_dump(mode="json")
    model_data = pl.DataFrame(
        {
            "indicator_id": [design.indicators[1].id],
            "value": [1.0],
            "anchor_time": ["2024-01-01T00:00:00"],
        }
    )
    store = ArtifactStore("user-123")
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": design.model_dump(mode="json")},
    )

    identification_report = store.write_artifact(
        "identification_report",
        derived_from={"model": artifact_revision("user-123", "model", 1)},
        produced_by="derive:identification_report",
        json_files={
            "identification_report.json": _identification(
                "screen_time", "sleep_quality"
            ).model_dump(mode="json")
        },
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"panel.parquet": model_data},
    )
    fitted = store.write_artifact(
        "model",
        derived_from={
            "model": artifact_revision("user-123", "model", 1),
            "panel": artifact_revision("user-123", "panel", 1),
        },
        produced_by="fit",
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    journal = StudyRepository("user-123")
    for seq, operation, produced in (
        (
            1,
            "statistical_model_spec",
            [definition, identification_report],
        ),
        (2, "measurements", [panel]),
        (3, "posterior", [fitted]),
    ):
        journal.append(
            AttemptRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                action="fit"
                if operation == "posterior"
                else "prepare_data"
                if operation in {"raw_data", "measurements"}
                else "edit_model",
                inputs={},
                status="applied",
                produced=produced,
                diagnostics=inference_log(conditioned, report=report).diagnostics
                if operation == "posterior"
                else {},
                trace_ids=[],
            )
        )

    captured: dict[str, Any] = {}
    loads = 0

    def fake_project_observation_data(*, data_for_model, model_spec, time_origin):
        nonlocal loads
        loads += 1
        assert model_spec == conditioned
        assert time_origin.isoformat() == report["time_origin"].replace("Z", "+00:00")
        captured["data_for_model"] = data_for_model
        captured["model"] = model_spec
        return pl.DataFrame(), pl.DataFrame()

    monkeypatch.setattr(tool_server, "project_observation_data", fake_project_observation_data)

    ctx = tool_server._build_analysis_context("user-123")

    # The runtime uses the model revision and panel pinned by the posterior.
    assert captured["model"] == conditioned
    assert list(numeric.state_names(captured["model"])) == ["screen_time", "sleep_quality"]
    assert captured["data_for_model"].equals(model_data)
    assert ctx["model"] == conditioned
    assert ctx["inference_report"].model_dump(mode="json") == report
    assert ctx["_outcome_name"] == "sleep_quality"
    assert ctx["_identifiable_treatments"] == ["screen_time"]
    again = tool_server._build_analysis_context("user-123")
    assert again["_simulation"] is ctx["_simulation"]
    assert loads == 1
    new_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
    )
    journal.append(
        AttemptRecord(
            seq=4,
            ts="2026-07-03T01:00:00+00:00",
            action="prepare_data",
            inputs={},
            status="applied",
            produced=[new_panel],
            trace_ids=[],
        )
    )
    with pytest.raises(tool_server.HTTPException, match="conditioned model") as stale:
        tool_server._build_analysis_context("user-123")
    assert stale.value.status_code == 409
    assert loads == 1
    tool_server._load_simulation.cache_clear()


def test_get_tool_schemas_exposes_declared_result_schema():
    client = TestClient(tool_server.app)

    response = client.get("/api/tools/analysis")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["get_model_info"]["result"] is None
    assert "simulate" not in tools


def test_get_model_info_uses_structure_for_variables_and_treatments():

    from tests.helpers import make_model

    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'tool_server/get_model_info_uses_structure_for_variables_and_treatments_complete_test_model.json').read_text())
    ctx: tool_server.ToolContext = {
        "model": model,
        "inference_report": InferenceReport(
            time_origin=None,
            inference_metadata=InferenceMetadata(
                method="marginal_particle_gibbs", n_samples=1, duration_seconds=0
            ),
        ),
        "_identifiable_treatments": ["screen_time"],
        "_outcome_name": "sleep",
        "_observation_timestamps": [],
    }

    payload = tool_server._build_model_info_payload(
        ctx,
        GetModelInfoInput(sections=["overview", "variables", "capabilities"]),
    )

    overview = payload["overview"]
    assert isinstance(overview, dict)
    assert overview["treatments"] == ["screen_time"]
    variables = payload["variables"]
    assert isinstance(variables, dict)
    constructs = TypeAdapter(list[JsonObject]).validate_python(variables["constructs"])
    indicators = TypeAdapter(list[JsonObject]).validate_python(variables["indicators"])
    assert [item["name"] for item in constructs] == ["screen_time", "sleep"]
    assert [item["id"] for item in constructs] == [construct.id for construct in model.constructs]
    assert [item["name"] for item in indicators] == [
        "daily_event_count",
        "sleep_issue_searches",
    ]
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, dict)
    simulation = capabilities["simulate"]
    assert isinstance(simulation, dict)
    targets = TypeAdapter(list[str]).validate_python(simulation["intervention_targets"])
    request_schema = TypeAdapter(JsonSchemaValue).validate_python(simulation["request"])
    assert set(targets) == {c.id for c in model.constructs}
    assert set(request_schema["required"]) == {"model_revision", "end"}
    assert request_schema["properties"]["interventions"]["default"] == []
