from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

import nof1_causal_lab.tool_server as tool_server
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from tests.helpers import fixture_entity_id
from tests.inference_fixtures import inference_log
from tests.model_fixtures import (
    parameter_draws,
)


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
        called = True
        return {"result": "should not run"}

    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        fake_impl,
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {}},
    )

    assert response.status_code == 422
    assert called is False


def test_execute_tool_surfaces_unexpected_exception_detail(monkeypatch):
    client = TestClient(tool_server.app)

    monkeypatch.setattr(tool_server, "_build_context", lambda *_args, **_kwargs: {})
    monkeypatch.setitem(
        tool_server._TOOL_IMPLS,
        ("latent-structure", "validate_latent_structure"),
        lambda _ctx, _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/api/tools/latent-structure/validate_latent_structure",
        json={"workspace_id": "user-123", "input": {"model_json": "{}"}},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "message": "boom",
            "exception_type": "RuntimeError",
            "context_id": "latent-structure",
            "tool_name": "validate_latent_structure",
        }
    }


def test_build_analysis_context_rehydrates_runtime_from_persisted_spec(monkeypatch, tmp_path):
    import polars as pl

    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.machine.moves import RunOperation
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.models.model_checks import check_execution
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import complete_test_model, make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    design = complete_test_model(
        make_model(["screen_time", "sleep_quality"], [("screen_time", "sleep_quality")])
    )
    design = design.revised(default_outcome=fixture_entity_id("construct", "sleep_quality"))
    check_execution(design)
    conditioned = condition_model(
        design,
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
    definition = store.write_version(
        "model",
        provenance="llm",
        derived_from={},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": design.model_dump(mode="json")},
    )

    identification_report = store.write_version(
        "identification_report",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="derive:identification_report",
        json_files={
            "identification_report.json": _identification(
                "screen_time", "sleep_quality"
            ).model_dump(mode="json")
        },
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": model_data},
    )
    fitted = store.write_version(
        "model",
        provenance="computed",
        derived_from={"model": 1, "panel": 1},
        produced_by="run:posterior",
        json_files={"model.json": conditioned.model_dump(mode="json")},
    )
    journal = EpisodeJournal("user-123")
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
            TransitionRecord(
                seq=seq,
                ts="2026-07-03T00:00:00+00:00",
                move=RunOperation(operation_id=operation),
                status="applied",
                produced=produced,
                diagnostics=inference_log(conditioned, report=report).diagnostics
                if operation == "posterior"
                else {},
                trace_ids=[],
                resume=None,
            )
        )

    rebuilt_runtime = SimpleNamespace(
        observation_support="support-runtime",
        observation_data=None,
    )
    captured: dict[str, Any] = {}
    loads = 0

    def fake_prepare_model_runtime(*, data_for_model, model, model_spec, sampler_config=None):
        nonlocal loads
        loads += 1
        del sampler_config
        assert model_spec == conditioned
        captured["data_for_model"] = data_for_model
        captured["model"] = model
        return rebuilt_runtime

    monkeypatch.setattr(tool_server, "prepare_model_runtime", fake_prepare_model_runtime)

    ctx = tool_server._build_analysis_context("user-123")

    assert isinstance(captured["model"], tool_server.SSMModel)
    # The runtime uses the model revision and panel pinned by the posterior.
    assert captured["model"].spec == conditioned
    assert list(numeric.state_names(captured["model"].spec)) == ["screen_time", "sleep_quality"]
    assert captured["data_for_model"].equals(model_data)
    assert ctx["_prepared_runtime"] is rebuilt_runtime
    assert ctx["model"] == conditioned.model_dump(mode="json")
    assert ctx["inference_report"] == report
    assert ctx["_outcome_name"] == "sleep_quality"
    assert ctx["_identifiable_treatments"] == ["screen_time"]
    again = tool_server._build_analysis_context("user-123")
    assert again["_simulation"] is ctx["_simulation"]
    assert loads == 1
    new_panel = store.write_version(
        "panel", provenance="computed", derived_from={"model": 1}, produced_by="run:measurements"
    )
    journal.append(
        TransitionRecord(
            seq=4,
            ts="2026-07-03T01:00:00+00:00",
            move=RunOperation(operation_id="measurements"),
            status="applied",
            produced=[new_panel],
            trace_ids=[],
            resume=None,
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

    from tests.helpers import complete_test_model, make_model

    model = make_model(["screen_time", "sleep"], [("screen_time", "sleep")])
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(
                    update={"indicators": (c.indicators[0].model_copy(update={"name": name}),)}
                )
                for c, name in zip(
                    model.constructs, ["daily_event_count", "sleep_issue_searches"], strict=True
                )
            ),
        ),
        default_outcome=model.constructs[1].id,
    )
    model = complete_test_model(model)
    spec = model
    ctx = {
        "model": model.model_dump(mode="json"),
        "inference_report": {"inference_metadata": {"method": "marginal_particle_gibbs"}},
        "_prepared_runtime": SimpleNamespace(spec=spec),
        "_fitted_artifact": SimpleNamespace(spec=spec),
        "_identifiable_treatments": ["screen_time"],
        "_outcome_name": "sleep",
        "_observation_timestamps": [],
    }

    payload = tool_server._build_model_info_payload(
        ctx,
        {"sections": ["overview", "variables", "capabilities"]},
    )

    assert payload["overview"]["treatments"] == ["screen_time"]
    assert [item["name"] for item in payload["variables"]["constructs"]] == ["screen_time", "sleep"]
    assert [item["id"] for item in payload["variables"]["constructs"]] == [
        construct.id for construct in model.constructs
    ]
    assert [item["name"] for item in payload["variables"]["indicators"]] == [
        "daily_event_count",
        "sleep_issue_searches",
    ]
    assert payload["capabilities"]["simulate"]["supported_targets"] == ["screen_time"]
