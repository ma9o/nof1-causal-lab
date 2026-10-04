"""One forward-generation contract with dated interventions and durable histories."""

from datetime import UTC, date, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts import SimulateRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.artifacts.simulation import SimulationEvidence, SimulationReport, SimulationSpec
from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.study.records import Applied
from tests.inference_fixtures import compile_model_fixture, particle_posterior
from tests.model_fixtures import stress_sleep_model

pytestmark = pytest.mark.contract


def test_simulation_request_is_a_dated_window_with_optional_interventions():
    request = SimulateRequest(model_revision="a" * 40, start=date(2026, 1, 1), horizon="12d")
    assert request.interventions == ()
    assert set(request.model_dump()) == {
        "action",
        "model_revision",
        "panel_revision",
        "start",
        "horizon",
        "interventions",
    }
    with pytest.raises(ValidationError, match="Extra inputs"):
        request.revised(comparison_panel_revision="b" * 40)


def test_interventions_are_placed_after_the_start_in_model_days():
    spec = SimulationSpec(
        start=date(2026, 1, 3),
        horizon="3d",
        interventions=(
            InterventionSpec(target="construct:x", value=2),
            InterventionSpec(target="construct:x", after="1d", value=4),
        ),
    )
    assert SimulationSpec.model_validate_json(spec.model_dump_json()) == spec
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    assert (spec.start_day(origin), spec.end_day(origin)) == (2.0, 5.0)
    assert [(event.time, event.value) for event in spec.assignments(origin)] == [
        (2.0, 2.0),
        (3.0, 4.0),
    ]


@pytest.mark.parametrize(
    "events",
    [
        [{"after": "-1d", "value": 2}],
        [{"after": "3d", "value": 2}],
        [{"after": "4d", "value": 2}],
        [{"after": "1d", "value": float("nan")}],
        [{"after": "1d", "value": float("inf")}],
        [{"after": "1d", "value": [[1, 2], [2, 4]]}],
        [{"kind": "hold", "start": 0, "end": 2, "value": 1}],
        [{"value": 1}, {"value": 2}],
        [{"after": "1d", "value": 1}, {"after": "24h", "value": 2}],
    ],
)
def test_intervention_offsets_values_and_conflicts_are_explicit(events):
    with pytest.raises(ValidationError):
        SimulationSpec.model_validate(
            {
                "start": "2026-01-01",
                "horizon": "3d",
                "interventions": [{"target": "construct:x", **event} for event in events],
            }
        )


def _response(origin: datetime = datetime(2026, 1, 1, tzinfo=UTC)):
    start = (datetime(2026, 1, 1, tzinfo=UTC) - origin).total_seconds() / 86400
    days = [start, start + 1, start + 2]

    result = {
        "outcome": "construct:y",
        "labels": {"construct:x": "Treatment", "construct:y": "Outcome"},
        "warnings": [],
    }

    evidence = {
        "design": {
            "start": "2026-01-01",
            "horizon": "2d",
            "interventions": [{"target": "construct:x", "value": 1}],
        },
        "time_origin": origin.isoformat(),
        "assignments": [{"target": "construct:x", "time": start, "value": 1}],
        "times": days,
        "draws": 100,
        "seed": 0,
        "model": {"workspace_id": "QUERY", "revision": "a" * 40, "path": "model.json"},
        "state_ids": ["construct:x", "construct:y"],
        "observation_layout": {
            "variables": [
                {
                    "id": "indicator:y",
                    "name": "y",
                    "measurement_dtype": "continuous",
                    "aggregation": "last",
                    "observation_window": "1d",
                }
            ],
            "support_start_times": "starts",
            "support_end_times": "ends",
            "mask": "mask",
        },
        "parameter_draws": {},
        "latent_paths": "paths",
        "observations": "observations",
        "reference_latent_paths": "reference-paths",
        "reference_observations": "reference-observations",
    }
    return {"evidence": evidence, "law": AuthoredLawProvenance().model_dump(mode="json"),
        "fit_reliability": "converged", "causal": {"kind": "available", "value": result}}


def test_one_request_can_produce_independently_pinned_responses():
    value = _response()
    first = SimulationReport.model_validate(value)
    value["evidence"]["model"]["revision"] = "b" * 40
    second = SimulationReport.model_validate(value)
    assert first.evidence.design == second.evidence.design
    assert first.evidence.model.revision == "a" * 40
    assert second.evidence.model.revision == "b" * 40
    assert SimulationReport.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize("missing", ["construct:x", "construct:y"])
def test_causal_layout_includes_the_outcome_and_interventions(missing):
    value = _response()
    value["evidence"]["state_ids"].remove(missing)
    with pytest.raises(ValidationError, match="include the outcome and interventions"):
        SimulationReport.model_validate(value)


@pytest.mark.parametrize("missing", ["reference_latent_paths", "reference_observations"])
def test_causal_reports_require_their_effects_and_paired_draws(missing):
    payload = _response()
    del payload["evidence"][missing]
    with pytest.raises(ValidationError, match="paired reference"):
        TypeAdapter(SimulationReport).validate_python(payload)


@pytest.mark.parametrize(
    ("kind", "current_panel", "reliable", "scientific_model_payload"),
    [
        pytest.param(
            "authored",
            True,
            "not_fitted",
            stress_sleep_model,
            id="authored-True-not_fitted",
        ),
        pytest.param(
            "authored",
            False,
            "not_fitted",
            stress_sleep_model,
            id="authored-False-not_fitted",
        ),
        pytest.param(
            "fitted",
            True,
            "converged",
            stress_sleep_model,
            id="fitted-True-converged",
        ),
        pytest.param(
            "fitted",
            True,
            "unconverged",
            stress_sleep_model,
            id="fitted-True-unconverged",
        ),
        pytest.param(
            "unknown",
            True,
            "unknown",
            stress_sleep_model,
            id="unknown-True-unknown",
        ),
    ],
)
def test_runner_derives_origin_and_reliability_from_current_panel_or_fit(
    tmp_path, monkeypatch, kind, current_panel, reliable, scientific_model_payload
):
    from importlib import import_module

    import jax.numpy as jnp

    from nof1_causal_lab.actions.runners import run_action
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
    )
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data
    from tests.helpers import run_async, write_question
    from tests.inference_fixtures import _report, inference_log, parameter_draws
    from tests.integration.runner_fixtures import panel_metadata

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("ORIGIN")
    model = scientific_model_payload()
    prior = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    fit_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
    )
    record = prior
    fit_origin = datetime(2024, 1, 2, tzinfo=UTC)
    if kind in {"fitted", "unknown"}:
        model, _ = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
        )
        record = store.write_artifact(
            "model",
            derived_from={"model": prior.revision, "panel": fit_panel.revision}
            if kind == "fitted"
            else {},
            produced_by="fit" if kind == "fitted" else "edit_model",
            json_files={"model.json": model.model_dump(mode="json")},
        )
        if kind == "fitted":
            from nof1_causal_lab.artifacts.identity import GitRef
            from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
            from nof1_causal_lab.study.records import ModelFitResult
            from tests.action_fixtures import applied_record

            fit = _report(model)
            fit = fit.revised(core=fit.core.revised(time_origin=fit_origin))
            if reliable == "unconverged":
                diagnostics = fit.core.inference_diagnostics
                assert diagnostics is not None
                poor = diagnostics.revised(
                    **{
                        "per_parameter": tuple(
                            row.revised(**{"r_hat": 1.2}) for row in diagnostics.per_parameter
                        )
                    }
                )
                fit = fit.revised(
                    core=fit.core.revised(
                        inference_diagnostics=poor, convergence=parameter_convergence(poor)
                    )
                )
            monkeypatch.setattr("nof1_causal_lab.study.lineage.fitted_law_report", lambda *_args: fit.core)
            StudyRepository("ORIGIN").append(
                applied_record(
                    Applied(
                        result=ModelFitResult(
                            model=GitRef(
                                workspace_id="ORIGIN", revision=prior.revision, path="model.json"
                            ),
                            panel=GitRef(
                                workspace_id="ORIGIN",
                                revision=fit_panel.revision,
                                path="panel.parquet",
                            ),
                            evidence=inference_log(model).record.attempt.outcome.result.evidence.revised(time_origin=fit_origin),
                        ),
                        effects=ActionEffects(produced=(prior, fit_panel, record)),
                    ),
                    seq=1,
                )
            )
    current_origin = datetime(2026, 1, 1, tzinfo=UTC)
    metadata = panel_metadata()
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={
            "metadata.json": metadata.revised(time_origin=current_origin).model_dump(mode="json")
        },
    )
    question = write_question(store)
    state = StudyState().with_artifacts(
        [question, record, panel] if current_panel else [question, record]
    )
    response = SimulationReport.model_validate(_response())
    expected_origin = (
        fit_origin
        if kind == "fitted"
        else current_origin
        if current_panel
        else response.evidence.design.start_instant
    )
    expected_panel = (
        fit_panel.revision if kind == "fitted" else panel.revision if current_panel else None
    )

    def generate(_model, _design, *, revision, time_origin, origin_panel_revision, **_kwargs):
        assert time_origin == expected_origin
        assert origin_panel_revision == expected_panel
        generated = SimulationEvidence.model_validate(_response(time_origin)["evidence"])
        return generated.revised(model=revision, origin_panel_revision=origin_panel_revision)

    action = import_module("nof1_causal_lab.actions.simulate")
    monkeypatch.setattr(action, "simulate", generate)
    monkeypatch.setattr(action, "read_simulation_report", lambda _store, evidence, _question: SimulationReport(
        evidence=evidence, law=AuthoredLawProvenance(), fit_reliability=reliable,
        causal=SimulationReport.model_validate(_response(evidence.time_origin)).causal))
    applied = run_async(
        run_action(
            "ORIGIN",
            SimulateRequest(
                model_revision=record.revision,
                panel_revision=panel.revision if current_panel else None,
                start=response.evidence.design.start,
                horizon=response.evidence.design.horizon,
                interventions=response.evidence.design.interventions,
            ),
            state,
        )
    )
    saved = applied.result.evidence
    assert saved.time_origin == expected_origin
    assert saved.origin_panel_revision == expected_panel
    assert "report" not in applied.result.model_dump()
