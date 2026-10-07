"""One forward-generation contract with dated interventions and durable histories."""

from datetime import UTC, date, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts import SimulateRequest
from nof1_causal_lab.actions.io import SimulateInput, SimulateOutput
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.artifacts.simulation import (
    SimulationReport,
    SimulationSpec,
)
from tests.action_fixtures import empty_simulation_summary
from tests.inference_fixtures import compile_model_fixture, particle_posterior
from tests.model_fixtures import x_y_model

pytestmark = pytest.mark.contract


def test_simulation_request_is_a_dated_window_with_optional_interventions():
    request = SimulateRequest[GitOid](
        input=SimulateInput[GitOid](
            simulation=SimulationSpec(start=date(2026, 1, 1), horizon="12d"),
            model_ref="a" * 40,
        )
    )
    assert request.input.simulation.interventions == ()
    assert set(request.model_dump()) == {"action", "input", "reasoning"}
    assert set(request.input.model_dump()) == {"model_ref", "simulation"}
    with pytest.raises(ValidationError, match="panel_ref"):
        request.input.revised(panel_ref="b" * 40)
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
        "differences": "differences",
        "frame": [0, 2],
        "summary": {"mean": 1, "median": 1, "lower_95": 0, "upper_95": 2, "prob_positive": 0.9},
        "reference_mean": 3,
        "manifest_effects": {},
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
        "arms": {
            "kind": "paired",
            "action": {"latent_paths": "paths", "observations": "observations"},
            "reference": {
                "latent_paths": "reference-paths",
                "observations": "reference-observations",
            },
        },
    }
    return {
        "evidence": evidence,
        "summary": empty_simulation_summary().model_dump(mode="json"),
        "law": AuthoredLawProvenance().model_dump(mode="json"),
        "fit_reliability": "converged",
        "causal": {"kind": "available", "value": result},
    }


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


@pytest.mark.parametrize("missing", ["latent_paths", "observations"])
def test_causal_reports_require_their_effects_and_paired_draws(missing):
    payload = _response()
    del payload["evidence"]["arms"]["reference"][missing]
    with pytest.raises(ValidationError, match="Field required"):
        TypeAdapter(SimulationReport).validate_python(payload)


def test_success_has_one_required_report_and_no_duplicate_paths():
    output = SimulateOutput(
        data=({},), report=SimulationReport.model_validate(_response()), arrays={}
    )
    payload = output.model_dump(mode="json")
    assert set(payload) == {"data", "report", "arrays"}
    for field in payload:
        with pytest.raises(ValidationError):
            SimulateOutput.model_validate({**payload, field: None})
        with pytest.raises(ValidationError):
            SimulateOutput.model_validate(
                {key: value for key, value in payload.items() if key != field}
            )
    with pytest.raises(ValidationError):
        output.revised(paths={})
    with pytest.raises(ValidationError):
        output.revised(data=())


def test_intervention_scope_requires_the_matching_arm_variant():
    payload = _response()
    payload["evidence"]["design"]["interventions"] = []
    with pytest.raises(ValidationError, match="paired reference"):
        SimulationReport.model_validate(payload)
    payload["evidence"]["design"]["interventions"] = [{"target": "construct:x", "value": 1}]
    payload["evidence"]["arms"] = {
        "kind": "single",
        "action": payload["evidence"]["arms"]["action"],
    }
    with pytest.raises(ValidationError, match="paired reference"):
        SimulationReport.model_validate(payload)


@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("current_panel", [False, True])
def test_runner_uses_model_time_binding_without_reading_panels(
    tmp_path, monkeypatch, conditioned, current_panel
):
    from importlib import import_module

    import jax.numpy as jnp

    from nof1_causal_lab.actions.errors import ActionExecutionError
    from nof1_causal_lab.actions.runners import run_action
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data
    from tests.helpers import run_async, write_question
    from tests.inference_fixtures import parameter_draws
    from tests.integration.runner_fixtures import panel_frame, panel_metadata

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("ORIGIN")
    model = x_y_model()
    fit_origin = datetime(2024, 1, 2, tzinfo=UTC)
    if conditioned:
        model, _ = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
            time_origin=fit_origin,
        )
    # An imported conditioned model must be executable without its original fit history.
    record = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
        parquet_files={"panel.parquet": panel_frame(n_days=2)},
    )
    state = StudyState().with_artifacts(
        [
            write_question(
                store, QuestionSpec(text="Effect on sleep", outcome=model.constructs[1].id)
            ),
            record,
            *([panel] if current_panel else []),
        ]
    )
    design = SimulationSpec(start=date(2026, 1, 1), horizon="2d")
    expected_origin = fit_origin if conditioned else design.start_instant
    calls = []

    def generate(_model, *, start, end, assignments, time_origin):
        calls.append(time_origin)
        assert time_origin == expected_origin
        assert (start, end) == (design.start_day(expected_origin), design.end_day(expected_origin))
        assert assignments == ()
        return ObservationPreflightFailure.rejected("checked model-owned coordinates")

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("Simulation must not read a data panel")

    monkeypatch.setattr(
        import_module("nof1_causal_lab.actions.simulate"), "generate_simulation_batch", generate
    )
    monkeypatch.setattr("nof1_causal_lab.study.data.read_data_history", unexpected_read)
    with pytest.raises(ActionExecutionError, match="checked model-owned coordinates"):
        run_async(
            run_action(
                "ORIGIN",
                SimulateRequest[GitOid](
                    input=SimulateInput[GitOid](model_ref=record.revision, simulation=design)
                ),
                state,
            )
        )
    assert calls == [expected_origin]
