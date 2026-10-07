"""Cold worker payload round trips must preserve native distribution constructors."""

import json
import subprocess
import sys
from textwrap import dedent
from uuid import uuid4

import pytest

from nof1_causal_lab.artifacts.identity import scientific_id
from tests.helpers import graph_constructs, make_model

pytestmark = pytest.mark.contract


def test_prior_request_round_trips_through_a_cold_workflow_sandbox():
    model = make_model(["symptoms"]).model_dump(mode="json")
    parameter_id = scientific_id("parameter", "baseline")
    graph_constructs(model)[0]["coefficients"] = [
        {"kind": "coefficient", "role": "initial_mean", "value": parameter_id}
    ]
    model["parameters"] = [
        {
            "id": parameter_id,
            "name": "baseline",
            "description": "Initial symptom burden",
            "distribution": "distribution:baseline",
        }
    ]
    model["distributions"] = {
        "distribution:baseline": {
            "distribution": "Beta",
            # The facade emits typed arrays, whose dtype encoding lazily imports
            # NumPy even when decoding plain Python scalar arguments succeeds.
            "params": {
                "concentration1": {"array": 2.0, "dtype": "float32"},
                "concentration0": {"array": 2.0, "dtype": "float32"},
            },
        }
    }
    envelope = {
        "attempt_id": str(uuid4()),
        "request": {
            "action": "edit_model",
            "input": {"parent_ref": "3" * 40, "model": model},
        },
    }
    # Constructing this law before entering the sandbox warms JAX's lazy imports
    # and masks the rejection, so decode raw JSON in a fresh worker process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import asyncio
                import importlib
                import sys

                from temporalio.api.common.v1 import Payload
                from temporalio.contrib.pydantic import pydantic_data_converter
                from temporalio.worker.workflow_sandbox._importer import Importer
                from temporalio.worker.workflow_sandbox._restrictions import RestrictionContext

                from nof1_causal_lab.actions.contracts import FitRequest, SimulateRequest
                from nof1_causal_lab.actions.io import FitInput, SimulateInput
                from nof1_causal_lab.artifacts.identity import GitOid
                from nof1_causal_lab.artifacts.simulation import SimulationSpec
                from nof1_causal_lab.study.records import AttemptRecord, Applied, EditAttempt
                from nof1_causal_lab.actions.temporal.messages import (
                    ActionInput, ActionRequest, EditModelActivityInput, EvaluateChecksInput, AttemptPublication,
                )
                from nof1_causal_lab.study.state import ArtifactRecord, StudyState
                from nof1_causal_lab.actions.effects import ActionEffects
                from nof1_causal_lab.actions.temporal.worker import study_workflow_runner

                payload = Payload(metadata={"encoding": b"json/plain"}, data=sys.stdin.buffer.read())
                converter = pydantic_data_converter.payload_converter
                importer = Importer(study_workflow_runner().restrictions, RestrictionContext())
                with importer.applied():
                    importer.restriction_context.is_runtime = True
                    restored, = converter.from_payloads([payload], [ActionRequest])
                    model_record = ArtifactRecord(artifact_id="model", revision="1" * 40)
                    panel_record = ArtifactRecord(artifact_id="panel", revision="2" * 40)
                    state = StudyState(current={"model": model_record, "panel": panel_record})
                    inputs = [
                        EditModelActivityInput(workspace_id="test", request=restored.request),
                        # These activities carry revision references, not a ModelSpec.
                        ActionInput(
                            workspace_id="test", state=state,
                            request=FitRequest[GitOid](input=FitInput[GitOid](
                                model_ref=model_record.revision,
                                data_ref=panel_record.revision, replicate_index=0,
                            )),
                        ),
                        ActionInput(
                            workspace_id="test", state=state,
                            request=SimulateRequest[GitOid](input=SimulateInput[GitOid](
                                model_ref=model_record.revision,
                                simulation=SimulationSpec(start="2026-01-01", horizon="2d"),
                            )),
                        ),
                        EvaluateChecksInput[None](
                            workspace_id="test", state=state, request=restored.request,
                            applied=Applied(result=None, effects=ActionEffects(produced=[model_record])),
                        ),
                    ]
                    activity_payloads = converter.to_payloads(inputs)
                    journal = AttemptPublication(
                        workspace_id="test", parent_id=None,
                        record=AttemptRecord(
                            seq=1, ts="2026-10-01T00:00:00Z", attempt_id=restored.attempt_id,
                            attempt=EditAttempt(action="edit_model", request=restored.request,
                                outcome=Applied(result=GitOid("4" * 40), effects=ActionEffects(produced=[model_record]))),
                        ),
                    )
                    inputs.append(journal)
                    activity_payloads.extend(converter.to_payloads([journal]))
                    activity_inputs = converter.from_payloads(
                        activity_payloads, [type(value) for value in inputs]
                    )
                    assert converter.to_payloads(activity_inputs) == activity_payloads
                    update_payloads = converter.to_payloads([restored])
                    assert converter.from_payloads(update_payloads, [ActionRequest]) == [restored]
                # Sandbox configuration must not alter the existing wire representation.
                assert converter.to_payloads(inputs) == activity_payloads
                assert converter.to_payloads([restored]) == update_payloads
                law = restored.request.input.model.distributions["distribution:baseline"]
                assert type(law).__name__ == "Beta"
                assert law.batch_shape == law.event_shape == ()

                # A later activity imports the fitter in an ordinary thread,
                # without pytest's jaxtyping registry or the workflow importer.
                async def import_fitter():
                    module = await asyncio.to_thread(
                        importlib.import_module,
                        "nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs",
                    )
                    assert callable(module.fit_marginal_particle_gibbs)

                asyncio.run(import_fitter())
                print("round trip verified")
                """),
        ],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "round trip verified"
