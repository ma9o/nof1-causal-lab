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
        "request": {"action": "edit_model", "expected_revision": None, "model": model},
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

                from nof1_causal_lab.artifacts.simulation import SimulationSpec
                from nof1_causal_lab.machine.artifacts import ArtifactRecord, EpisodeState
                from nof1_causal_lab.machine.execution import (
                    FitOperation, SimulateOperation, TransitionEffects,
                )
                from nof1_causal_lab.machine.temporal.messages import (
                    ActionRequest, EditModelInput, EvaluateChecksInput, JournalInput, OperationInput,
                )
                from nof1_causal_lab.machine.temporal.worker import episode_workflow_runner

                payload = Payload(metadata={"encoding": b"json/plain"}, data=sys.stdin.buffer.read())
                converter = pydantic_data_converter.payload_converter
                importer = Importer(episode_workflow_runner().restrictions, RestrictionContext())
                with importer.applied():
                    importer.restriction_context.is_runtime = True
                    restored, = converter.from_payloads([payload], [ActionRequest])
                    model_record = ArtifactRecord(artifact_id="model", revision="1" * 40)
                    panel_record = ArtifactRecord(artifact_id="panel", revision="2" * 40)
                    state = EpisodeState(current={"model": model_record, "panel": panel_record})
                    inputs = [
                        EditModelInput(workspace_id="test", request=restored.request, state=state),
                        # These activities carry revision references, not a ModelSpec.
                        OperationInput(
                            workspace_id="test", operation=FitOperation(), state=state,
                            input_revisions={"model": model_record.revision, "panel": panel_record.revision},
                        ),
                        OperationInput(
                            workspace_id="test",
                            operation=SimulateOperation(design=SimulationSpec(start=0, end=2)),
                            state=state, input_revisions={"model": model_record.revision},
                        ),
                        EvaluateChecksInput(
                            workspace_id="test", action="edit_model", state=state,
                            effects=TransitionEffects(produced=[model_record]),
                        ),
                    ]
                    activity_payloads = converter.to_payloads(inputs)
                    journal = JournalInput(
                        workspace_id="test", seq=1, action="edit_model", status="applied",
                        inputs=restored.request.model_dump(mode="json", exclude={"action"}),
                        produced=[model_record], resume=None, attempt_id=restored.attempt_id,
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
                law = restored.request.model.distributions["distribution:baseline"]
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
