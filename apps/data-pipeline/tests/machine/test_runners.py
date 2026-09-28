"""Local execution accepts only complete operations belonging to its runners."""

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import (
    FitOperation,
    LocalOperation,
    PrepareSimulationOperation,
    SimulateOperation,
)
from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
from nof1_causal_lab.machine.temporal.messages import OperationInput
from tests.git_fixtures import git_oid
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "payload",
    [
        {"operation_id": "raw_data"},
        {"operation_id": "measurements"},
        {"operation_id": "simulate"},
        {"operation_id": "simulated_measurements"},
        {"operation_id": "posterior", "design": {"end": 1}},
        {"operation_id": "simulate", "design": {"end": 1}, "settings": {}},
    ],
)
def test_local_operation_boundary_rejects_incomplete_or_mismatched_jobs(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(LocalOperation).validate_python(payload)


@pytest.mark.parametrize(
    "operation",
    [
        FitOperation(),
        SimulateOperation(design=SimulationSpec(start=0, end=1)),
        PrepareSimulationOperation(source=SimulationReplicateRef(revision=git_oid(1), replicate=0)),
    ],
)
def test_local_operations_keep_their_payload_through_temporal(operation):
    original = OperationInput(workspace_id="test", operation=operation, state=EpisodeState())

    async def round_trip():
        encoded = await pydantic_data_converter.encode([original])
        return await pydantic_data_converter.decode(encoded, [OperationInput])

    assert run_async(round_trip()) == [original]
