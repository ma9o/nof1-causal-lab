"""Runner requests keep their concrete type through Temporal."""

from datetime import date

import pytest

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest
from nof1_causal_lab.actions.temporal.client import pydantic_data_converter
from nof1_causal_lab.actions.temporal.messages import ActionInput
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.study.state import StudyState
from tests.git_fixtures import git_oid
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "action",
    [
        FitRequest(model_revision=git_oid(1), panel_revision=git_oid(2)),
        SimulateRequest(
            model_revision=git_oid(1),
            start=date(2026, 1, 1),
            horizon="1w",
            interventions=({"target": "construct:x", "after": "2d", "value": 1},),
        ),
        PrepareDataRequest(input=SimulationReplicateRef(revision=git_oid(3), replicate=0)),
    ],
)
def test_action_inputs_keep_their_request_through_temporal(action):
    original = ActionInput(workspace_id="test", request=action, state=StudyState())

    async def round_trip():
        encoded = await pydantic_data_converter.encode([original])
        return await pydantic_data_converter.decode(encoded, [ActionInput])

    assert run_async(round_trip()) == [original]
