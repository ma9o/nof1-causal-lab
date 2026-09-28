"""Private job planning for the four scientific actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.machine.execution import (
    ExecutionOperation,
    FitOperation,
    PrepareFilesOperation,
    PrepareSimulationOperation,
    SimulateOperation,
)


@dataclass(frozen=True)
class ActionExecution:
    """Implementation job and exact inputs selected by a scientific request."""

    operation: ExecutionOperation
    input_revisions: dict[ArtifactId, GitOid]


def plan_execution(request: PrepareDataRequest | FitRequest | SimulateRequest) -> ActionExecution:
    if isinstance(request, PrepareDataRequest):
        from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef

        if isinstance(request.input, SimulationReplicateRef):
            return ActionExecution(
                operation=PrepareSimulationOperation(source=request.input), input_revisions={}
            )
        return ActionExecution(
            operation=PrepareFilesOperation(preparation=request.input), input_revisions={}
        )
    if isinstance(request, FitRequest):
        return ActionExecution(
            operation=FitOperation(settings=request.settings),
            input_revisions={"model": request.model_revision, "panel": request.panel_revision},
        )
    from nof1_causal_lab.artifacts.simulation import SimulationSpec

    return ActionExecution(
        operation=SimulateOperation(
            design=SimulationSpec(
                start=request.start, end=request.end, interventions=request.interventions
            ),
        ),
        input_revisions={"model": request.model_revision},
    )
