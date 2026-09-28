"""Private job planning for the four scientific actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid, OperationId
from nof1_causal_lab.machine.execution import ExecutionOptions


@dataclass(frozen=True)
class ActionExecution:
    """Implementation job and exact inputs selected by a scientific request."""

    operation_id: OperationId
    input_revisions: dict[ArtifactId, GitOid]
    options: ExecutionOptions


def plan_execution(request: PrepareDataRequest | FitRequest | SimulateRequest) -> ActionExecution:
    if isinstance(request, PrepareDataRequest):
        from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef

        if isinstance(request.source, SimulationReplicateRef):
            return ActionExecution(
                "simulated_measurements", {}, ExecutionOptions(simulation_source=request.source)
            )
        return ActionExecution(
            "measurements",
            {},
            ExecutionOptions(
                file_source=request.source,
                preparation=request.preparation,
                max_windows=request.max_windows,
            ),
        )
    if isinstance(request, FitRequest):
        return ActionExecution(
            "posterior",
            {"model": request.model_revision, "panel": request.panel_revision},
            ExecutionOptions(fit_settings=request.settings),
        )
    from nof1_causal_lab.artifacts.simulation import SimulationSpec

    return ActionExecution(
        "simulate",
        {"model": request.model_revision},
        ExecutionOptions(
            simulation=SimulationSpec.model_validate(
                request.model_dump(exclude={"action", "model_revision"})
            ),
        ),
    )
