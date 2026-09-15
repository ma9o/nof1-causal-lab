"""Translate scientific requests into pinned, durable work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
)
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.machine.moves import ExecOptions, Move, RunOperation, WriteArtifact

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@dataclass(frozen=True)
class ActionCommand:
    """The execution command and payload for one scientific request."""

    move: Move
    options: ExecOptions
    payload: JsonObject | None = None


def action_command(request: ScientificActionRequest) -> ActionCommand:
    if isinstance(request, EditModelRequest):
        return ActionCommand(
            WriteArtifact(artifact_id="model", expected_model_version=request.expected_version),
            ExecOptions(),
            request.model.model_dump(mode="json"),
        )
    if isinstance(request, PrepareDataRequest):
        if request.source == "files":
            return ActionCommand(RunOperation(operation_id="raw_data"), ExecOptions())
        assert request.model_version is not None
        assert request.raw_data_version is not None
        return ActionCommand(
            RunOperation(
                operation_id="measurements",
                input_versions={
                    "model": request.model_version,
                    "raw_data": request.raw_data_version,
                },
            ),
            ExecOptions(max_windows=request.max_windows),
        )
    if isinstance(request, FitRequest):
        return ActionCommand(
            RunOperation(
                operation_id="posterior",
                input_versions={"model": request.model_version, "panel": request.panel_version},
            ),
            ExecOptions(fit_settings=request.settings),
        )
    assert isinstance(request, SimulateRequest)
    pins: dict[ArtifactId, int] = {"model": request.model_version}
    if request.comparison_panel_version is not None:
        pins["panel"] = request.comparison_panel_version
    return ActionCommand(
        RunOperation(operation_id="simulate", input_versions=pins),
        ExecOptions(
            simulation=request.design, comparison_panel_version=request.comparison_panel_version
        ),
    )
