"""Running calls and complete, repeatable action results."""

from collections.abc import Mapping
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from nof1_causal_lab.actions.contracts import ScientificActionRequest
from nof1_causal_lab.actions.progress import ProgressEvent
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ActionId, GitOid, IndicatorId
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.posterior_diagnostics import PPCOverlay
from nof1_causal_lab.json_types import JsonObject, JsonValue
from nof1_causal_lab.study.records import ActionAttempt, ActionMessage
from nof1_causal_lab.study.snapshot_models import ModelSnapshot, Sourced
from nof1_causal_lab.study.view_models import DataDiffReport, DataDiffRequest
from nof1_causal_lab.study.visual_models import ObservationHistory, ParameterDraws, SimulationPaths
from nof1_causal_lab.utils.llm import LLMTrace


class RunningPoll(Value):
    kind: Literal["running"] = "running"
    attempt_id: UUID
    request: ScientificActionRequest | DataDiffRequest  # noqa: FIELD003 -- External HTTP clients repeat these canonical arguments, including call-time file hashes, to poll this exact call.
    messages: tuple[ActionMessage, ...] = ()
    events: tuple[ProgressEvent, ...] = ()  # noqa: FIELD003 -- The public action response must include extraction progress for external HTTP clients.


class CompletedPoll(Value):
    kind: Literal["completed"] = "completed"
    # Failure before publication is still a completed typed outcome.
    commit_id: GitOid | None
    attempt: ActionAttempt
    messages: tuple[ActionMessage, ...] = ()
    snapshot: ModelSnapshot | None = None
    inference_report: Sourced[InferenceReport] | None = None
    data_comparison: DataDiffReport | None = None
    checks: ModelCheckReport | None = None
    observation_histories: Mapping[IndicatorId, ObservationHistory] = Field(default_factory=dict)
    predictive_overlays: Mapping[IndicatorId, PPCOverlay] = Field(default_factory=dict)
    simulation_paths: SimulationPaths | None = None
    parameter_draws: ParameterDraws | None = None
    traces: Mapping[str, LLMTrace] = Field(default_factory=dict)
    artifacts: Mapping[str, JsonObject] = Field(default_factory=dict)  # noqa: FIELD003 -- External HTTP clients need the retained artifact payloads now that artifact read routes are removed.
    arrays: Mapping[str, JsonValue] = Field(default_factory=dict)  # noqa: FIELD003 -- The action API contract returns complete numerical arrays to external HTTP clients without another read route.


type ActionPoll = Annotated[RunningPoll | CompletedPoll, Field(discriminator="kind")]


class RunningAction(Value):
    attempt_id: UUID
    action: ActionId
    request: ScientificActionRequest | DataDiffRequest  # noqa: FIELD003 -- External HTTP clients obtain the running call's canonical arguments from the timeline to repeat it.
    messages: tuple[ActionMessage, ...]
    events: tuple[ProgressEvent, ...] = ()
