"""What an executed action produced; the workflow publishes it with the attempt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.study.state import ArtifactRecord, RetractedArtifact


class ActionEffects(BaseModel):
    """What an executed action did to the store: the workflow installs this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diagnostics: JsonObject = Field(default_factory=dict)
    produced: list[ArtifactRecord] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    checks: ModelCheckReport | None = None
