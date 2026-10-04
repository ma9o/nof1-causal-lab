"""What an executed action produced; the workflow publishes it with the attempt."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.study.state import ArtifactRecord, RetractedArtifact


class ActionEffects(Value):
    """What an executed action did to the store: the workflow installs this."""

    produced: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    retracted: tuple[RetractedArtifact, ...] = Field(default_factory=tuple)
