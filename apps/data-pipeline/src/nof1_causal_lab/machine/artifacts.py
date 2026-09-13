"""Artifact taxonomy: the nodes of the episode state machine.

The machine's state is a versioned artifact store. Each artifact version is
immutable and stamped with provenance plus the exact input versions it was
derived from — that stamp is what makes staleness and freshness *derived*
properties rather than stored flags. Applied transition effects select which
versions are current and let a timeline scrubber reconstruct past state.

These are pydantic models (frozen) rather than dataclasses because they
cross serialization boundaries verbatim: Temporal update/activity payloads
(via the pydantic data converter) and the tool-server facade's JSON API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS, ArtifactId

type Provenance = Literal["computed", "human", "llm"]


class ArtifactVersionInfo(BaseModel):
    """Artifact version metadata records how a stored artifact was produced and which inputs it
    used.

    ``derived_from`` pins the exact input versions the payload was computed
    from. For root artifacts (user writes) it is empty. ``created_at`` is
    stamped by the activity that produced the version — never inside workflow
    code, where wall-clock time is non-deterministic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    version: int
    provenance: Provenance
    derived_from: dict[ArtifactId, int] = Field(default_factory=dict)
    produced_by: str | None = None
    created_at: str = ""


class EpisodeState(BaseModel):
    """Episode state identifies the artifact versions currently selected by the transition
    journal.

    ``current`` maps artifact id → the version info that is *current* for the
    episode. Absent key = the artifact does not exist (either never produced,
    or produced-when-nonempty semantics withheld it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    current: dict[ArtifactId, ArtifactVersionInfo] = Field(default_factory=dict)

    def get(self, artifact_id: ArtifactId) -> ArtifactVersionInfo | None:
        return self.current.get(artifact_id)

    def has(self, artifact_id: ArtifactId) -> bool:
        return artifact_id in self.current

    def matches_inputs(self, output: ArtifactId, *inputs: ArtifactId) -> bool:
        """Whether an artifact pins exactly the selected versions of these inputs."""
        info = self.get(output)
        return info is not None and all(
            (selected := self.get(artifact_id)) is not None
            and info.derived_from.get(artifact_id) == selected.version
            for artifact_id in inputs
        )

    def with_versions(self, infos: list[ArtifactVersionInfo]) -> EpisodeState:
        """Return a new state with ``infos`` installed as current versions."""
        merged = dict(self.current)
        for info in infos:
            merged[info.artifact_id] = info
        return EpisodeState(current={aid: merged[aid] for aid in ARTIFACT_IDS if aid in merged})

    def without(self, artifact_ids: list[ArtifactId]) -> EpisodeState:
        """Return a new state with the given artifacts removed from ``current``.

        Used for produced-when-nonempty semantics: re-running a stage whose
        previous version produced an optional artifact, but whose new run did
        not, must retract the old version — otherwise downstream stages would
        silently consume a payload derived from superseded inputs.
        """
        removed = set(artifact_ids)
        return EpisodeState(
            current={aid: info for aid, info in self.current.items() if aid not in removed}
        )
