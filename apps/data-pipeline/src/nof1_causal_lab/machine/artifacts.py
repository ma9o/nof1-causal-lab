"""Artifact taxonomy: the nodes of the episode state machine.

Git trees select immutable scientific artifacts. Each artifact records its
input tree OIDs and scientific fingerprints, which determine freshness. EpisodeState is the runtime projection of a selected commit tree.

These are pydantic models (frozen) rather than dataclasses because they
cross serialization boundaries verbatim: Temporal update/activity payloads
(via the pydantic data converter) and the tool-server facade's JSON API.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS, ArtifactId, GitOid
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport  # noqa: TC001


class ArtifactRecord(BaseModel):
    """Artifact revision metadata records how a stored artifact was produced and which inputs it
    used.

    ``derived_from`` pins the exact input versions the payload was computed
    from. For initial model revisions it is empty. ``created_at`` is
    stamped by the activity that produced the revision — never inside workflow
    code, where wall-clock time is non-deterministic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    revision: GitOid
    derived_from: dict[ArtifactId, GitOid] = Field(default_factory=dict)
    model_inputs: dict[str, str] = Field(default_factory=dict)
    consumed_model_inputs: dict[str, str] = Field(default_factory=dict)
    produced_by: str | None = None
    created_at: str = ""


class EpisodeState(BaseModel):
    """Episode state projects the artifact trees selected by one Git commit.

    ``current`` maps artifact id → the revision info that is *current* for the
    episode. Absent key = the artifact does not exist (either never produced,
    or produced-when-nonempty semantics withheld it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    current: dict[ArtifactId, ArtifactRecord] = Field(default_factory=dict)
    checks: ModelCheckReport | None = None

    def get(self, artifact_id: ArtifactId) -> ArtifactRecord | None:
        return self.current.get(artifact_id)

    def has(self, artifact_id: ArtifactId) -> bool:
        return artifact_id in self.current

    def matches_inputs(self, output: ArtifactId, *inputs: ArtifactId) -> bool:
        """Whether an artifact pins exactly the selected versions of these inputs."""
        info = self.get(output)
        return info is not None and all(
            (selected := self.get(artifact_id)) is not None
            and (
                info.derived_from.get(artifact_id) == selected.revision
                or (
                    artifact_id == "model"
                    and bool(info.consumed_model_inputs)
                    and all(
                        selected.model_inputs.get(key) == value
                        for key, value in info.consumed_model_inputs.items()
                    )
                )
            )
            for artifact_id in inputs
        )

    def with_artifacts(self, infos: list[ArtifactRecord]) -> EpisodeState:
        """Return a new state with ``infos`` installed as current versions."""
        merged = dict(self.current)
        for info in infos:
            merged[info.artifact_id] = info
        return self.model_copy(
            update={"current": {aid: merged[aid] for aid in ARTIFACT_IDS if aid in merged}}
        )

    def without(self, artifact_ids: list[ArtifactId]) -> EpisodeState:
        """Return a new state with the given artifacts removed from ``current``.

        Used for produced-when-nonempty semantics: re-running a stage whose
        previous revision produced an optional artifact, but whose new run did
        not, must retract the old revision — otherwise downstream stages would
        silently consume a payload derived from superseded inputs.
        """
        removed = set(artifact_ids)
        return self.model_copy(
            update={
                "current": {aid: info for aid, info in self.current.items() if aid not in removed}
            }
        )
