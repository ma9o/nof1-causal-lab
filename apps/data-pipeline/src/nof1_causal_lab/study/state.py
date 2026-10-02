"""Study state: the artifact versions a Git commit selects, and their freshness.

Git trees select immutable scientific artifacts. Each artifact records its input
tree OIDs and scientific fingerprints, which determine freshness. StudyState is the
runtime projection of a selected commit tree.

These are pydantic models (frozen) rather than dataclasses because they
cross serialization boundaries verbatim: Temporal update/activity payloads
(via the pydantic data converter) and the tool-server facade's JSON API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS, ArtifactId, GitOid
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport


class ArtifactRecord(Value):
    """Artifact revision metadata records how a stored artifact was produced and which inputs it
    used.

    ``derived_from`` pins the exact input versions the payload was computed
    from. For initial model revisions it is empty. ``created_at`` is
    stamped by the activity that produced the revision — never inside workflow
    code, where wall-clock time is non-deterministic.
    """

    artifact_id: ArtifactId
    revision: GitOid
    derived_from: Mapping[ArtifactId, GitOid] = Field(default_factory=dict)
    model_inputs: Mapping[str, str] = Field(default_factory=dict)
    consumed_model_inputs: Mapping[str, str] = Field(default_factory=dict)
    produced_by: str | None = None
    created_at: str = ""


class StudyState(Value):
    """Study state projects the artifact trees selected by one Git commit.

    ``current`` maps artifact id → the revision info that is *current* for the
    study. Absent key = the artifact does not exist (either never produced,
    or produced-when-nonempty semantics withheld it).
    """

    current: Mapping[ArtifactId, ArtifactRecord] = Field(default_factory=dict)
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

    def with_artifacts(self, infos: Sequence[ArtifactRecord]) -> StudyState:
        """Return a new state with ``infos`` installed as current versions."""
        merged = dict(self.current)
        for info in infos:
            merged[info.artifact_id] = info
        return self.model_copy(
            update={"current": {aid: merged[aid] for aid in ARTIFACT_IDS if aid in merged}}
        )

    def with_checks(self, checks: ModelCheckReport) -> StudyState:
        return self.model_copy(update={"checks": checks})

    def without(self, artifact_ids: list[ArtifactId]) -> StudyState:
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


class RetractedArtifact(Value):
    """A current artifact removed by an action, with the finding that caused it."""

    artifact_id: ArtifactId
    reason_ref: str


def validate_model_base(state: StudyState, expected_revision: GitOid | None) -> str | None:
    current = state.get("model")
    revision = current.revision if current else None
    if expected_revision != revision:
        return f"Model revision conflict: expected {expected_revision}, current {revision}"
    return None


def apply_effects(
    state: StudyState,
    produced: Sequence[ArtifactRecord],
    retracted: Sequence[RetractedArtifact] | None = None,
    checks: ModelCheckReport | None = None,
) -> StudyState:
    """Install produced versions and retractions into a new state."""
    next_state = state.with_artifacts(produced)
    if retracted:
        next_state = next_state.without([item.artifact_id for item in retracted])
    if checks is not None:
        next_state = next_state.with_checks(checks)
    return next_state


def is_stale(state: StudyState, artifact_id: ArtifactId) -> bool:
    """Whether an artifact's input chain references superseded versions.

    Derived artifacts are never stale. If their parents change, the action that
    changed the parents also recomputes or retracts the derivation.
    """
    if artifact_id == "model":
        return False
    return _staleness(state, artifact_id, frozenset())


def _staleness(state: StudyState, artifact_id: ArtifactId, visiting: frozenset[ArtifactId]) -> bool:
    info = state.get(artifact_id)
    if info is None or artifact_id in visiting:
        return False
    marked = visiting.union((artifact_id,))
    for input_id in info.derived_from:
        current = state.get(input_id)
        if current is None or not state.matches_inputs(artifact_id, input_id):
            return True
        if input_id != "model" and _staleness(state, input_id, marked):
            return True
    return False


class ArtifactFreshness(Value):
    artifact_id: ArtifactId
    exists: bool
    stale: bool
    revision: GitOid | None = None
    retracted: bool = False
    produced_by: str | None = None


def freshness_report(state: StudyState) -> list[ArtifactFreshness]:
    """Per-artifact existence/staleness — the navigator's and UI's state view."""
    report: list[ArtifactFreshness] = []
    for artifact_id in ARTIFACT_IDS:
        info = state.get(artifact_id)
        report.append(
            ArtifactFreshness(
                artifact_id=artifact_id,
                exists=info is not None,
                stale=is_stale(state, artifact_id),
                revision=info.revision if info else None,
                produced_by=info.produced_by if info else None,
            )
        )
    return report
