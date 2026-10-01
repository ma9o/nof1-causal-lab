"""Attempt records stored in study commits, their commit views and dependency links."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.identity import ActionId, GitOid
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.study.state import ArtifactRecord, RetractedArtifact, StudyState

if TYPE_CHECKING:
    from collections.abc import Sequence


class ActionMessage(BaseModel):
    """One label emitted by an action; scientific measurements belong in its body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: AwareDatetime
    level: Literal["debug", "info", "warn", "error"]
    label: str = Field(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")


type JournalStatus = Literal["applied", "rejected", "raised"]


class AttemptRecord(BaseModel):
    """One journaled action attempt — applied, rejected, or raised.

    Stored in its owning Git commit. Rejected and raised attempts retain logs
    without advancing the scientific branch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    attempt_id: UUID | None = None
    branch: str = "main"
    ts: str
    action: ActionId
    inputs: JsonObject = Field(default_factory=dict)
    status: JournalStatus
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: JsonObject = Field(default_factory=dict)
    checks: ModelCheckReport | None = None
    messages: tuple[ActionMessage, ...] = ()
    produced: list[ArtifactRecord] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    trace_ids: list[str]


class StudyRevision(AttemptRecord):
    """One Git commit's parent links and its action log."""

    commit_id: GitOid
    parent_ids: list[GitOid]


class RecordDependency(BaseModel):
    """A journal record used an output of an earlier record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    source_seq: int
    argument: str
    check: bool = Field(
        description="Only the action's checks read the output; its request did not name it.",
    )


def _argument_revisions(inputs: JsonObject) -> list[tuple[str, str]]:
    """Revisions a request named: fields ending in `revision`, and refs that carry one.

    The field names the argument (`expected_revision` is the model an edit revises); a ref names
    its field, and prepare_data's `input` ref is a simulation replicate.
    """
    pins: list[tuple[str, str]] = []
    for key, value in inputs.items():
        if key.endswith("revision") and isinstance(value, str):
            pins.append(
                ("model" if key == "expected_revision" else key.removesuffix("_revision"), value)
            )
        for ref in value if isinstance(value, list) else [value]:
            if isinstance(ref, dict) and isinstance(revision := ref.get("revision"), str):
                pins.append(("simulation" if key == "input" else key, revision))
    return pins


def record_dependencies(records: Sequence[StudyRevision]) -> list[RecordDependency]:
    """Link each record to the earlier records whose outputs it took, in journal order.

    Arguments are the revisions its request named, plus the previous revision of any artifact it
    revised. Other pins on its new outputs are reads by its checks. A simulation is referenced by
    its commit; every other output by its artifact revision, which keeps its first producer.
    """
    producers: dict[str, int] = {}
    for record in records:
        if record.status != "applied":
            continue
        for artifact in record.produced:
            producers.setdefault(artifact.revision, record.seq)
        if record.action == "simulate":
            producers[record.commit_id] = record.seq
    dependencies: list[RecordDependency] = []
    for record in records:
        arguments: dict[int, str] = {}
        for name, revision in _argument_revisions(record.inputs):
            source = producers.get(revision)
            if source is not None and source < record.seq:
                arguments.setdefault(source, name)
        pins = [
            (artifact.artifact_id, artifact_id, source)
            for artifact in record.produced
            if producers.get(artifact.revision) == record.seq
            for artifact_id, revision in artifact.derived_from.items()
            if (source := producers.get(revision)) is not None and source < record.seq
        ]
        for produced, artifact_id, source in pins:
            if artifact_id == produced:
                arguments.setdefault(source, artifact_id)
        checks: dict[int, str] = {}
        for _, artifact_id, source in pins:
            if source not in arguments:
                checks.setdefault(source, artifact_id)
        dependencies += [
            RecordDependency(seq=record.seq, source_seq=source, argument=argument, check=False)
            for source, argument in arguments.items()
        ]
        dependencies += [
            RecordDependency(seq=record.seq, source_seq=source, argument=argument, check=True)
            for source, argument in checks.items()
        ]
    return dependencies


class BranchBase(BaseModel):
    """Immutable execution base, captured before an action starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit_id: GitOid
    state: StudyState
