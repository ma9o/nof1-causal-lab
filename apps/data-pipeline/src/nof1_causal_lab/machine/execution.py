"""Private execution results, input selection and artifact freshness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.data_preparation import (  # noqa: TC001
    FilePreparationSpec,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS, ArtifactId, GitOid
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec  # noqa: TC001
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifacts import ArtifactRecord  # noqa: TC001

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition


class RetractedArtifact(BaseModel):
    """A current artifact removed by an action, with the finding that caused it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    reason_ref: str


class TransitionEffects(BaseModel):
    """What an executed action did to the store: the workflow installs this."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    diagnostics: JsonObject = Field(default_factory=dict)
    produced: list[ArtifactRecord] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    checks: ModelCheckReport | None = None


class FitOperation(BaseModel):
    """Condition the selected model on its selected panel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: Literal["posterior"] = "posterior"
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)


class SimulateOperation(BaseModel):
    """Generate the declared simulation from the selected model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: Literal["simulate"] = "simulate"
    design: SimulationSpec


class PrepareFilesOperation(BaseModel):
    """Prepare uploaded observations through the ingestion and extraction workflows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: Literal["measurements"] = "measurements"
    preparation: FilePreparationSpec


class PrepareSimulationOperation(BaseModel):
    """Materialize one recorded simulation replicate as observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: Literal["simulated_measurements"] = "simulated_measurements"
    source: SimulationReplicateRef


type LocalOperation = Annotated[
    FitOperation | SimulateOperation | PrepareSimulationOperation,
    Field(discriminator="operation_id"),
]
type ExecutionOperation = Annotated[
    FitOperation | SimulateOperation | PrepareSimulationOperation | PrepareFilesOperation,
    Field(discriminator="operation_id"),
]


def validate_model_base(state: EpisodeState, expected_revision: GitOid | None) -> str | None:
    current = state.get("model")
    revision = current.revision if current else None
    if expected_revision != revision:
        return f"Model revision conflict: expected {expected_revision}, current {revision}"
    return None


def input_pins(state: EpisodeState, spec: Transition) -> dict[ArtifactId, GitOid]:
    """The exact input versions a run of ``spec`` at ``state`` consumes."""
    pins: dict[ArtifactId, GitOid] = {}
    for artifact in spec.consumes:
        info = state.get(artifact)
        if info is None:
            raise ValueError(f"{spec.operation_id} input '{artifact}' does not exist")
        pins[artifact] = info.revision
    for artifact in spec.optional_consumes:
        info = state.get(artifact)
        if info is not None:
            pins[artifact] = info.revision
    return pins


def apply_transition(
    state: EpisodeState,
    produced: list[ArtifactRecord],
    retracted: list[RetractedArtifact] | None = None,
    checks: ModelCheckReport | None = None,
) -> EpisodeState:
    """Install produced versions and retractions into a new state."""
    next_state = state.with_artifacts(produced)
    if retracted:
        next_state = next_state.without([item.artifact_id for item in retracted])
    if checks is not None:
        next_state = next_state.model_copy(update={"checks": checks})
    return next_state


def run_retractions(
    state: EpisodeState,
    spec: Transition,
    produced: list[ArtifactRecord],
) -> list[RetractedArtifact]:
    """Optional co-outputs to retract after a successful run of ``spec``."""
    produced_ids = {info.artifact_id for info in produced}
    return [
        RetractedArtifact(
            artifact_id=artifact,
            reason_ref=f"{spec.operation_id}.produces_optional.{artifact}",
        )
        for artifact in spec.produces_optional
        if artifact not in produced_ids and state.has(artifact)
    ]


def is_stale(state: EpisodeState, artifact_id: ArtifactId) -> bool:
    """Whether an artifact's input chain references superseded versions.

    Derived artifacts are never stale. If their parents change, the action that
    changed the parents also recomputes or retracts the derivation.
    """
    if artifact_id == "model":
        return False
    return _staleness(state, artifact_id, frozenset())


def _staleness(
    state: EpisodeState, artifact_id: ArtifactId, visiting: frozenset[ArtifactId]
) -> bool:
    info = state.get(artifact_id)
    if info is None or artifact_id in visiting:
        return False
    marked = visiting | {artifact_id}
    for input_id in info.derived_from:
        current = state.get(input_id)
        if current is None or not state.matches_inputs(artifact_id, input_id):
            return True
        if input_id != "model" and _staleness(state, input_id, marked):
            return True
    return False


class ArtifactFreshness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: ArtifactId
    exists: bool
    stale: bool
    revision: GitOid | None = None
    retracted: bool = False
    produced_by: str | None = None


def freshness_report(state: EpisodeState) -> list[ArtifactFreshness]:
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
