"""Write-move executors: schema-validated artifact writes.

Writes are the only way a caller bypasses a delegated transition. They are not
raw file writes: each executor validates the public payload shape, stamps
human/LLM provenance, pins any existing contextual inputs declared by the
artifact graph, and then runs the same derivation cascade as a computed
transition result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.derivations import complete_derivation_cascade
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, write_pins
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState, Provenance


def _validated(
    artifact_id: ArtifactId,
    model_cls: type[BaseModel],
    payload: object,
) -> UncheckedJsonObject:
    try:
        return model_cls.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ArtifactWriteRejected(str(exc), artifact_id=artifact_id) from exc


def _write_question(
    store: ArtifactStore,
    payload: UncheckedJsonObject,
    provenance: Provenance,
) -> ArtifactVersionInfo:
    validated = _validated("question", ARTIFACT_CONTRACTS["question"], payload)
    return store.write_version(
        "question",
        provenance=provenance,
        derived_from={},
        produced_by=None,
        json_files={json_filename("question", "question"): validated},
    )


_CONTRACT_WRITES: frozenset[ArtifactId] = frozenset({"baseline_report"})


def _write_contract_artifact(
    store: ArtifactStore,
    state: EpisodeState,
    artifact_id: ArtifactId,
    payload: UncheckedJsonObject,
    provenance: Provenance,
) -> ArtifactVersionInfo:
    validated = _validated(artifact_id, ARTIFACT_CONTRACTS[artifact_id], payload)
    for result in validated["simulation_results"]:
        basis = result["provenance"]
        if basis["model"]["workspace_id"] != store.workspace_id:
            raise ArtifactWriteRejected(
                "Retained simulations must belong to this workspace", artifact_id=artifact_id
            )
        if basis["model"]["version"] not in store.list_versions("model"):
            raise ArtifactWriteRejected(
                "Retained simulation refers to an absent model revision", artifact_id=artifact_id
            )
        from nof1_causal_lab.machine.inference import inference_record
        from nof1_causal_lab.machine.store import EpisodeJournal

        record = inference_record(
            EpisodeJournal(store.workspace_id).read_all(), basis["model"]["version"]
        )
        if record is None:
            raise ArtifactWriteRejected(
                "Retained simulation requires a model produced by inference",
                artifact_id=artifact_id,
            )
    filename = json_filename(artifact_id, artifact_id)
    pins = write_pins(state, transition_spec("baseline_report").consumes)
    return store.write_version(
        artifact_id,
        provenance=provenance,
        derived_from=pins,
        produced_by=None,
        json_files={filename: validated},
    )


def write_model_revision(
    store: ArtifactStore,
    state: EpisodeState,
    payload: object,
    *,
    provenance: Provenance,
    expected_model_version: int | None,
    derived_from: dict[ArtifactId, int],
    produced_by: str | None,
) -> ArtifactVersionInfo:
    """The common optimistic commit boundary for human and operation-authored models."""
    from nof1_causal_lab.machine.moves import validate_model_base

    if reason := validate_model_base(state, expected_model_version):
        raise ArtifactWriteRejected(reason, artifact_id="model")
    if derived_from.get("model", 0) != expected_model_version:
        raise ArtifactWriteRejected(
            "Model provenance must name the expected base revision", artifact_id="model"
        )
    validated = _validated("model", ARTIFACT_CONTRACTS["model"], payload)
    return store.write_version(
        "model",
        provenance=provenance,
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={json_filename("model", "model"): validated},
    )


def execute_write(
    workspace_id: str,
    artifact_id: ArtifactId,
    payload: UncheckedJsonObject,
    provenance: Provenance,
    state: EpisodeState,
    expected_model_version: int | None = None,
) -> TransitionEffects:
    """Validate, persist, and cascade a write move.

    A cascade failure removes versions written during this failed move before
    re-raising, so rejected writes do not become current and do not leave
    listable orphan versions.
    """
    store = ArtifactStore(workspace_id)
    if artifact_id == "model":
        info = write_model_revision(
            store,
            state,
            payload,
            provenance=provenance,
            expected_model_version=expected_model_version,
            derived_from={"model": expected_model_version} if expected_model_version else {},
            produced_by=None,
        )
        return complete_derivation_cascade(store, state, [info])
    if artifact_id == "question":
        info = _write_question(store, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    if artifact_id in _CONTRACT_WRITES:
        info = _write_contract_artifact(store, state, artifact_id, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    raise ArtifactWriteRejected(
        f"artifact '{artifact_id}' has no write executor",
        artifact_id=artifact_id,
    )
