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
from nof1_causal_lab.machine.graph import ROOTS, transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, write_pins
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState, Provenance


def _validated(
    artifact_id: ArtifactId,
    model_cls: type[BaseModel],
    payload: UncheckedJsonObject,
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


def _write_saved_scenarios(
    store: ArtifactStore,
    state: EpisodeState,
    payload: UncheckedJsonObject,
    provenance: Provenance,
) -> ArtifactVersionInfo:
    validated = _validated("saved_scenarios", ARTIFACT_CONTRACTS["saved_scenarios"], payload)
    for scenario in validated["scenarios"]:
        for item in scenario["evaluations"]:
            evaluation = item["evaluation"]
            evaluation_store = ArtifactStore(evaluation["model"]["id"])
            if evaluation["posterior"]["version"] not in evaluation_store.list_versions(
                "posterior"
            ):
                raise ArtifactWriteRejected(
                    "Saved evaluation refers to an absent posterior version",
                    artifact_id="saved_scenarios",
                )
    roots = {root.artifact_id: root for root in ROOTS}
    return store.write_version(
        "saved_scenarios",
        provenance=provenance,
        derived_from=write_pins(state, roots["saved_scenarios"].write_pins),
        produced_by=None,
        json_files={json_filename("saved_scenarios", "saved_scenarios"): validated},
    )


_CONTRACT_WRITES: frozenset[ArtifactId] = frozenset(
    {
        "latent_structure",
        "measurement_structure",
        "statistical_model_spec",
        "baseline_report",
    }
)


def _write_contract_artifact(
    store: ArtifactStore,
    state: EpisodeState,
    artifact_id: ArtifactId,
    payload: UncheckedJsonObject,
    provenance: Provenance,
) -> ArtifactVersionInfo:
    validated = _validated(artifact_id, ARTIFACT_CONTRACTS[artifact_id], payload)
    filename = json_filename(artifact_id, artifact_id)
    pins = write_pins(state, transition_spec(artifact_id).consumes)
    return store.write_version(
        artifact_id,
        provenance=provenance,
        derived_from=pins,
        produced_by=None,
        json_files={filename: validated},
    )


def execute_write(
    workspace_id: str,
    artifact_id: ArtifactId,
    payload: UncheckedJsonObject,
    provenance: Provenance,
    state: EpisodeState,
) -> TransitionEffects:
    """Validate, persist, and cascade a write move.

    A cascade failure removes versions written during this failed move before
    re-raising, so rejected writes do not become current and do not leave
    listable orphan versions.
    """
    store = ArtifactStore(workspace_id)
    if artifact_id == "question":
        info = _write_question(store, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    if artifact_id == "saved_scenarios":
        info = _write_saved_scenarios(store, state, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    if artifact_id in _CONTRACT_WRITES:
        info = _write_contract_artifact(store, state, artifact_id, payload, provenance)
        return complete_derivation_cascade(store, state, [info])
    raise ArtifactWriteRejected(
        f"artifact '{artifact_id}' has no write executor",
        artifact_id=artifact_id,
    )
