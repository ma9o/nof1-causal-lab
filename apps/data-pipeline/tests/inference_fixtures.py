"""Committed inference evidence for tests using small explicit numerical values."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo
from nof1_causal_lab.machine.moves import RunOperation
from nof1_causal_lab.machine.store import TransitionRecord
from nof1_causal_lab.models.model_inputs import input_fingerprints


def inference_log(model, *, version=2, seq=3, report=None):
    pins: dict[ArtifactId, int] = {"model": version - 1, "panel": 1}
    return TransitionRecord(
        seq=seq,
        ts="2026-07-03T00:00:00+00:00",
        move=RunOperation(operation_id="posterior"),
        status="applied",
        trace_ids=[],
        resume=None,
        produced=[
            ArtifactVersionInfo(
                artifact_id="model",
                version=version,
                provenance="computed",
                produced_by="run:posterior",
                derived_from=pins,
                model_inputs=input_fingerprints(model),
            )
        ],
        diagnostics={
            "input_pins": pins,
            "engine_evidence": {
                "engine": "marginal_particle_gibbs",
                "latent_transition": "euler_maruyama",
            },
            "report": report
            or {
                "inference_metadata": {
                    "method": "marginal_particle_gibbs",
                    "n_samples": 3,
                    "duration_seconds": 0.0,
                },
                "inference_diagnostics": {},
            },
        },
    )
