"""Committed inference evidence for tests using small explicit numerical values."""

from typing import TYPE_CHECKING

from tests.git_fixtures import git_oid

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.machine.artifacts import ArtifactRecord
from nof1_causal_lab.machine.store import TransitionRecord
from nof1_causal_lab.models.model_inputs import input_fingerprints

_PRIOR_REVISION, _FITTED_REVISION = git_oid(1), git_oid(2)


def inference_log(
    model, *, revision=_FITTED_REVISION, prior_revision=_PRIOR_REVISION, seq=3, report=None
):
    pins: dict[ArtifactId, GitOid] = {"model": prior_revision, "panel": git_oid(1)}
    return TransitionRecord(
        seq=seq,
        ts="2026-07-03T00:00:00+00:00",
        action="fit",
        operation_id="posterior",
        inputs={},
        status="applied",
        trace_ids=[],
        resume=None,
        produced=[
            ArtifactRecord(
                artifact_id="model",
                revision=revision,
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
                "inference_diagnostics": {
                    "mcmc": {
                        "num_chains": 4,
                        "per_parameter": [
                            {
                                "parameter": "beta",
                                "r_hat": 1.0,
                                "ess_bulk": 800.0,
                                "ess_tail": 600.0,
                            }
                        ],
                    }
                },
            },
        },
    )
