"""Proof-carrying boundaries for numeric causal analysis."""

import pickle
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.causal_design import (
    CausalDesign,
    IdentifiabilityStatus,
    IdentifiedTreatmentStatus,
)
from nof1_causal_lab.artifacts.identity import CausalDesignRef, ConstructRef
from nof1_causal_lab.artifacts.latent_structure import (
    CausalEdge,
    Construct,
    LatentStructure,
    Role,
    TemporalStatus,
)
from nof1_causal_lab.artifacts.measurement_structure import MeasurementStructure
from nof1_causal_lab.artifacts.posterior import PosteriorProvenance
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
    certify_reportable_posterior,
)
from nof1_causal_lab.models.ssm.inference.types import (
    FittedArtifact,
    JointPosteriorDraws,
    ParticleMCMCPosterior,
    WarmupProposal,
)
from tests.ssm_spec_fixtures import block_ssm_spec, full_dense_matrix_dynamics_spec


def _design() -> CausalDesign:
    return CausalDesign(
        latent=LatentStructure(
            default_outcome=ConstructRef(id="construct:a1cddfa8657e4a8cb3ae"),
            constructs=[
                Construct(
                    id="construct:c270c8ac9df81ed2ec60",
                    name="treatment",
                    description="Treatment",
                    role=Role.EXOGENOUS,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
                Construct(
                    id="construct:a1cddfa8657e4a8cb3ae",
                    name="outcome",
                    description="Outcome",
                    role=Role.ENDOGENOUS,
                    temporal_status=TemporalStatus.TIME_VARYING,
                ),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:c270c8ac9df81ed2ec60",
                    effect_id="construct:a1cddfa8657e4a8cb3ae",
                    id="edge:b3472c637d08744910a8",
                    description="Test edge",
                )
            ],
        ),
        measurement=MeasurementStructure(indicators=[], model_clock="1d"),
        identifiability=IdentifiabilityStatus(
            identifiable_treatments={
                "construct:c270c8ac9df81ed2ec60": IdentifiedTreatmentStatus(
                    method="do_calculus",
                    estimand="E[outcome | do(treatment)]",
                )
            }
        ),
    )


def _artifact(
    *,
    workspace_id: str = "workspace",
    causal_design_version: int = 1,
) -> FittedArtifact:
    return FittedArtifact(
        result=ParticleMCMCPosterior(
            draws=JointPosteriorDraws(
                parameters={"vf_0_decay": jnp.ones((2, 2), dtype=jnp.float32)}
            )
        ),
        spec=block_ssm_spec(
            n_latent=2,
            n_manifest=0,
            dynamics_spec=full_dense_matrix_dynamics_spec(2),
            latent_names=["treatment", "outcome"],
            manifest_names=[],
        ),
        times=jnp.array([0.0, 1.0], dtype=jnp.float32),
        provenance=PosteriorProvenance(
            causal_design=CausalDesignRef(
                workspace_id=workspace_id,
                version=causal_design_version,
            ),
            compiled_ssm_version=3,
            panel_version=4,
        ),
    )


def test_identification_proof_is_estimand_specific() -> None:
    design = _design()
    design_ref = CausalDesignRef(workspace_id="workspace", version=1)

    proof = certify_identified_estimand(
        design,
        causal_design_ref=design_ref,
        treatment="treatment",
        outcome="outcome",
    )

    assert proof.treatment == "treatment"
    assert proof.outcome == "outcome"
    assert proof.method == "do_calculus"
    assert proof.estimand == "E[outcome | do(treatment)]"


def test_identification_proof_rejects_unidentified_treatment() -> None:
    with pytest.raises(ValueError, match="is not identified"):
        certify_identified_estimand(
            _design(),
            causal_design_ref=CausalDesignRef(workspace_id="workspace", version=1),
            treatment="outcome",
            outcome="outcome",
        )


def test_reportable_posterior_rejects_warmup_from_untyped_boundary() -> None:
    artifact = _artifact()
    untyped_warmup: Any = WarmupProposal(
        _samples={"vf_0_decay": jnp.ones((2, 2), dtype=jnp.float32)}
    )
    artifact = replace(artifact, result=untyped_warmup)

    with pytest.raises(TypeError, match="requires a ParticleMCMCPosterior"):
        certify_reportable_posterior(artifact)


def test_reportable_posterior_rejects_empty_particle_draws() -> None:
    artifact = _artifact()
    artifact = replace(
        artifact, result=ParticleMCMCPosterior(draws=JointPosteriorDraws(parameters={}))
    )

    with pytest.raises(ValueError, match="no retained samples"):
        certify_reportable_posterior(artifact)


def test_causal_analysis_rejects_cross_design_evidence() -> None:
    design = _design()
    estimand = certify_identified_estimand(
        design,
        causal_design_ref=CausalDesignRef(workspace_id="workspace", version=2),
        treatment="treatment",
        outcome="outcome",
    )

    with pytest.raises(ValueError, match="different designs"):
        CertifiedCausalAnalysis(
            causal_design=design,
            causal_design_ref=CausalDesignRef(workspace_id="workspace", version=2),
            estimands=(estimand,),
            posterior=certify_reportable_posterior(_artifact(causal_design_version=1)),
        )


def test_causal_analysis_joins_matching_proofs() -> None:
    design = _design()
    design_ref = CausalDesignRef(workspace_id="workspace", version=1)
    analysis = CertifiedCausalAnalysis(
        causal_design=design,
        causal_design_ref=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                causal_design_ref=design_ref,
                treatment="treatment",
                outcome="outcome",
            ),
        ),
        posterior=certify_reportable_posterior(_artifact()),
    )

    assert analysis.treatments == ["treatment"]
    assert analysis.outcome == "outcome"


def test_fitted_artifact_preserves_aligned_joint_draws_without_sampler_diagnostics():
    artifact = _artifact()
    parameters = {"beta": jnp.arange(6.0).reshape(3, 2)}
    paths = jnp.arange(12.0).reshape(3, 2, 2)
    draws = JointPosteriorDraws(parameters=parameters, latent_paths=paths)
    result = ParticleMCMCPosterior(draws=draws, diagnostics={"likelihood_backend": lambda: None})
    restored = pickle.loads(pickle.dumps(replace(artifact, result=result)))
    np.testing.assert_array_equal(restored.result.get_samples()["beta"], parameters["beta"])
    np.testing.assert_array_equal(restored.result.draws.latent_paths, paths)
    assert restored.result.diagnostics == {}
    assert restored.provenance == artifact.provenance
    assert restored.result.draws.describe().model_dump() == {
        "n_draws": 3,
        "parameter_shapes": {"beta": [2]},
        "latent_shape": (2, 2),
    }
    assert result.draws is draws


@pytest.mark.parametrize(
    ("parameters", "paths", "message"),
    [
        ({"beta": jnp.zeros((3, 2))}, jnp.zeros((2, 4, 2)), "share the draw axis"),
        ({"beta": jnp.zeros(3), "gamma": jnp.zeros(4)}, None, "share the draw axis"),
        ({"beta": jnp.zeros(3)}, jnp.zeros((1, 3, 4, 2)), "draw, time, and state axes"),
        ({"beta": jnp.array(1.0)}, None, "leading draw axis"),
    ],
)
def test_joint_posterior_rejects_misaligned_or_ambiguous_draw_axes(parameters, paths, message):
    with pytest.raises(ValueError, match=message):
        JointPosteriorDraws(parameters=parameters, latent_paths=paths)
