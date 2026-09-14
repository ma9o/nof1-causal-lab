"""Proof-carrying boundaries for numeric causal analysis."""

from typing import Any

import jax.numpy as jnp
import pytest

from nof1_causal_lab.artifacts.identification import (
    IdentifiabilityStatus,
    IdentificationReport,
    IdentifiedTreatmentStatus,
)
from nof1_causal_lab.artifacts.identity import ConstructRef, ModelRevision
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    ParticleMCMCPosterior,
    WarmupProposal,
)


def _design():
    from tests.helpers import complete_test_model, make_model

    model = complete_test_model(make_model(["treatment", "outcome"], [("treatment", "outcome")]))
    return model.revised(default_outcome=ConstructRef(id=model.constructs[1].id))


def _identification():
    model = _design()
    return IdentificationReport(
        outcome=model.constructs[1].id,
        status=IdentifiabilityStatus(
            identifiable_treatments={
                model.constructs[0].id: IdentifiedTreatmentStatus(
                    method="do_calculus", estimand="E[outcome | do(treatment)]"
                )
            }
        ),
    )


def _conditioned():
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from tests.model_fixtures import parameter_draws

    model = _design()
    result = ParticleMCMCPosterior(
        JointPosteriorDraws(parameter_draws(model, 3), jnp.arange(12.0).reshape(3, 2, 2))
    )
    return condition_model(model, result, times=jnp.arange(2))


def test_identification_proof_is_estimand_specific() -> None:
    design = _design()
    design_ref = ModelRevision(workspace_id="workspace", version=1)

    proof = certify_identified_estimand(
        design,
        _identification(),
        model_revision=design_ref,
        treatment="treatment",
        outcome="outcome",
    )

    assert proof.treatment == "treatment"
    assert proof.outcome == "outcome"
    assert proof.method == "do_calculus"
    assert proof.estimand == "E[outcome | do(treatment)]"


def test_identification_contract_rejects_linear_iv_evidence_for_nonlinear_models():
    from pydantic import ValidationError

    payload = _identification().model_dump(mode="json")
    finding = next(iter(payload["status"]["identifiable_treatments"].values()))
    finding.update(method="instrumental_variable", estimand="IV(Z) [requires linearity]")
    with pytest.raises(ValidationError, match="do_calculus"):
        IdentificationReport.model_validate(payload)


def test_identification_proof_rejects_unidentified_treatment() -> None:
    with pytest.raises(ValueError, match="is not identified"):
        certify_identified_estimand(
            _design(),
            _identification(),
            model_revision=ModelRevision(workspace_id="workspace", version=1),
            treatment="outcome",
            outcome="outcome",
        )


def test_conditioning_rejects_warmup_from_untyped_boundary():
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model

    untyped: Any = WarmupProposal(_samples={})
    with pytest.raises(TypeError, match="production particle-MCMC"):
        condition_model(_design(), untyped, times=jnp.arange(2))


def test_causal_reporting_requires_retained_uncertainty_and_exact_engine_evidence():
    from nof1_causal_lab.models.causal_proofs import certify_conditioned_model
    from tests.inference_fixtures import inference_log

    model = _conditioned()
    revision = ModelRevision(workspace_id="workspace", version=2)
    record = inference_log(model)
    certify_conditioned_model(model, revision, record)
    with pytest.raises(ValueError, match="committed inference"):
        certify_conditioned_model(model, revision.model_copy(update={"version": 3}), record)
    with pytest.raises(ValueError, match="differs from"):
        certify_conditioned_model(_design(), revision, record)
    with pytest.raises(ValueError, match="production particle-MCMC"):
        certify_conditioned_model(
            model,
            revision,
            record.model_copy(
                update={
                    "diagnostics": {
                        **record.diagnostics,
                        "engine_evidence": {"engine": "map", "latent_transition": "euler_maruyama"},
                    }
                }
            ),
        )
    prior = _design()
    with pytest.raises(ValueError, match="no retained joint uncertainty"):
        certify_conditioned_model(prior, revision, inference_log(prior))


def test_causal_analysis_joins_matching_proofs():
    from tests.inference_fixtures import inference_log

    design = _conditioned()
    design_ref = ModelRevision(workspace_id="workspace", version=2)
    analysis = CertifiedCausalAnalysis(
        model=design,
        identification=_identification(),
        model_revision=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                _identification(),
                model_revision=design_ref,
                treatment="treatment",
                outcome="outcome",
            ),
        ),
        inference=inference_log(design),
    )
    assert analysis.treatments == ["treatment"]
    assert analysis.outcome == "outcome"


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
