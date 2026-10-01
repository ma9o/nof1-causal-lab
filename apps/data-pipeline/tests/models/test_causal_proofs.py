"""Proof-carrying boundaries for numeric causal analysis."""

from nof1_causal_lab.artifacts.model_spec import ModelSpec

import subprocess
from pathlib import Path
from textwrap import dedent

import jax.numpy as jnp
import pytest

from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    ParticleMCMCPosterior,
)
from tests.git_fixtures import git_oid
from tests.model_fixtures import compile_fit_fixture

pytestmark = pytest.mark.contract




def _identification():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/identification__design.json').read_text())
    return IdentificationReport(
        outcome=model.constructs[1].id,
        treatments={
            model.constructs[0].id: IdentifiedTreatmentStatus(
                method="do_calculus", estimand="E[outcome | do(treatment)]"
            )
        },
    )




def test_identification_proof_is_estimand_specific() -> None:
    design = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/identification_proof_is_estimand_specific__design.json').read_text())
    design_ref = GitRef(workspace_id="workspace", revision=git_oid(1), path="model.json")

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
    finding = next(iter(payload["treatments"].values()))
    finding.update(method="instrumental_variable", estimand="IV(Z) [requires linearity]")
    with pytest.raises(ValidationError, match="do_calculus"):
        IdentificationReport.model_validate(payload)


@pytest.mark.parametrize(('explicit_finding', '_design_payload'), [
    pytest.param(False, 'causal_proofs/identification_proof_rejects_unidentified_treatment__design_false.json', id='False'),
    pytest.param(True, 'causal_proofs/identification_proof_rejects_unidentified_treatment__design_true.json', id='True'),
])
def test_identification_proof_rejects_unidentified_treatment(explicit_finding, _design_payload) -> None:
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / _design_payload).read_text())
    report = IdentificationReport(
        outcome=model.constructs[1].id,
        treatments={model.constructs[0].id: NonIdentifiableTreatmentStatus(notes="Unidentified")}
        if explicit_finding
        else {},
    )
    with pytest.raises(ValueError, match="is not identified"):
        certify_identified_estimand(
            model,
            report,
            model_revision=GitRef(workspace_id="workspace", revision=git_oid(1), path="model.json"),
            treatment="treatment",
            outcome="outcome",
        )


def test_conditioning_rejects_warmup_statically(tmp_path):
    probe = tmp_path / "conditioning_types.py"
    probe.write_text(
        dedent("""\
            from jax import Array
            from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
            from nof1_causal_lab.models.ssm.inference.persistence import condition_model
            from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior, WarmupProposal

            def condition(model: CompiledFitInputs, posterior: ParticleMCMCPosterior, warmup: WarmupProposal, times: Array):
                condition_model(model, posterior, times=times)
                condition_model(model, warmup, times=times)
            """)
    )
    checked = subprocess.run(
        [
            "ty",
            "check",
            str(probe),
            "--project",
            str(Path(__file__).parents[2]),
            "--output-format",
            "concise",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 1
    assert checked.stdout.count("error[") == 1, checked.stdout + checked.stderr
    assert "invalid-argument-type" in checked.stdout
    assert "ParticleMCMCPosterior" in checked.stdout
    assert "WarmupProposal" in checked.stdout


def test_causal_reporting_requires_retained_uncertainty_and_converged_exact_engine_evidence():
    from nof1_causal_lab.models.causal_proofs import certify_conditioned_model
    from tests.inference_fixtures import inference_log

    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/causal_reporting_requires_retained_uncertainty_and_converged_exact_engine_evidence__conditioned.json').read_text())
    revision = GitRef(workspace_id="workspace", revision=git_oid(2), path="model.json")
    record = inference_log(model)
    certify_conditioned_model(model, revision, record)
    with pytest.raises(ValueError, match="committed fit"):
        certify_conditioned_model(
            model,
            type(revision).model_validate({**revision.model_dump(), "revision": git_oid(3)}),
            record,
        )
    with pytest.raises(ValueError, match="differs from"):
        certify_conditioned_model(ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/causal_reporting_requires_retained_uncertainty_and_converged_exact_engine_evidence__design.json').read_text()), revision, record)
    with pytest.raises(ValueError, match="production particle-MCMC"):
        certify_conditioned_model(
            model,
            revision,
            type(record).model_validate(
                {
                    **record.model_dump(),
                    "diagnostics": {
                        **record.diagnostics,
                        "engine_evidence": {"engine": "map", "latent_transition": "euler_maruyama"},
                    },
                }
            ),
        )
    prior = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/causal_reporting_requires_retained_uncertainty_and_converged_exact_engine_evidence__design_2.json').read_text())
    with pytest.raises(ValueError, match="no retained joint uncertainty"):
        certify_conditioned_model(prior, revision, inference_log(prior))
    report = record.diagnostics["report"]
    mixed_poorly = {
        **report,
        "inference_diagnostics": {
            "mcmc": {
                "num_chains": 4,
                "per_parameter": [
                    {"parameter": "beta", "r_hat": 1.05, "ess_bulk": 800.0, "ess_tail": 90.0}
                ],
            }
        },
    }
    with pytest.raises(ValueError, match=r"R-hat < 1\.01 fails.*tail ESS ≥ 400 fails"):
        certify_conditioned_model(model, revision, inference_log(model, report=mixed_poorly))


def test_causal_analysis_joins_matching_proofs():
    from tests.inference_fixtures import inference_log

    design = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'causal_proofs/causal_analysis_joins_matching_proofs__conditioned.json').read_text())
    design_ref = GitRef(workspace_id="workspace", revision=git_oid(2), path="model.json")
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
