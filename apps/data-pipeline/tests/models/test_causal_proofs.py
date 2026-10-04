"""Proof-carrying boundaries for numeric causal analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.models.causal_proofs import (
    CausalCertificationError,
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from tests.git_fixtures import git_oid
from tests.model_fixtures import (
    load_model_fixture,
)


def _treatment_outcome() -> ModelSpec:
    return load_model_fixture("causal_proofs/treatment_outcome.json")


def _conditioned_treatment_outcome() -> ModelSpec:
    return load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


pytestmark = pytest.mark.contract


def _identification():
    model = _treatment_outcome()
    return IdentificationReport(
        outcome=model.constructs[1].id,
        treatments={
            model.constructs[0].id: IdentifiedTreatmentStatus(estimand="E[outcome | do(treatment)]")
        },
    )


def test_identification_proof_is_estimand_specific() -> None:
    design = _treatment_outcome()
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
    assert proof.estimand == "E[outcome | do(treatment)]"


def test_identification_contract_rejects_linear_iv_evidence_for_nonlinear_models():
    from pydantic import ValidationError

    payload = _identification().model_dump(mode="json")
    finding = next(iter(payload["treatments"].values()))
    finding.update(method="instrumental_variable", estimand="IV(Z) [requires linearity]")
    with pytest.raises(ValidationError, match="Extra inputs"):
        IdentificationReport.model_validate(payload)


@pytest.mark.parametrize(
    ("explicit_finding", "_design_payload"),
    [
        pytest.param(
            False,
            _treatment_outcome,
            id="False",
        ),
        pytest.param(
            True,
            _treatment_outcome,
            id="True",
        ),
    ],
)
def test_identification_proof_rejects_unidentified_treatment(
    explicit_finding, _design_payload
) -> None:
    model = _design_payload()
    report = IdentificationReport(
        outcome=model.constructs[1].id,
        treatments={model.constructs[0].id: NonIdentifiableTreatmentStatus(notes="Unidentified")}
        if explicit_finding
        else {},
    )
    with pytest.raises(CausalCertificationError, match="is not identified"):
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
            from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
            from nof1_causal_lab.artifacts.model_spec import ModelSpec
            from nof1_causal_lab.models.ssm.inference.persistence import condition_model
            from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior, WarmupProposal

            def condition(model: ModelSpec, compiled: CompiledModel, posterior: ParticleMCMCPosterior, warmup: WarmupProposal, times: Array):
                condition_model(model, compiled, posterior, times=times)
                condition_model(model, compiled, warmup, times=times)
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

    model = _conditioned_treatment_outcome()
    revision = GitRef(workspace_id="workspace", revision=git_oid(2), path="model.json")
    from tests.inference_fixtures import _report

    record = inference_log(model)
    report = _report(model)
    certify_conditioned_model(model, revision, record, model, report.core)
    with pytest.raises(CausalCertificationError, match="committed fit"):
        certify_conditioned_model(
            model,
            revision.revised(revision=git_oid(3)),
            record,
            model,
            report.core,
        )
    with pytest.raises(CausalCertificationError, match="differs from"):
        certify_conditioned_model(
            _treatment_outcome(),
            revision,
            record,
            model,
            report.core,
        )
    from nof1_causal_lab.artifacts.checks import NotEvaluated

    unavailable = report.revised(
        core=report.core.revised(
            engine=NotEvaluated(
                subject="production_engine",
                reason="ARCHIVED_ENGINE_NOT_RETAINED",
                detail="Exact engine evidence not retained",
            )
        )
    )
    with pytest.raises(CausalCertificationError, match="retained exact-engine evidence"):
        certify_conditioned_model(model, revision, record, model, unavailable.core)
    prior = _treatment_outcome()
    with pytest.raises(CausalCertificationError, match="committed fit"):
        certify_conditioned_model(prior, revision, inference_log(prior), prior, _report(prior).core)
    from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence

    diagnostics = report.core.inference_diagnostics
    assert diagnostics is not None
    poorly_mixed = diagnostics.revised(
        **{
            "per_parameter": tuple(
                row.revised(**{"r_hat": 1.05, "ess_tail": 90.0})
                for row in diagnostics.per_parameter
            )
        }
    )
    mixed_poorly = report.revised(
        core=report.core.revised(
            inference_diagnostics=poorly_mixed,
            convergence=parameter_convergence(poorly_mixed),
        )
    )
    with pytest.raises(CausalCertificationError, match=r"r_hat fails.*ess_tail fails"):
        certify_conditioned_model(model, revision, record, model, mixed_poorly.core)


def test_causal_analysis_joins_matching_proofs():
    from tests.inference_fixtures import _report, inference_log

    design = _conditioned_treatment_outcome()
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
        fitted_model=design,
        report=_report(design).core,
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
