"""Committed inference evidence for tests using small explicit numerical values."""

from typing import TYPE_CHECKING

import jax.numpy as jnp

from tests.git_fixtures import git_oid

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.study.state import ArtifactRecord

_PRIOR_REVISION, _FITTED_REVISION = git_oid(1), git_oid(2)


def inference_log(
    model,
    *,
    revision=_FITTED_REVISION,
    prior_revision=_PRIOR_REVISION,
    seq=3,
    report=None,
    workspace_id="workspace",
):
    pins: dict[ArtifactId, GitOid] = {"model": prior_revision, "panel": git_oid(1)}
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.records import ModelFitResult, StudyRevision
    from tests.action_fixtures import applied_record

    record = applied_record(
        ModelFitResult(
            model=GitRef(workspace_id=workspace_id, revision=prior_revision, path="model.json"),
            panel=GitRef(workspace_id=workspace_id, revision=pins["panel"], path="panel.parquet"),
            produced=(
                ArtifactRecord(
                    artifact_id="model",
                    revision=revision,
                    produced_by="fit",
                    derived_from=pins,
                    model_inputs=input_fingerprints(model),
                ),
            ),
            report=report if report is not None else _report(model),
        ),
        seq=seq,
        ts="2026-07-03T00:00:00+00:00",
    )
    return StudyRevision(commit_id=git_oid(100 + seq), parent_ids=(), record=record)


def _report(model):
    from nof1_causal_lab.artifacts.checks import Evaluated
    from nof1_causal_lab.artifacts.identity import ParameterRef
    from nof1_causal_lab.artifacts.posterior import (
        InferenceMetadata,
        InferenceReport,
        InferenceReportDetail,
    )
    from nof1_causal_lab.artifacts.posterior_diagnostics import (
        ChainDiagnostics,
        ParameterDiagnostics,
        ParticleMCMCEvidence,
    )
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
    from tests.model_fixtures import compile_model_fixture

    binding = parameter_bindings(compile_model_fixture(model))[0][0]
    element, label = next(iter(binding.elements.items()))
    diagnostics = ChainDiagnostics(
        num_chains=4,
        num_samples=3,
        per_parameter=(
            ParameterDiagnostics(
                parameter=label,
                subject=ParameterRef(parameter_id=binding.parameter_id, element_id=element),
                r_hat=1.0,
                ess_bulk=800.0,
                ess_tail=600.0,
                mcse_mean=None,
            ),
        ),
    )
    return InferenceReport(
        time_origin="2024-01-01T00:00:00Z",
        inference_metadata=InferenceMetadata(
            method="marginal_particle_gibbs", n_samples=3, duration_seconds=0
        ),
        engine=Evaluated(
            subject="production_engine", outcome="passed", evidence=ParticleMCMCEvidence()
        ),
        inference_diagnostics=diagnostics,
        sampler_diagnostics=None,
        convergence=parameter_convergence(diagnostics),
        detail=InferenceReportDetail(),
    )


def particle_posterior(draws):
    """Publish explicit single-chain test telemetry with fixed supplied draws."""
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from nof1_causal_lab.models.ssm.inference.types import (
        ParticleMCMCPosterior,
        ProductionDiagnostics,
    )

    chain_samples = {name: values[None, ...] for name, values in draws.parameters.items()}
    n_samples = draws.describe().n_draws
    return ParticleMCMCPosterior(
        draws=draws,
        diagnostics=ProductionDiagnostics(
            mcmc=TrajectoryMCMCResult(
                chain_samples=chain_samples,
                chain_extra_fields={},
                num_chains=1,
                num_samples=n_samples,
            ),
            observation_log_probs=jnp.zeros((1, n_samples, 0)),
        ),
    )
