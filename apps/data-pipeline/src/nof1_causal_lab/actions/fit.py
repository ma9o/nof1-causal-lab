"""Retain posterior atoms and native telemetry; derive reports through the read cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from pydantic import TypeAdapter

from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated
from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior import (
    InferenceEvidence,
    InferenceMetadata,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleMCMCEvidence
from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.study.store import cached_value

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.observation_data import ObservationDataset
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.numpyro_json import ArrayLoader
    from nof1_causal_lab.sampler_config import SamplerSpec
    from nof1_causal_lab.study.store import ArtifactStore


class FitResult(TypedDict):
    """The fitted model definition paired with its retained numerical inference evidence."""

    _model: ModelSpec
    evidence: InferenceEvidence


def fit(
    *,
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    time_origin: datetime | None,
    sampler: SamplerSpec,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    compute_loo_diagnostics: bool,
) -> FitResult:
    """The exact producer's buffers cross the compute boundary without report reductions."""
    from nof1_causal_lab.actions.inference.fit import fit_model

    fitted = fit_model(selection, data_for_model, time_origin=time_origin, sampler=sampler)
    if "error" in fitted:
        raise ModelFitError(
            fitted["error"], diagnostics={"duration_seconds": fitted["duration_seconds"]}
        )
    result = fitted["result"]
    native = result.diagnostics

    def retain(values: object | None) -> str | None:
        return array_writer(np.asarray(values)) if values is not None else None

    conditioned, identity = condition_model(
        selection.model,
        fitted["panel"].model,
        result,
        times=fitted["panel"].times,
        array_writer=array_writer,
        array_loader=array_loader,
    )
    evidence = InferenceEvidence(
        distribution=identity,
        engine=ParticleMCMCEvidence(),
        time_origin=time_origin,
        duration_seconds=fitted["duration_seconds"],
        num_chains=native.mcmc.num_chains,
        chain_extra_fields={
            name: array_writer(np.asarray(values))
            for name, values in native.mcmc.get_extra_fields(group_by_chain=True).items()
        },
        observation_log_probs=retain(native.observation_log_probs)
        if compute_loo_diagnostics
        else None,
        observed_rows=retain(np.any(np.isfinite(fitted["panel"].observations), axis=1))
        if compute_loo_diagnostics
        else None,
        exact_observation_rows=retain(native.exact_observation_rows),
        sampler_diagnostics=native.marginal_particle_gibbs,
        phase_extra_fields={
            phase: {name: array_writer(np.asarray(values)) for name, values in buffers.items()}
            for phase, buffers in (native.marginal_particle_gibbs_phase_extra_fields or {}).items()
        },
        warmup_complete_log_posterior_history=retain(native.warmup_complete_log_posterior_history),
        all_complete_log_posterior_history=retain(native.all_complete_log_posterior_history),
        initial_latent_delta=retain(result.initial_latent_delta),
        final_latent_delta=retain(result.final_latent_delta),
    )
    return {"_model": conditioned, "evidence": evidence}


def read_inference_report(
    store: ArtifactStore,
    revision: GitOid,
    evidence: InferenceEvidence,
) -> InferenceReport:
    """Rebuild exact-chain reductions from scientific atoms and retained native telemetry."""
    from nof1_causal_lab.study.store import read_model

    def render() -> InferenceReport:
        import jax.numpy as jnp

        from nof1_causal_lab.models.ssm.inference.diagnostics_viz import compute_posterior_marginals
        from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
        from nof1_causal_lab.models.ssm.inference.types import (
            JointPosteriorDraws,
            ParticleMCMCPosterior,
            ProductionDiagnostics,
        )
        from nof1_causal_lab.numpyro_json import empirical_atoms

        model = read_model(store, revision)
        layouts = {evidence.distribution: model.law_layouts[evidence.distribution]}
        atoms = {identity: empirical_atoms(model.distributions[identity]) for identity in layouts}
        count = next(iter(atoms.values())).shape[0]
        samples: dict[str, jnp.ndarray] = {
            element: jnp.asarray(atoms[identity][:, layout.parameter_columns[element]])
            for identity, layout in layouts.items()
            for _, elements in layout.parameters
            for element in elements
        }
        references = {
            ParameterCoordinate(site_name=element, indices=()): (
                layout.labels[element],
                ParameterRef(parameter_id=parameter, element_id=element),
            )
            for layout in layouts.values()
            for parameter, elements in layout.parameters
            for element in elements
        }
        marginals = compute_posterior_marginals(samples, references)
        extra = {
            name: jnp.asarray(store.read_array(ref))
            for name, ref in evidence.chain_extra_fields.items()
        }
        diagnostics = None
        traces, ranks, loo = (), (), None
        if evidence.num_chains is not None and samples:
            chains = evidence.num_chains
            if count % chains:
                raise ValueError("Retained atoms do not match their original chain count")
            posterior = ParticleMCMCPosterior.from_run(
                draws=JointPosteriorDraws(parameters=samples),
                diagnostics=ProductionDiagnostics(
                    mcmc=TrajectoryMCMCResult(
                        chain_samples={
                            name: values.reshape(chains, -1) for name, values in samples.items()
                        },
                        chain_extra_fields=extra,
                        num_chains=chains,
                        num_samples=count // chains,
                    ),
                    observation_log_probs=jnp.asarray(
                        store.read_array(evidence.observation_log_probs)
                    )
                    if evidence.observation_log_probs is not None
                    else jnp.empty((chains, count // chains, 0)),
                    exact_observation_rows=jnp.asarray(
                        store.read_array(evidence.exact_observation_rows)
                    )
                    if evidence.exact_observation_rows is not None
                    else None,
                ),
            )
            diagnostics = posterior.get_inference_diagnostics(references)
            traces, ranks = posterior.get_chain_detail(references)
            if evidence.observation_log_probs is not None:
                assert evidence.observed_rows is not None
                loo = posterior.get_loo_diagnostics(
                    observed_rows=jnp.asarray(store.read_array(evidence.observed_rows))
                )

        def rows(ref: str | None) -> tuple[tuple[float, ...], ...] | None:
            return (
                tuple(tuple(float(value) for value in row) for row in store.read_array(ref))
                if ref is not None
                else None
            )

        return InferenceReport(
            core=InferenceReportCore(
                time_origin=evidence.time_origin,
                inference_metadata=InferenceMetadata(
                    n_samples=count, duration_seconds=evidence.duration_seconds
                ),
                engine=Evaluated(
                    subject="production_engine", outcome="passed", evidence=evidence.engine
                )
                if evidence.engine is not None
                else NotEvaluated(
                    subject="production_engine",
                    reason="ARCHIVED_ENGINE_NOT_RETAINED",
                    detail="The original fit retained no exact-engine marker.",
                ),
                inference_diagnostics=diagnostics,
                sampler_diagnostics=evidence.sampler_diagnostics,
                convergence=parameter_convergence(diagnostics),
                loo_diagnostics=loo[0] if loo is not None else None,
                posterior_marginals=marginals,
            ),
            detail=InferenceReportDetail(
                trace_data=traces,
                rank_histograms=ranks,
                pareto_k=loo[1] if loo is not None else (),
                divergent=tuple(bool(value) for value in extra["diverging"].reshape(-1))
                if "diverging" in extra
                else None,
                initial_latent_delta=rows(evidence.initial_latent_delta),
                final_latent_delta=rows(evidence.final_latent_delta),
            ),
        )

    value, _ = cached_value(
        store.workspace_id,
        ("inference-report", revision, evidence.model_dump_json(round_trip=True)),
        TypeAdapter(InferenceReport),
        render,
    )
    return value
