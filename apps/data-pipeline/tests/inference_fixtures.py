"""Committed inference evidence for tests using small explicit numerical values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import dynestyx as dsx
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.posterior import InferenceReportCore
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.autoreparam import Strategy, _minimal_reparam
from nof1_causal_lab.models.ssm.inference.shared import assemble_parameter_draws
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.numpyro_json import empirical_atoms
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import ArtifactRecord
from tests.git_fixtures import git_oid
from tests.model_fixtures import _make_lgss_data_model_fixture

if TYPE_CHECKING:
    from numpyro.primitives import Message

    from nof1_causal_lab.artifacts.identity import ArtifactId, ConstructId, GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

_PRIOR_REVISION, _FITTED_REVISION = git_oid(1), git_oid(2)


def inference_log(
    model,
    *,
    revision=_FITTED_REVISION,
    prior_revision=_PRIOR_REVISION,
    seq=3,
    workspace_id="workspace",
):
    pins: dict[ArtifactId, GitOid] = {"model": prior_revision, "panel": git_oid(1)}
    from nof1_causal_lab.artifacts.posterior import InferenceEvidence
    from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleMCMCEvidence
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.records import ModelFitResult, StudyRevision
    from tests.action_fixtures import applied_record

    record = applied_record(
        Applied(
            result=ModelFitResult(
                model=GitRef(workspace_id=workspace_id, revision=prior_revision, path="model.json"),
                panel=GitRef(
                    workspace_id=workspace_id, revision=pins["panel"], path="panel.parquet"
                ),
                evidence=InferenceEvidence(
                    distribution=next(iter(model.law_layouts)),
                    engine=ParticleMCMCEvidence(),
                    time_origin="2024-01-01T00:00:00Z",
                    duration_seconds=0,
                ),
            )
            if model.law_layouts
            else None,
            effects=ActionEffects(
                produced=(
                    ArtifactRecord(
                        artifact_id="model", revision=revision, produced_by="fit", derived_from=pins
                    ),
                )
            ),
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
        core=InferenceReportCore(
            time_origin="2024-01-01T00:00:00Z",
            inference_metadata=InferenceMetadata(n_samples=3, duration_seconds=0),
            engine=Evaluated(
                subject="production_engine", outcome="passed", evidence=ParticleMCMCEvidence()
            ),
            inference_diagnostics=diagnostics,
            sampler_diagnostics=None,
            convergence=parameter_convergence(diagnostics),
        ),
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
    return ParticleMCMCPosterior.from_run(
        draws=draws,
        diagnostics=ProductionDiagnostics(
            mcmc=TrajectoryMCMCResult(
                chain_samples=chain_samples,
                chain_extra_fields={},
                num_chains=1,
                num_samples=n_samples,
            ),
            observation_log_probs=jnp.zeros(
                (1, n_samples, draws.latent_paths.shape[1] if draws.latent_paths is not None else 0)
            ),
        ),
    )


def affine_test_evolution(A, covariance, b=None, B=None):
    """Library-owned exact affine reference, restricted to test data and comparisons."""
    return dsx.discretize_state_evolution(
        dsx.StochasticContinuousTimeStateEvolution(
            drift=dsx.AffineDrift(A=A, b=b, B=B),
            diffusion=dsx.FullDiffusion(jnp.linalg.cholesky(covariance)),
        ),
        ExactAffineConfig(covariance_jitter=0.0),
    )


class MinimalReparam(Strategy):
    """Test-owned minimal reparameterization strategy."""

    @override
    def configure(self, msg: Message):
        return _minimal_reparam(msg["fn"], is_observed=msg.get("is_observed", False))


def make_lgss_data(
    *,
    T: int = 100,
    dt: float = 1.0,
    decay_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Build 1D linear-Gaussian SSM data plus a free-parameter ModelSpec.

    Returns a dict with ``observations``, ``times``, ``spec``, the true
    parameter values, and ``n_latent`` for convenience. Used by recovery
    checks that fit the same canonical 1D model with different inference
    methods.
    """
    n_latent, n_manifest = 1, 1

    true_dynamics = jnp.array([[decay_diag]])
    true_diff_cov = jnp.array([[diff_sd**2]])
    true_obs_var = jnp.array([[obs_sd**2]])

    parameters = affine_test_evolution(true_dynamics, true_diff_cov).params_at(0.0, dt)
    Ad, Qd = parameters.A, parameters.cov
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(seed)
    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, obs_key = random.split(key)
    observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    spec = _make_lgss_data_model_fixture()

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_decay_diag": decay_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }


def make_observation_support_runtime(**kwargs: Any) -> ObservationSupportRuntime:
    """Build ObservationSupportRuntime while accepting 2D interval coefficient inputs."""
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime.assembled(**kwargs)


def parameter_draws(model: ModelSpec, n_draws: int) -> dict[str, jnp.ndarray]:
    """Repeat the authored prior reference point without invoking inference."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
    from nof1_causal_lab.prior_distributions import prior_reference_value

    priors, _, _ = compile_priors(compile_model_fixture(model), StructuralSelection(model, None))
    return {
        name: jnp.broadcast_to(value, (n_draws, *value.shape))
        for name, law in priors.items()
        for value in [jnp.asarray(prior_reference_value(law))]
    }


def compile_fit_fixture(spec: ModelSpec, outcome: ConstructId | None = None):
    """Require real compilation in fixtures instead of forging fit evidence."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        compile_ssm_inputs_from_model,
    )

    inputs = compile_ssm_inputs_from_model(StructuralSelection(spec, outcome))
    assert isinstance(inputs, CompiledFitInputs), inputs
    return inputs


def compile_model_fixture(spec: ModelSpec, outcome: ConstructId | None = None):
    """Compile native execution facts without imposing the fitting law restrictions."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    return compile_executable_model(StructuralSelection(spec, outcome))


def bind_panel_fixture(model, observations, times, *, support=None):
    """Publish a complete numerical test panel, including its identity-bearing rows."""
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support
    from nof1_causal_lab.models.ssm.runtime import BoundPanel, bind_panel

    observations, times = jnp.asarray(observations), jnp.asarray(times)
    support = (
        simulation_observation_support(model, np.asarray(times)) if support is None else support
    )
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    rows = []
    for i, observation in enumerate(model.observations):
        for t, at in enumerate(np.asarray(times)):
            start, end = support.support_start_times[t, i], support.support_end_times[t, i]
            rows.append(
                {
                    "indicator_id": str(observation.id),
                    "value": float(observations[t, i]),
                    "anchor_time": origin + timedelta(days=float(at)),
                    "support_start": origin + timedelta(days=float(start))
                    if np.isfinite(start)
                    else None,
                    "support_end": origin + timedelta(days=float(end))
                    if np.isfinite(end)
                    else None,
                    "support_kind": support.support_kinds[i],
                    "summary_operator": support.summary_operators[i],
                    "anchor_policy": support.anchor_policies[i],
                    "observation_window": support.observation_windows[i],
                }
            )
    panel = bind_panel(pl.DataFrame(rows), model=model, time_origin=origin)
    assert isinstance(panel, BoundPanel), panel
    return panel


def _scientific_draws(model_spec: CompiledModel) -> JointPosteriorDraws:
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    if len(model_spec.laws) != 1:
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    law = model_spec.laws[0]
    states = tuple(state.id for state in model_spec.states if not state.is_input)
    if set(law.layout.constructs) != set(states):
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    parameters, paths = law.layout.unpack(jnp.asarray(empirical_atoms(law.distribution)))
    return JointPosteriorDraws(
        parameters=dict(parameters.items()),
        latent_paths=jnp.stack([paths[identity] for identity in law.layout.constructs], axis=-1),
        state_ids=law.layout.constructs,
    )


def model_draws(
    model_spec: CompiledModel, *, input_values: jnp.ndarray | None = None
) -> JointPosteriorDraws:
    """Derive native tensors from the current model's aligned particle distribution."""
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    retained = _scientific_draws(model_spec)
    samples = assemble_parameter_draws(
        model_spec, retained.parameters, count=retained.describe().n_draws
    )
    paths = retained.latent_paths
    state_ids = [state.id for state in model_spec.states if not state.is_input]
    if paths is not None:
        if set(retained.state_ids) != set(state_ids):
            raise ValueError("Stored trajectories do not match ModelSpec construct identities")
        paths = paths[..., [retained.state_ids.index(identity) for identity in state_ids]]
        if numeric.input_mask(model_spec).any():
            if input_values is None:
                raise ValueError(
                    "Retained histories with exogenous inputs require the replayed panel path"
                )
            full_ids = numeric.state_ids(model_spec)
            full_paths = jnp.broadcast_to(input_values, (paths.shape[0], *input_values.shape))
            paths = full_paths.at[
                :, :, jnp.asarray([full_ids.index(identity) for identity in state_ids])
            ].set(paths)
            state_ids = full_ids
    return JointPosteriorDraws(parameters=samples, latent_paths=paths, state_ids=tuple(state_ids))
