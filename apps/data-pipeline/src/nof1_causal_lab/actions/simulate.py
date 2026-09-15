"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.expressions import hill_applications
from nof1_causal_lab.artifacts.simulation import SimulationFinding, SimulationReport
from nof1_causal_lab.models.posterior_predictive import measure_predictive_checks
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    predictive_keys,
    simulate_predictive_draws,
)
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    DesignInfo,
    measure_construct_simulation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.simulation import SimulationSpec


def simulate(
    model: ModelSpec,
    design: SimulationSpec,
    *,
    revision: ModelRevision,
    write_array: Callable[[np.ndarray], str],
    comparison_data: pl.DataFrame | None = None,
    comparison_panel_version: int | None = None,
) -> SimulationReport:
    """Replicate trajectories; comparisons and paired edge experiments consume that batch."""
    model.check_execution()
    times = jnp.asarray(design.times)
    indicator_ids = tuple(numeric.observation_ids(model))
    state_ids = tuple(numeric.state_ids(model))
    observations = None
    support = None
    if comparison_data is not None:
        from nof1_causal_lab.models.ssm.observation_support import (
            augment_wide_data_with_support_boundaries,
            compile_observation_support_runtime,
            validate_observation_support,
        )
        from nof1_causal_lab.models.ssm.runtime import prepare_fit_inputs, project_observation_data

        wide, rows = project_observation_data(comparison_data, model_spec=model)
        wide = augment_wide_data_with_support_boundaries(
            rows, wide, numeric.observation_names(model)
        )
        validate_observation_support(model, wide)
        observations, observed_times, names, wide = prepare_fit_inputs(model, wide)
        if observed_times.shape != times.shape or not np.allclose(
            observed_times + design.comparison_time_offset, times
        ):
            raise ValueError("Comparison data and simulation design must have the same time grid")
        support = compile_observation_support_runtime(rows, wide, names)
        if support is not None and design.comparison_time_offset:
            from dataclasses import replace

            support = replace(
                support,
                anchor_times=support.anchor_times + design.comparison_time_offset,
                support_start_times=support.support_start_times + design.comparison_time_offset,
                support_end_times=support.support_end_times + design.comparison_time_offset,
            )
    if comparison_data is None:
        from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support

        support = simulation_observation_support(model, np.asarray(times))
    mask = None if observations is None else ~jnp.isnan(observations)
    laws = sample_model_laws(model, draws=design.draws, key=predictive_keys(design.seed).parameters)
    initial = None
    if design.initial_state == "retained":
        if laws.latent_paths is None or design.state_time not in model.time_points:
            raise ValueError("The model has no retained joint state at state_time")
        initial = laws.latent_paths[:, model.time_points.index(design.state_time), :]
    elif design.initial_state == "fixed":
        if set(design.state_values) != set(state_ids):
            raise ValueError("Fixed initial states must name every model state exactly once")
        initial = jnp.broadcast_to(
            jnp.asarray([design.state_values[k] for k in state_ids]), (design.draws, len(state_ids))
        )
    elif design.initial_state == "equilibrium":
        import jax

        from nof1_causal_lab.models.ssm.dynamics import Intervention, compute_steady_state
        from nof1_causal_lab.models.ssm.dynamics.posterior import posterior_dynamics_from_samples

        dynamics = posterior_dynamics_from_samples(model, laws.parameters)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *dynamics.param_samples)
        initial = jax.vmap(
            lambda p: compute_steady_state(dynamics.vector_field, p, Intervention.none()),
            axis_size=design.draws,
        )(stacked)
    from nof1_causal_lab.models.ssm.counterfactual.orchestration import ClampSpec

    clamps = []
    for clamp in design.interventions:
        if clamp.target not in state_ids:
            raise ValueError(f"Intervention target is not a model state: {clamp.target}")
        clamps.append(
            ClampSpec(index=state_ids.index(clamp.target), **clamp.model_dump(exclude={"target"}))
        )
    prediction = simulate_predictive_draws(
        model,
        laws.parameters,
        times,
        seed=design.seed,
        initial_states=initial,
        process_noise=design.process_noise,
        observation_noise=design.observation_noise,
        clamps=tuple(clamps),
        observation_support=support,
        observation_mask=mask,
    )
    indices: dict[str, np.ndarray] = {
        identity: np.flatnonzero(np.isfinite(prediction["observations"][:, :, i]).any(axis=0))
        if observations is None
        else np.flatnonzero(np.isfinite(observations[:, i]))
        for i, identity in enumerate(indicator_ids)
    }
    values: dict[str, np.ndarray] = {
        identity: np.asarray([])
        if observations is None
        else np.asarray(observations[indices[identity], i])
        for i, identity in enumerate(indicator_ids)
    }
    measurement_design = DesignInfo(
        t_grid=times,
        manifest_ids=indicator_ids,
        obs_index_by_indicator=indices,
        values_by_indicator=values,
        n_draws=design.draws,
        seed=design.seed,
        observation_support=support,
        c1b_growth_ratio=design.confinement_growth_ratio,
        c1b_max_explosive_frac=design.confinement_failure_fraction,
    )
    findings = []
    components = numeric.dynamics_expressions(model)
    for state_index, identity in enumerate(
        state_ids if set(design.checks) & {"dynamics", "measurement"} else ()
    ):
        incoming = [
            component
            for component in components
            if component.target == state_index and component.edge_owned
        ]
        parents = sorted({source for component in incoming for source in component.sources})
        hills = {
            source
            for component in incoming
            if any(hill_applications(component.expression))
            for source in component.sources
        }
        target = ConstructSimulationTarget(
            construct=model.get_construct(identity),
            edge_parents=tuple(model.get_construct(state_ids[source]).name for source in parents),
            hill_parents=tuple(
                model.get_construct(state_ids[source]).name for source in sorted(hills)
            ),
        )
        measured, _ = measure_construct_simulation(
            model,
            prediction,
            measurement_design,
            target,
            edge_contrasts=design.edge_contrasts,
        )
        findings.extend(
            SimulationFinding(
                check=result.check,
                target=result.target,
                value=result.value,
                criterion=result.band,
                passed=result.passed,
                explanation=result.note,
            )
            for result in measured
            if ("measurement" if result.check.startswith("C5") else "dynamics") in design.checks
        )
    checks = (
        None
        if observations is None
        or "data_comparison" not in design.checks
        or not design.observation_noise
        else measure_predictive_checks(prediction["observations"], observations, indicator_ids)
    )
    outputs = {
        "latents",
        "linear_predictors",
        "observations",
        "observations_mask",
        "expected_observations",
        "reference_latents",
        "reference_observations",
    }
    if "data_comparison" in design.checks and checks is None:
        findings.append(
            SimulationFinding(
                check="data_comparison",
                target="observations",
                value="not_evaluated",
                criterion="Comparison observations and sampled emission noise",
                passed=None,
                explanation="Select comparison data and enable observation noise to assess predictive calibration.",
            )
        )
    return SimulationReport(
        model=revision,
        design=design,
        comparison_panel_version=comparison_panel_version,
        state_ids=state_ids,
        indicator_ids=indicator_ids,
        parameter_draws={
            name: write_array(np.asarray(value))
            for name, value in prediction.items()
            if name not in outputs
        },
        latent_paths=write_array(np.asarray(prediction["latents"])),
        observations=write_array(np.asarray(prediction["observations"])),
        findings=tuple(findings),
        predictive_checks=checks,
        reference_latent_paths=write_array(np.asarray(prediction["reference_latents"]))
        if design.interventions
        else None,
        reference_observations=write_array(np.asarray(prediction["reference_observations"]))
        if design.interventions
        else None,
    )
