"""Forward simulation of the declared nonlinear dynamics under interventions.

Dynestyx runs deterministic drift paths. The SDE adapter retains direct Diffrax
integration for indexed Brownian increments and seeded replay. Estimands and
orchestration over posterior draws live in the counterfactual package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import diffrax as dfx
import dynestyx as dsx
import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from dynestyx.solvers import solve_ode_state_path
from jax import Array

from .vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from .intervention import Intervention
    from .vector_field import VectorField


class SimulationConfig(eqx.Module):
    """Solver configuration. An ``eqx.Module`` (pytree) rather than a plain
    dataclass so a *traced* ``sde_dt`` array can flow through ``filter_jit``
    as a leaf: per-draw CFL-capped step sizes then reuse one compiled
    program instead of baking each value in as a constant (one XLA compile
    per prior draw)."""

    rtol: float = 1e-4
    atol: float = 1e-6
    max_steps: int = 4096
    sde_dt: float | Array | None = None
    """Constant step size for the SDE solver. ``None`` → ``(t1 - t0) / 200``."""
    sde_brownian_tol: float = 1e-3
    """Tolerance for ``VirtualBrownianTree``; smaller = finer Brownian path."""
    use_indexed_brownian_path: bool = eqx.field(static=True, default=False)
    """Use deterministic integer-step Brownian increments for fixed-step simulation."""


class _IndexedBrownianPath(dfx.AbstractBrownianPath[Array | dfx.BrownianIncrement]):
    """Fast deterministic Brownian increments for a fixed-step forward solve.

    Every solver interval is keyed by its integer step index rather than the
    floating-point bit patterns of its endpoints. This has the fixed-step speed
    of ``UnsafeBrownianPath`` while making the same seed and schedule replay the
    same increments across process restarts.
    """

    # Diffrax declares these through Equinox AbstractVar. Ty currently models
    # AbstractVar as a class variable even though Diffrax requires instance fields.
    t0: Array  # ty: ignore[invalid-attribute-override]
    t1: Array  # ty: ignore[invalid-attribute-override]
    shape: tuple[int, ...] = eqx.field(static=True)
    key: Array
    step_size: Array
    levy_area: type[dfx.BrownianIncrement] = eqx.field(static=True, default=dfx.BrownianIncrement)

    @eqx.filter_jit  # noqa: V105 - required by the Diffrax AbstractBrownianPath protocol
    @override
    def evaluate(self, t0, t1=None, left: bool = True, use_levy: bool = False):
        del left
        if t1 is None:
            raise ValueError("Indexed Brownian paths only support interval increments")
        dtype = jnp.result_type(t0, t1)
        start = jnp.asarray(t0, dtype=dtype)
        end = jnp.asarray(t1, dtype=dtype)
        step_index = jnp.rint((start - self.t0) / self.step_size).astype(jnp.int32)
        increment_key = random.fold_in(self.key, step_index)
        dt = end - start
        increment = random.normal(increment_key, self.shape, dtype=dtype) * jnp.sqrt(dt)
        if use_levy:
            return self.levy_area(dt=dt, W=increment)
        return increment


def simulate(
    vector_field: VectorField,
    params: tuple[dict[str, Array], ...],
    intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
    *,
    key: Array | None = None,
    diffusion_cov: Array | None = None,
    input_effect: Array | None = None,
    transition_inputs: Array | None = None,
) -> Array:
    """Forward-simulate the SSM trajectory under ``intervention``.

    Two modes:

    - **Deterministic drift trajectory** (default, ``key=None``): the ODE
      ``dy/dt = f(t, y; θ)`` is integrated with an adaptive Tsit5 solver.
      For nonlinear drift, this path generally differs from the SDE mean.

    - **Single SDE sample** (``key`` and ``diffusion_cov`` both provided):
      the SDE ``dy = f(t, y; θ) dt + L dW`` is integrated with a Heun
      SDE solver, where ``L = chol(diffusion_cov + jitter·I)`` and ``dW``
      uses indexed increments or ``VirtualBrownianTree`` according to the
      solver configuration. Independent keys produce independent sample paths
      for distributional counterfactuals.

    Args:
        vector_field: Drift callable for the SSM.
        params: Parameter pytree for the field.
        intervention: Override set active over the integration window.
        initial_state: ``(n_latent,)`` state at ``time_grid[0]``. Hard
            variable overrides are applied to this state before integration.
        time_grid: ``(T,)`` monotonically increasing array of evaluation
            times. ``time_grid[0]`` is the integration start.
        config: Solver tolerances; defaults are conservative for stable
            linear systems and adequate for moderately stiff non-linear
            fields.
        key: Optional JAX PRNG key. When provided alongside
            ``diffusion_cov``, the simulator returns one SDE sample.
        diffusion_cov: Optional ``(n_latent, n_latent)`` PSD diffusion
            covariance ``G·G'``. State-independent (additive Wiener).
        input_effect: Optional ``(n_latent, n_inputs)`` known-input coefficients.
        transition_inputs: ``(T, n_inputs)`` forcing values indexed by destination
            state, with the last interval's value held at the final endpoint.

    Returns:
        ``(T, n_latent)`` state trajectory at the requested grid. Deterministic
        drift trajectory in ODE mode, one sample path in SDE mode.
    """
    from nof1_causal_lab.models.ssm.execution.dynamical_model import (
        StructuralDrift,
        continuous_state_evolution,
    )

    cfg = config or SimulationConfig()
    args = VectorFieldArgs(params=params, intervention=intervention)

    t0 = time_grid[0]
    t1 = time_grid[-1]
    y0 = vector_field.initial_condition(initial_state, args, t0)
    control_dim = 0 if input_effect is None else input_effect.shape[1]
    controls = None
    if input_effect is not None and control_dim:
        if transition_inputs is None:
            raise ValueError("SSM has known input effects but transition_inputs was not provided.")
        controls = jnp.asarray(transition_inputs, dtype=input_effect.dtype)
        expected = (time_grid.shape[0], control_dim)
        if controls.shape != expected:
            raise ValueError(f"transition_inputs must have shape {expected}, got {controls.shape}")

    if key is None or diffusion_cov is None:
        if (key is None) != (diffusion_cov is None):
            raise ValueError(
                "SDE mode requires both 'key' and 'diffusion_cov'; got "
                f"key={'set' if key is not None else 'None'}, "
                f"diffusion_cov={'set' if diffusion_cov is not None else 'None'}."
            )
        initial_dt = jnp.maximum((t1 - t0) / 256.0, 1e-6)
        # The latent process is fully observed here; only its state path is used.
        dynamics = dsx.DynamicalModel(
            initial_condition=dist.Delta(y0, event_dim=1),
            state_evolution=dsx.DeterministicContinuousTimeStateEvolution(
                drift=StructuralDrift(vector_field, args, input_effect),
            ),
            observation_model=_latent_observation,
            control_dim=control_dim,
            t0=t0,
        )
        return solve_ode_state_path(
            dynamics,
            initial_state=y0,
            t0=t0,
            path_times=time_grid,
            # Inputs index the destination state. Dynestyx's continuous controls
            # are right-continuous values at interval starts.
            ctrl_times=None if controls is None else time_grid,
            ctrl_values=(
                None if controls is None else jnp.concatenate([controls[1:], controls[-1:]])
            ),
            diffeqsolve_settings={
                "solver": dfx.Tsit5(),
                "stepsize_controller": dfx.PIDController(rtol=cfg.rtol, atol=cfg.atol),
                "dt0": initial_dt,
                "max_steps": cfg.max_steps,
                "throw": False,
            },
        )

    # SDE path: f(t,y) dt + L dW with L = chol(diffusion_cov).
    n_latent = vector_field.n_latent
    dtype = y0.dtype
    diffusion_cov = jnp.asarray(diffusion_cov, dtype=dtype)
    jitter = jnp.asarray(1e-8, dtype=dtype)
    evolution = continuous_state_evolution(
        vector_field,
        params,
        diffusion_cov + jitter * jnp.eye(n_latent, dtype=dtype),
        input_effect,
        intervention=intervention,
    )

    if cfg.use_indexed_brownian_path:
        if cfg.sde_dt is None:
            raise ValueError("Indexed Brownian simulation requires an explicit fixed step size")
        brownian = _IndexedBrownianPath(
            t0=t0,
            t1=t1,
            shape=(n_latent,),
            key=key,
            step_size=jnp.asarray(cfg.sde_dt),
        )
        adjoint = dfx.ForwardMode()
    else:
        brownian = dfx.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=cfg.sde_brownian_tol,
            shape=(n_latent,),
            key=key,
        )
        adjoint = dfx.RecursiveCheckpointAdjoint()

    def drift_at(t, y, runtime):
        active_evolution, grid, inputs = runtime
        control = None
        if inputs is not None:
            index = jnp.clip(jnp.searchsorted(grid, t, side="right"), 1, grid.shape[0] - 1)
            control = inputs[index]
        return active_evolution.total_drift(x=y, u=control, t=t)

    ode_term = dfx.ODETerm(drift_at)
    diffusion_term = dfx.ControlTerm(
        lambda t, y, a: a[0].diffusion.as_matrix(x=y, u=None, t=t, state_dim=n_latent), brownian
    )
    term = dfx.MultiTerm(ode_term, diffusion_term)
    solver = dfx.Heun()
    dt0 = cfg.sde_dt if cfg.sde_dt is not None else float((t1 - t0) / 200.0)
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(evolution, time_grid, controls),
        saveat=dfx.SaveAt(ts=time_grid),
        max_steps=cfg.max_steps,
        throw=False,
        adjoint=adjoint,
    )
    return solution.ys


def _latent_observation(x, u, t):
    del u, t
    return dist.Delta(x, event_dim=1)


def simulate_pair(
    vector_field: VectorField,
    params: tuple[dict[str, Array], ...],
    baseline_intervention: Intervention,
    action_intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
) -> tuple[Array, Array, Array]:
    """Simulate baseline and action paths and return ``(baseline, action,
    effect)`` where ``effect = action - baseline``.

    Sharing ``initial_state`` and ``time_grid`` between the two integrations
    makes the contrast a pure subtraction at matching grid points.
    """
    baseline = simulate(
        vector_field,
        params,
        baseline_intervention,
        initial_state,
        time_grid,
        config,
    )
    action = simulate(
        vector_field,
        params,
        action_intervention,
        initial_state,
        time_grid,
        config,
    )
    return baseline, action, action - baseline
