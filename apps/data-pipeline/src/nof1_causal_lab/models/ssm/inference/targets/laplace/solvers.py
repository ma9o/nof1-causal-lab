"""Optimistix iteration and adjoints backed by the trajectory's sparse precision."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from optimistix._search import EvalGradHessianInv

from .shared import _factor_block_profile_cholesky, _solve_block_profile_from_cholesky

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import PyTree
    from optimistix._search import Eval
    from optimistix._solver.backtracking import _BacktrackingState
    from optimistix._solver.gauss_newton import _NewtonDescentState

_MODE_DAMPING = 1e-6


type Factor = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
type Solve = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
type System = Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]


def _apply_inverse(chol, lower, row_upper, row_lower, solve: Solve, vector) -> jax.Array:
    return solve(chol, lower, vector, row_upper, row_lower)


def _curvature_at(curvature, parameters, y):
    return curvature(y, parameters)


class _StructuredNewton(
    optx.AbstractQuasiNewton[jax.Array, None, EvalGradHessianInv[jax.Array], None]
):
    """Supply Newton curvature; Optimistix owns the search, iteration and termination."""

    system: System = eqx.field(kw_only=True)
    bandwidth: int = eqx.field(kw_only=True)
    row_upper: jax.Array = eqx.field(kw_only=True)
    row_lower: jax.Array = eqx.field(kw_only=True)
    factor: Factor = _factor_block_profile_cholesky
    solve: Solve = _solve_block_profile_from_cholesky
    rtol: float = 1e-3
    atol: float = 1e-3
    norm: Callable[[PyTree[jax.Array]], jax.Array] = optx.two_norm
    use_inverse: bool = True  # noqa: V107 - Optimistix reads this inherited solver setting.
    descent: optx.AbstractDescent[  # noqa: V107 - Optimistix invokes this descent.
        jax.Array, EvalGradHessianInv[jax.Array], _NewtonDescentState[jax.Array]
    ] = cast(
        "optx.AbstractDescent[jax.Array, EvalGradHessianInv[jax.Array], _NewtonDescentState[jax.Array]]",
        optx.NewtonDescent(),
    )
    search: optx.AbstractSearch[
        jax.Array, EvalGradHessianInv[jax.Array], Eval, _BacktrackingState
    ] = cast(
        "optx.AbstractSearch[jax.Array, EvalGradHessianInv[jax.Array], Eval, _BacktrackingState]",
        optx.BacktrackingArmijo(),
    )
    verbose: Callable[..., None] = lambda **_kwargs: None  # noqa: V107 - Optimistix invokes this callback.

    @override  # noqa: V105 - Optimistix calls this initialization override.
    def init_hessian(self, y, f, grad):
        # Optimistix requests a template here, before evaluating the objective.
        chol = jnp.broadcast_to(jnp.eye(y.shape[1], dtype=y.dtype), (*y.shape, y.shape[1]))
        lower = jnp.zeros((self.bandwidth, *chol.shape), dtype=y.dtype)
        return self._hessian_info(y, f, grad, chol, lower), None

    def _hessian_info(self, y, f, grad, chol, lower):
        inverse = lx.FunctionLinearOperator(
            eqx.Partial(_apply_inverse, chol, lower, self.row_upper, self.row_lower, self.solve),
            jax.ShapeDtypeStruct(y.shape, y.dtype),
            tags=lx.positive_semidefinite_tag,
            closure_convert=False,
        )
        return optx.FunctionInfo.EvalGradHessianInv(f, grad, inverse)

    @override  # noqa: V105 - Optimistix calls this override.
    def update_hessian(self, y, y_eval, f_info, f_eval_info, hessian_update_state):
        del y, f_info, hessian_update_state
        diag, upper, _rhs = self.system(y_eval)
        diag = diag + _MODE_DAMPING * jnp.eye(y_eval.shape[1], dtype=y_eval.dtype)[None]
        chol, lower = self.factor(diag, upper, self.row_upper, self.row_lower)
        return self._hessian_info(y_eval, f_eval_info.f, f_eval_info.grad, chol, lower), None

    @override
    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result
        return y, aux, {"num_accepted_steps": state.num_accepted_steps - 1}


class _BandedAdjointSolver(lx.AbstractLinearSolver[tuple[jax.Array, jax.Array]]):
    """Recover Hessian bands by coloring JVPs, without forming a dense Jacobian."""

    bandwidth: int

    @override
    def init(self, operator, options):
        del options
        structure = operator.in_structure()
        n_time, n_state = structure.shape
        times, states = jnp.arange(n_time), jnp.arange(n_state)
        n_colors = min(n_time, 2 * self.bandwidth + 1)
        colors = (times[:, None] % n_colors) * n_state + states[None]
        probes = jnp.eye(n_colors * n_state, dtype=structure.dtype)[colors]
        products = jax.vmap(operator.mv)(jnp.moveaxis(probes, -1, 0))
        diag = products[colors[:, None, :], times[:, None, None], states[None, :, None]]
        upper = jnp.stack(
            tuple(
                jnp.where(
                    (times + offset < n_time)[:, None, None],
                    products[
                        colors[jnp.minimum(times + offset, n_time - 1), None, :],
                        times[:, None, None],
                        states[None, :, None],
                    ],
                    0.0,
                )
                for offset in range(1, self.bandwidth + 1)
            )
        )
        row_upper = jnp.minimum(self.bandwidth, n_time - 1 - times)
        row_lower = jnp.minimum(self.bandwidth, times)
        return _factor_block_profile_cholesky(diag, upper, row_upper, row_lower)

    @override  # noqa: V105 - Lineax calls this override.
    def compute(self, state, vector, options):
        del options
        chol, lower = state
        times = jnp.arange(chol.shape[0])
        row_upper = jnp.minimum(self.bandwidth, chol.shape[0] - 1 - times)
        row_lower = jnp.minimum(self.bandwidth, times)
        solution = _solve_block_profile_from_cholesky(chol, lower, vector, row_upper, row_lower)
        return solution, lx.RESULTS.successful, {}

    @override  # noqa: V105 - Lineax calls this adjoint override.
    def transpose(self, state, options):
        return state, options

    @override
    def conj(self, state, options):
        return jax.tree.map(jnp.conj, state), options

    @override  # noqa: V105 - Lineax calls this rank declaration.
    def assume_full_rank(self):
        return True


def solve_latent_mode(
    log_joint,
    system,
    initial: jax.Array,
    args,
    *,
    bandwidth,
    row_upper,
    row_lower,
    max_steps,
    factor=_factor_block_profile_cholesky,
    solve=_solve_block_profile_from_cholesky,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve an initialization mode with a sparse Newton step and sparse implicit AD."""
    curvature = eqx.filter_closure_convert(system, initial, args)
    solution = optx.minimise(
        lambda y, parameters: -log_joint(y, parameters),
        _StructuredNewton(
            system=eqx.Partial(_curvature_at, curvature, args),
            bandwidth=bandwidth,
            row_upper=row_upper,
            row_lower=row_lower,
            factor=factor,
            solve=solve,
        ),
        initial,
        args=args,
        max_steps=max(max_steps, 1) + 1,
        adjoint=optx.ImplicitAdjoint(_BandedAdjointSolver(bandwidth)),
        throw=False,
    )
    return solution.value, {
        "init_log_joint": log_joint(initial, args),
        "n_iterations": jnp.asarray(solution.stats["num_steps"] - 1, dtype=jnp.int32),
        "n_accepted_steps": solution.stats["num_accepted_steps"],
        "final_damping": jnp.asarray(_MODE_DAMPING, dtype=initial.dtype),
    }
