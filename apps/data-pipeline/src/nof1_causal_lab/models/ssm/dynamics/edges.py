"""Vector-field components — edges and full-vector terms.

A ``VectorFieldComponent`` is anything that contributes to ``dη/dt``. Components
are composed by ``VectorField`` into the system vector field. Two
broad kinds live here:

- **Single-target edges** with explicit source / target indices:
  ``LinearEdge``, ``HillEdge``, ``MultiplicativeEdge``. These are the
  user-facing primitives the LLM elicits for the non-linear vocabulary
  (Linear ≈ baseline coupling; Hill ≈ saturating dose-response;
  Multiplicative ≈ true bilinear interaction).

- **Full-vector terms** that contribute to several latents at once:
  ``DenseLinear`` wraps a posterior-shaped ``A @ η + c`` in one XLA op
  (the fast path for the existing dense-matrix posterior posterior);
  ``DiagonalDecay`` and ``Intercept`` are full-vector background terms;
  ``StateDecay`` and ``StateIntercept`` are scalar, target-owned versions
  used by compiler-produced component specs.

Every component implements the same ``contribute`` signature, so
``VectorField`` is a single uniform loop. Each component reads
its own slice of ``args.params`` (matched by position in the components
tuple), which keeps parameter shapes scoped to the component that owns
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp

from nof1_causal_lab.scalar_functions import hill_response

if TYPE_CHECKING:
    from jax import Array


@runtime_checkable
class VectorFieldComponent(Protocol):
    """Anything that contributes to ``dη/dt``.

    ``contribute`` returns the updated vector-field value. ``eta`` is
    the raw state; ``eta_per_edge`` is the ``(n_target, n_source)``
    matrix with edge-input overrides already applied (single-target
    edges read one entry; full-vector terms typically use one or the
    other depending on whether they care about per-target source
    overrides).
    """

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        eta_per_edge: Array,
        t: Array,
        params: dict[str, Array],
        /,
    ) -> Array: ...


# ---------------------------------------------------------------------------
# Full-vector terms
# ---------------------------------------------------------------------------


class DenseLinear(eqx.Module):
    """``A @ η + c`` contribution as a single matmul.

    Reads ``params['drift']`` (``(n, n)``) and optional ``params['cint']``
    (``(n,)``). This is the fast path for the existing posterior dense
    posterior; it consumes the matrix shape directly without the
    per-edge scatter overhead that would otherwise scale as ``n²`` ops.
    Edge-input overrides go through ``eta_per_edge`` so a single
    ``DenseLinear`` term still honours surgical interventions on
    specific ``j → i`` couplings.
    """

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        A = params["drift"]
        cint = params.get("cint", jnp.zeros(A.shape[0], dtype=accumulator.dtype))
        return accumulator + (A * eta_per_edge).sum(axis=1) + cint


class DiagonalDecay(eqx.Module):
    """Per-latent relaxation term ``-decay · η``.

    Used in the vector-field primitive case where the user has explicit
    natural timescales (decay rate ``ρ_i`` for each latent). For the
    effect-compartment pattern, set ``decay[target] = k_e0`` to match
    a ``LinearEdge`` with weight ``k_e0`` and obtain
    ``dC_e/dt = k_e0 · (C_p − C_e)``.
    """

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        return accumulator + (-params["decay"] * eta)


class StateDecay(eqx.Module):
    """Single-latent relaxation term ``-decay * eta[target]``."""

    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        return accumulator.at[self.target].add(-params["decay"] * eta[self.target])


class Intercept(eqx.Module):
    """Per-latent constant intercept ``c``."""

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        return accumulator + params["cint"]


class StateIntercept(eqx.Module):
    """Single-latent constant intercept contribution."""

    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        _eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        return accumulator.at[self.target].add(params["cint"])


# ---------------------------------------------------------------------------
# Single-target edges
# ---------------------------------------------------------------------------


class LinearEdge(eqx.Module):
    """``w · η[source]`` contribution at ``target``.

    Also expresses *effect compartments* — set the target's
    ``DiagonalDecay`` rate to the same value as this edge's weight and
    you get ``dC_e/dt = w · (C_p − C_e)``.
    """

    source: int = eqx.field(static=True)
    target: int = eqx.field(static=True)

    def contribute(
        self,
        accumulator: Array,
        _eta: Array,
        eta_per_edge: Array,
        _t: Array,
        params: dict[str, Array],
    ) -> Array:
        contribution = params["weight"] * eta_per_edge[self.target, self.source]
        return accumulator.at[self.target].add(contribution)






