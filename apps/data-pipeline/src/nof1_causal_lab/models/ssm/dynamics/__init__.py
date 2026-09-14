"""Continuous-time vector-field dynamics for SSMs.

This package owns the dynamics vocabulary:
vector-field components, dynamics specs, interventions, stability checks,
steady states, simulation, and dynamics-spec serialization.

Block-level SSM parameter structure lives in ``ssm.structure``.
CT-to-DT matrix discretization lives in ``ssm.discretization``.
Prior/predictive validation lives in ``ssm.predictive``.
"""

from __future__ import annotations

from .edges import (
    DiagonalDecay,
    Intercept,
    LinearEdge,
    StateDecay,
    StateIntercept,
    VectorFieldComponent,
)
from .intervention import (
    EdgeInputOverride,
    Intervention,
    Override,
    PrecomputedValueFn,
    ValueFn,
    VariableOverride,
    constant_value,
    linear_ramp,
    precomputed_value,
)
from .linearisation import Linearisation, infer_linearisation
from .posterior import (
    PosteriorDynamicsSamples,
    component_param_samples_from_site_samples,
    posterior_dynamics_from_samples,
)
from .serialization import (
    dynamics_spec_to_dict,
)
from .simulator import SimulationConfig, simulate
from .spec import (
    CompiledDynamics,
    DynamicsSpec,
    compile_dynamics,
    iter_dynamics_semantic_bindings,
)
from .steady_state import compute_steady_state
from .vector_field import VectorField, VectorFieldArgs

__all__ = [
    "CompiledDynamics",
    "DynamicsSpec",
    "DiagonalDecay",
    "VectorFieldComponent",
    "EdgeInputOverride",
    "Intercept",
    "Intervention",
    "Linearisation",
    "LinearEdge",
    "Override",
    "PosteriorDynamicsSamples",
    "PrecomputedValueFn",
    "SimulationConfig",
    "StateDecay",
    "StateIntercept",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "compile_dynamics",
    "dynamics_spec_to_dict",
    "component_param_samples_from_site_samples",
    "compute_steady_state",
    "constant_value",
    "infer_linearisation",
    "iter_dynamics_semantic_bindings",
    "linear_ramp",
    "posterior_dynamics_from_samples",
    "precomputed_value",
    "simulate",
]
