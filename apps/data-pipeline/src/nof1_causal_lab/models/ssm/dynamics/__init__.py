"""Continuous-time vector-field dynamics for SSMs.

This package owns the dynamics vocabulary:
vector-field components, dynamics specs, interventions, stability checks,
simulation, and dynamics-spec serialization.

Block-level SSM parameter structure lives in ``ssm.structure``.
CT-to-DT matrix discretization lives in ``ssm.discretization``.
Prior/predictive validation lives in ``ssm.predictive``.
"""

from __future__ import annotations

from .draws import DynamicsDraws, dynamics_from_samples
from .edges import (
    DiagonalDecay,
    Intercept,
    LinearEdge,
    StateDecay,
    StateIntercept,
    VectorFieldComponent,
)
from .intervention import (
    ConstantValueFn,
    EdgeInputOverride,
    Intervention,
    LinearRampValueFn,
    Override,
    PrecomputedValueFn,
    ValueFn,
    VariableOverride,
)
from .linearisation import Linearisation, infer_linearisation
from .serialization import (
    dynamics_spec_to_dict,
)
from .simulator import (
    BrownianTreeSpec,
    IndexedBrownianSpec,
    ProcessNoise,
    SimulationConfig,
    simulate,
)
from .spec import (
    CompiledDynamics,
    DynamicsSpec,
    compile_dynamics,
    iter_dynamics_semantic_bindings,
)
from .vector_field import VectorField, VectorFieldArgs

__all__ = [
    "BrownianTreeSpec",
    "CompiledDynamics",
    "DynamicsSpec",
    "DiagonalDecay",
    "VectorFieldComponent",
    "ConstantValueFn",
    "EdgeInputOverride",
    "LinearRampValueFn",
    "Intercept",
    "Intervention",
    "IndexedBrownianSpec",
    "Linearisation",
    "LinearEdge",
    "Override",
    "DynamicsDraws",
    "PrecomputedValueFn",
    "ProcessNoise",
    "SimulationConfig",
    "StateDecay",
    "StateIntercept",
    "ValueFn",
    "VariableOverride",
    "VectorField",
    "VectorFieldArgs",
    "compile_dynamics",
    "dynamics_spec_to_dict",
    "infer_linearisation",
    "iter_dynamics_semantic_bindings",
    "dynamics_from_samples",
    "simulate",
]
