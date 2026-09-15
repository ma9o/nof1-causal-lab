"""Gradual construct admission: build a model one construct at a time, gated by
the reachability battery on the cumulative partial model.

Constructs are admitted along the causal DAG's topological order (parents before
children). Each admission bundles the construct's contribution to the model —
its self-dynamics parameters, its incoming edges (from already-admitted
parents), and its emission(s) — into the growing :class:`~nof1_causal_lab.
artifacts.ModelSpec`, compiles the *cumulative partial* model, runs the **exact**
prior predictive (Diffrax over the true nonlinear drift, real emission
families), and feeds the resulting arrays to the reachability battery
(:mod:`nof1_causal_lab.models.ssm.reachability`).

Nothing here linearizes: the partial model is compiled and simulated through the
same exact engine the fit uses (``sample_prior_predictive_from_runtime``). A
partial sub-DAG compiles fine as long as every retained estimation state keeps
measurement support and the cumulative loading matrix can reach full column
rank.

The verdict (admit / revise / accept) comes from :func:`reachability.
stage_outcome`; there is no status enum stored on any artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import networkx as nx

from nof1_causal_lab.artifacts.construct import (
    CausalEdgeSpec,
    replace_constructs,
)
from nof1_causal_lab.artifacts.identity import DistributionId  # noqa: TC001
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec  # noqa: TC001
from nof1_causal_lab.models.model_structure import model_for_constructs
from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model
from nof1_causal_lab.models.ssm.parameterization import build_prior_runtime_bundle
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    sample_prior_predictive_from_runtime,
)
from nof1_causal_lab.models.ssm.reachability import (
    CheckResult,
    stage_outcome,
)
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    DesignInfo,
    MeasurementTiming,
    _elapsed_ms,
    measure_construct_simulation,
)
from nof1_causal_lab.numpyro_json import NumPyroDistribution  # noqa: TC001
from nof1_causal_lab.utils.model_structure import (
    get_edges,
    get_state_names,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

# --------------------------------------------------------------------------- #
# Proposal / accumulation types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConstructContribution(ConstructSimulationTarget):
    """A bounded authoring update containing canonical scientific entities."""

    edges: tuple[CausalEdgeSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    distributions: dict[DistributionId, NumPyroDistribution] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionState:
    """One candidate ModelSpec plus the progress of its admission operation."""

    model: ModelSpec
    names: tuple[str, ...] = ()
    annotations: tuple[str, ...] = ()

    def completed_model(self, *, restrict: bool = False) -> ModelSpec:
        from nof1_causal_lab.models.prior_planning import complete_model

        model = model_for_constructs(self.model, set(self.names)) if restrict else self.model
        return complete_model(model)


@dataclass(frozen=True)
class ConstructAdmissionReport:
    """Result of attempting to admit one construct."""

    name: str
    results: tuple[CheckResult, ...]
    timings: tuple[MeasurementTiming, ...]
    outcome: str
    annotations: tuple[str, ...]
    admitted: bool


@dataclass(frozen=True)
class ConstructAdmissionUnit:
    """One schedulable component of the construct graph.

    Singleton components may run in parallel once their predecessor components
    have been admitted.  Members of a lagged-feedback component stay in
    ``state_order`` and are admitted sequentially inside the component.
    """

    unit_id: str
    constructs: tuple[str, ...]
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class FullAdmissionValidation:
    """One exact shared simulation plus per-construct full-model reports."""

    reports: tuple[ConstructAdmissionReport, ...]
    timings: tuple[MeasurementTiming, ...]


# --------------------------------------------------------------------------- #
# Planning + model restriction
# --------------------------------------------------------------------------- #


def build_construct_units(model: ModelSpec) -> list[ConstructAdmissionUnit]:
    """Build the SCC-condensed fork/join plan for construct admission.

    The universe is the model's derived ``state_order`` — constructs
    measurement-structure marginalized, anchored, or dropped out of estimation carry no
    state, so there is nothing to admit for them (restricting the plan to one
    would fail compilation with an empty state_order).

    Ties (independent roots) break by the state_order position for
    determinism. Time-invariant confounders, being edge sources, naturally sort
    first. Lagged feedback loops are legal latent structure (the latent-structure
    validator only forbids *contemporaneous* cycles), so the sort runs on the
    condensation: members of a feedback cycle are admitted back-to-back in
    state_order, and model_for_constructs defers the closing edge until
    the whole cycle is admitted.
    """
    constructs = get_state_names(model)
    order_index = {name: i for i, name in enumerate(constructs)}
    graph = nx.DiGraph()
    graph.add_nodes_from(constructs)
    for edge in get_edges(model):
        cause = edge["cause"]
        effect = edge["effect"]
        if cause in order_index and effect in order_index:
            graph.add_edge(cause, effect)
    condensation = nx.condensation(graph)
    component_members = {
        node: tuple(
            sorted(
                (str(member) for member in data["members"]),
                key=lambda member: order_index[member],
            )
        )
        for node, data in condensation.nodes(data=True)
    }
    scc_index = {node: order_index[members[0]] for node, members in component_members.items()}
    component_order = list(
        nx.lexicographical_topological_sort(
            condensation,
            key=lambda node: scc_index[node],
        )
    )
    unit_id_by_component = {
        node: (
            component_members[node][0]
            if len(component_members[node]) == 1
            else f"feedback:{component_members[node][0]}"
        )
        for node in component_order
    }
    return [
        ConstructAdmissionUnit(
            unit_id=unit_id_by_component[node],
            constructs=component_members[node],
            predecessors=tuple(
                unit_id_by_component[parent]
                for parent in sorted(
                    condensation.predecessors(node),
                    key=lambda component: scc_index[component],
                )
            ),
        )
        for node in component_order
    ]


def build_construct_order(model: ModelSpec) -> list[str]:
    """Deterministic flattened order used for assembly and stable presentation."""
    return [construct for unit in build_construct_units(model) for construct in unit.constructs]


# --------------------------------------------------------------------------- #
# Cumulative-model compilation + exact prior predictive
# --------------------------------------------------------------------------- #


def _compile_partial(
    state: AdmissionState,
) -> tuple[ModelSpec, dict[str, dist.Distribution]]:
    """Compile the cumulative partial model to an ModelSpec + prior registry."""
    spec = state.completed_model(restrict=True)
    registry, _bindings, _diagnostics, _edge_lag, _auxiliary = compile_ssm_inputs_from_model(spec)
    return spec, registry


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


def trial_admission_state(
    state: AdmissionState, contribution: ConstructContribution
) -> AdmissionState:
    """The cumulative state that *would* result from admitting ``contribution``.

    Used both to run the battery (:func:`admit_construct`) and to derive the
    design (grid + observed data) against the same partial model.
    """
    from nof1_causal_lab.models.model_parameters import referenced_parameter_ids

    edges = {edge.id: edge for edge in state.model.edges}
    edges.update((edge.id, edge) for edge in contribution.edges)
    constructs = tuple(
        contribution.construct if item.id == contribution.construct.id else item
        for item in state.model.constructs
    )
    parameters = {parameter.id: parameter for parameter in state.model.parameters}
    parameters.update((parameter.id, parameter) for parameter in contribution.parameters)
    candidate_edges = replace_constructs(edges.values(), constructs)
    referenced = referenced_parameter_ids(*constructs, *candidate_edges)
    extra = {parameter.id for parameter in contribution.parameters} - referenced_parameter_ids(
        contribution.construct, *contribution.edges
    )
    if extra:
        raise ValueError(f"Parameters are not referenced by model components: {sorted(extra)}")
    retained_parameters = tuple(
        parameter for parameter in parameters.values() if parameter.id in referenced
    )
    law_references = {item.distribution for item in (*constructs, *retained_parameters)}
    candidate = state.model.revised(
        edges=candidate_edges,
        parameters=retained_parameters,
        distributions={
            identity: law
            for identity, law in state.model.distributions.items()
            if identity in law_references
        }
        | contribution.distributions,
    )
    return AdmissionState(
        model=candidate,
        names=tuple(dict.fromkeys((*state.names, contribution.name))),
        annotations=state.annotations,
    )


def admit_construct(
    state: AdmissionState,
    contribution: ConstructContribution,
    design: DesignInfo,
    accepted: Mapping[tuple[str, str], str] | None = None,
) -> tuple[AdmissionState, ConstructAdmissionReport]:
    """Attempt to admit one construct; run the battery on the cumulative model."""
    trial = trial_admission_state(state, contribution)

    spec, pred, timings = _compile_and_sample_admission_state(
        trial,
        design,
        compilation_label="ModelSpec compilation",
        predictive_label="Exact prior-predictive simulation",
    )

    results, diagnostic_timings = measure_construct_simulation(spec, pred, design, contribution)
    timings.extend(diagnostic_timings)

    started = perf_counter_ns()
    outcome, annotations = stage_outcome(results, accepted or {})
    timings.append(
        MeasurementTiming(
            phase="admission_decision",
            label="Admission decision",
            duration_ms=_elapsed_ms(started),
        )
    )
    admitted = outcome.startswith("ADMITTED")
    report = ConstructAdmissionReport(
        name=contribution.name,
        results=tuple(results),
        timings=tuple(timings),
        outcome=outcome,
        annotations=annotations,
        admitted=admitted,
    )
    if not admitted:
        return state, report
    return replace(trial, annotations=(*state.annotations, *annotations)), report


def _sample_partial(
    spec: ModelSpec,
    registry: dict[str, dist.Distribution],
    design: DesignInfo,
) -> dict[str, jnp.ndarray]:
    """Exact prior-predictive draws for a compiled partial model on the design grid."""
    bundle = build_prior_runtime_bundle(spec, registry)
    return sample_prior_predictive_from_runtime(
        spec,
        bundle,
        design.t_grid,
        observation_support=design.observation_support,
        num_samples=design.n_draws,
        seed=design.seed,
    )


def _compile_and_sample_admission_state(
    state: AdmissionState,
    design: DesignInfo,
    *,
    compilation_label: str,
    predictive_label: str,
) -> tuple[ModelSpec, dict[str, jnp.ndarray], list[MeasurementTiming]]:
    started = perf_counter_ns()
    spec, registry = _compile_partial(state)
    timings = [
        MeasurementTiming(
            phase="model_compilation",
            label=compilation_label,
            duration_ms=_elapsed_ms(started),
        )
    ]

    started = perf_counter_ns()
    pred = _sample_partial(spec, registry, design)
    jax.block_until_ready(pred)
    timings.append(
        MeasurementTiming(
            phase="prior_predictive",
            label=predictive_label,
            duration_ms=_elapsed_ms(started),
        )
    )
    return spec, pred, timings


def recheck_member(
    state: AdmissionState,
    target: ConstructContribution,
    design: DesignInfo,
) -> tuple[tuple[CheckResult, ...], tuple[MeasurementTiming, ...]]:
    """Re-run the battery on an already-admitted member against the closed-loop model.

    ``state`` is the cumulative state *after* a feedback loop closed (it already contains
    ``target`` and the closing edge), and ``target`` carries the member's closed-loop edge
    set (``edge_parents`` now include the feedback source). Informational: the caller
    surfaces the results as a coupled recheck; they do not gate the admission.
    """
    spec, pred, timings = _compile_and_sample_admission_state(
        state,
        design,
        compilation_label="ModelSpec compilation",
        predictive_label="Exact prior-predictive simulation",
    )
    results, diagnostic_timings = measure_construct_simulation(spec, pred, design, target)
    timings.extend(diagnostic_timings)
    return tuple(results), tuple(timings)


def validate_full_admission_state(
    state: AdmissionState,
    targets: tuple[ConstructContribution, ...],
    design: DesignInfo,
    accepted: Mapping[str, Mapping[tuple[str, str], str]] | None = None,
) -> FullAdmissionValidation:
    """Gate publication with one exact full-model simulation and all construct batteries."""
    spec, pred, timings = _compile_and_sample_admission_state(
        state,
        design,
        compilation_label="Full-model compilation",
        predictive_label="Exact full-model prior-predictive simulation",
    )

    accepted = accepted or {}
    reports: list[ConstructAdmissionReport] = []
    for target in targets:
        results, diagnostic_timings = measure_construct_simulation(spec, pred, design, target)
        outcome, annotations = stage_outcome(results, accepted.get(target.name, {}))
        reports.append(
            ConstructAdmissionReport(
                name=target.name,
                results=tuple(results),
                timings=tuple(diagnostic_timings),
                outcome=outcome,
                annotations=annotations,
                admitted=outcome.startswith("ADMITTED"),
            )
        )
    return FullAdmissionValidation(reports=tuple(reports), timings=tuple(timings))
