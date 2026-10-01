"""Notebook model assembly and exact full-model checks, without admission state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scripts.fixtures.study import referenced_parameter_ids

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    measure_construct_simulation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
    from nof1_causal_lab.artifacts.identity import DistributionId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.models.ssm.reachability import CheckResult
    from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo
    from nof1_causal_lab.numpyro_json import NumPyroDistribution


@dataclass(frozen=True)
class ConstructEditSpec(ConstructSimulationTarget):
    """Notebook-authored definitions to merge into one whole candidate model."""

    edges: tuple[CausalEdgeSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    distributions: dict[DistributionId, NumPyroDistribution] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructPredictiveReport:
    name: str
    results: tuple[CheckResult, ...]


def evaluate_case_study(
    model: ModelSpec,
    edits: Sequence[ConstructEditSpec],
    design: DesignInfo,
) -> tuple[ModelSpec, dict[str, ConstructPredictiveReport]]:
    """Assemble the full authored model, generate once, and measure every retained state."""
    constructs = {item.id: item for item in model.constructs}
    edges = {edge.id: edge for edge in model.edges}
    parameters = {parameter.id: parameter for parameter in model.parameters}
    laws = dict(model.distributions)
    for edit in edits:
        constructs[edit.construct.id] = edit.construct
        edges.update((edge.id, edge) for edge in edit.edges)
        parameters.update((parameter.id, parameter) for parameter in edit.parameters)
        laws.update(edit.distributions)
    referenced = referenced_parameter_ids(*constructs.values(), *edges.values())
    parameters = {key: parameter for key, parameter in parameters.items() if key in referenced}
    used_laws = {item.distribution for item in (*constructs.values(), *parameters.values())}
    candidate = model.revised(
        edges=replace_constructs(tuple(edges.values()), tuple(constructs.values())),
        parameters=tuple(parameters.values()),
        distributions={key: law for key, law in laws.items() if key in used_laws},
    )
    batch = generate_simulation_batch(
        candidate,
        SimulationSpec(start=float(design.t_grid[0]), end=float(design.t_grid[-1])),
        time_origin=None,
        times=design.t_grid,
        draws=design.n_draws,
        seed=design.seed,
    )
    by_name = {edit.name: edit for edit in edits}
    reports = {}
    for name in numeric.state_names(candidate):
        findings, _ = measure_construct_simulation(
            candidate, batch.prediction, design, by_name[name]
        )
        reports[name] = ConstructPredictiveReport(name, tuple(findings))
    return candidate, reports


def model_with_prior_payloads(model: ModelSpec, payloads: Mapping[str, JsonObject]) -> ModelSpec:
    """Resolve notebook prior submissions into model-owned laws and parameter references."""
    from notebooks.model_authoring import with_parameter_distributions

    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.distributions import PriorDistributionFamily
    from nof1_causal_lab.prior_distributions import distribution_from_params

    by_id = {parameter.id: parameter for parameter in model.parameters}
    unknown = payloads.keys() - by_id.keys()
    if unknown:
        raise ValueError(f"Prior input does not correspond to any parameter: {sorted(unknown)}")
    laws = {}
    parameters = []
    for parameter in model.parameters:
        if parameter.id not in payloads:
            parameters.append(parameter)
            continue
        payload = payloads[parameter.id]
        supplied_id = payload.get("parameter_id")
        if supplied_id is not None and supplied_id != parameter.id:
            raise ValueError(f"Prior for {parameter.name!r} references a different parameter")
        params = payload["params"]
        if not isinstance(params, dict):
            raise ValueError("Prior constructor params must be a JSON object")
        laws[parameter.id] = distribution_from_params(
            PriorDistributionFamily(payload["distribution"]), params
        )
        parameters.append(
            ParameterSpec.model_validate(
                {
                    **parameter.model_dump(mode="python"),
                    "reference_interval_days": payload.get("reference_interval_days"),
                }
            )
        )
    return with_parameter_distributions(model.revised(parameters=tuple(parameters)), laws)
