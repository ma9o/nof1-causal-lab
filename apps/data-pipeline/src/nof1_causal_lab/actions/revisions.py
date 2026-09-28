"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.artifacts.checks import SpecificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition  # noqa: TC001
from nof1_causal_lab.artifacts.identity import (
    ConstructId,
    ConstructRef,
    EdgeId,
    GitOid,
    GitRef,
    ParameterId,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import InferenceReport  # noqa: TC001
from nof1_causal_lab.artifacts.simulation import SimulationReport  # noqa: TC001
from nof1_causal_lab.machine.artifacts import ArtifactRecord  # noqa: TC001


class RevisionCatalog(BaseModel):
    """A revision catalog lists immutable model, source and observation inputs for selection."""

    model_config = ConfigDict(extra="forbid")
    models: list[ArtifactRecord]
    raw_data: list[ArtifactRecord]
    panels: list[ArtifactRecord]


class ParameterChange(BaseModel):
    """A parameter change compares one parameter's fixed value or law across model revisions."""

    model_config = ConfigDict(extra="forbid")
    parameter_id: ParameterId
    before: ParameterSpec | None
    after: ParameterSpec | None
    change: str


class ConstructComparison(BaseModel):
    """A construct's definitions and changed owned parameters in two model revisions."""

    model_config = ConfigDict(extra="forbid")
    construct_id: ConstructId
    before: ConstructSpec | None
    after: ConstructSpec | None
    change: Literal["added", "removed", "revised", "unchanged"]
    parameter_ids: list[ParameterId]
    before_disposition: StructuralItemDisposition | None
    after_disposition: StructuralItemDisposition | None


class ComparisonConnection(BaseModel):
    """Endpoint references and temporal relation for one side of a causal edge comparison."""

    model_config = ConfigDict(extra="forbid")
    cause: ConstructRef
    effect: ConstructRef
    lagged: bool
    description: str


class EdgeComparison(BaseModel):
    """An explicit causal edge's definitions and changed mechanism parameters."""

    model_config = ConfigDict(extra="forbid")
    edge_id: EdgeId
    before: ComparisonConnection | None
    after: ComparisonConnection | None
    change: Literal["added", "removed", "revised", "unchanged"]
    parameter_ids: list[ParameterId]
    before_disposition: StructuralItemDisposition | None
    after_disposition: StructuralItemDisposition | None


class ModelGraphComparison(BaseModel):
    """Aligned scientific entities for rendering a graph difference without browser inference."""

    model_config = ConfigDict(extra="forbid")
    constructs: list[ConstructComparison]
    edges: list[EdgeComparison]


class ModelComparison(BaseModel):
    """A comparison joins graph, parameter decisions and evidence at two committed checkpoints."""

    model_config = ConfigDict(extra="forbid")
    before: GitRef
    after: GitRef
    parameters: list[ParameterChange]
    graph: ModelGraphComparison
    changed_inputs: list[str]
    before_checks: SpecificationReport
    after_checks: SpecificationReport
    before_fit: InferenceReport | None
    after_fit: InferenceReport | None
    before_simulation: SimulationReport | None
    after_simulation: SimulationReport | None


def compare_parameters(left: ModelSpec, right: ModelSpec) -> list[ParameterChange]:
    """Compare native parameter decisions and law contents by persistent identity."""
    old, new = {p.id: p for p in left.parameters}, {p.id: p for p in right.parameters}
    changes = []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity), new.get(identity)
        if a == b and (
            a is None
            or a.distribution is None
            or json.dumps(
                left.model_dump(mode="json")["distributions"][a.distribution], sort_keys=True
            )
            == json.dumps(
                right.model_dump(mode="json")["distributions"][a.distribution], sort_keys=True
            )
        ):
            continue
        change = (
            "added"
            if a is None
            else "removed"
            if b is None
            else "released"
            if a.value is not None and b.value is None
            else "pinned"
            if a.value is None and b.value is not None
            else "revised"
        )
        changes.append(ParameterChange(parameter_id=identity, before=a, after=b, change=change))
    return changes


def compare_model_graph(
    left: ModelSpec, right: ModelSpec, parameters: list[ParameterChange]
) -> ModelGraphComparison:
    """Localize definition changes using canonical containment, including shared coefficients."""
    from nof1_causal_lab.models.model_parameters import referenced_parameter_ids
    from nof1_causal_lab.models.model_structure import model_graph_entities

    changed_parameters = {item.parameter_id for item in parameters}
    laws = [model.model_dump(mode="json")["distributions"] for model in (left, right)]
    graphs = [model_graph_entities(model) for model in (left, right)]
    dispositions = [
        {item.target.id: item for item in model.structural_dispositions}
        if model.measurement_clock is not None and model.indicators
        else {}
        for model in (left, right)
    ]

    def definition(entity: ConstructSpec | CausalEdgeSpec | None, side: int):
        if entity is None:
            return None
        if isinstance(entity, CausalEdgeSpec):
            return {
                **entity.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause": entity.cause.id,
                "effect": entity.effect.id,
            }
        model = (left, right)[side]
        return {
            **entity.model_dump(mode="json"),
            "law": laws[side].get(entity.distribution),
            "time_points": model.time_points if entity.distribution is not None else (),
            "default_outcome": model.default_outcome == entity.id,
        }

    def entities(before, after, comparison_type, identity_field):
        old, new = {item.id: item for item in before}, {item.id: item for item in after}
        result = []
        for identity in sorted(old.keys() | new.keys()):
            a, b = old.get(identity), new.get(identity)
            owned = referenced_parameter_ids(*(item for item in (a, b) if item is not None))
            parameter_ids = sorted(owned & changed_parameters)
            change = (
                "added"
                if a is None
                else "removed"
                if b is None
                else "revised"
                if definition(a, 0) != definition(b, 1) or parameter_ids
                else "unchanged"
            )
            result.append(
                comparison_type(
                    **{
                        identity_field: identity,
                        "before": read(a),
                        "after": read(b),
                        "change": change,
                        "parameter_ids": parameter_ids,
                        "before_disposition": dispositions[0].get(identity),
                        "after_disposition": dispositions[1].get(identity),
                    }
                )
            )
        return result

    def read(entity):
        if isinstance(entity, CausalEdgeSpec):
            return ComparisonConnection(
                cause=ConstructRef(id=entity.cause.id),
                effect=ConstructRef(id=entity.effect.id),
                lagged=entity.lagged,
                description=entity.description,
            )
        return entity

    return ModelGraphComparison(
        constructs=entities(graphs[0][0], graphs[1][0], ConstructComparison, "construct_id"),
        edges=entities(graphs[0][1], graphs[1][1], EdgeComparison, "edge_id"),
    )


def compare_checkpoints(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelComparison:
    from nof1_causal_lab.actions.checks import check_specification
    from nof1_causal_lab.machine.snapshots import ModelReader, SnapshotRevisionNotFound
    from nof1_causal_lab.models.model_inputs import input_fingerprints

    before, after = (
        ModelReader(workspace_id, at=before_id),
        ModelReader(workspace_id, at=after_id),
    )
    left, right = before.model, after.model
    if left is None or right is None:
        raise SnapshotRevisionNotFound("Both checkpoints must contain a model")
    fits = [reader.inference_report for reader in (before, after)]
    simulations = [reader.simulation() for reader in (before, after)]
    changes = compare_parameters(left, right)
    fingerprints = input_fingerprints(left)
    return ModelComparison(
        before=GitRef(
            workspace_id=workspace_id, revision=before.commit_id, path="artifacts/model/model.json"
        ),
        after=GitRef(
            workspace_id=workspace_id, revision=after.commit_id, path="artifacts/model/model.json"
        ),
        parameters=changes,
        graph=compare_model_graph(left, right, changes),
        before_checks=check_specification(left),
        after_checks=check_specification(right),
        before_fit=fits[0].value if fits[0] and fits[0].source.validity == "fresh" else None,
        after_fit=fits[1].value if fits[1] and fits[1].source.validity == "fresh" else None,
        before_simulation=simulations[0].value
        if simulations[0] and simulations[0].source.validity == "fresh"
        else None,
        after_simulation=simulations[1].value
        if simulations[1] and simulations[1].source.validity == "fresh"
        else None,
        changed_inputs=[
            key for key, value in input_fingerprints(right).items() if value != fingerprints[key]
        ],
    )
