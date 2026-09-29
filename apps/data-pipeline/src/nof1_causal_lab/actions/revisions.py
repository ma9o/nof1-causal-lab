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
from nof1_causal_lab.json_types import JsonValue  # noqa: TC001
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


class ModelDefinitionChange(BaseModel):
    """One changed field in identity-keyed scientific model definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    change: Literal["added", "removed", "revised"]
    before: JsonValue
    after: JsonValue


class ModelDiffReport(BaseModel):
    """A model diff joins definition changes and evidence at two model revisions or checkpoints."""

    model_config = ConfigDict(extra="forbid")
    before: GitRef
    after: GitRef
    definition_changes: list[ModelDefinitionChange]
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
    old_laws = left.model_dump(mode="json")["distributions"]
    new_laws = right.model_dump(mode="json")["distributions"]
    changes = []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity), new.get(identity)
        if a == b and (
            a is None
            or a.distribution is None
            or json.dumps(old_laws[a.distribution], sort_keys=True)
            == json.dumps(new_laws[a.distribution], sort_keys=True)
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


def compare_model_definitions(left: ModelSpec, right: ModelSpec) -> list[ModelDefinitionChange]:
    """Compare every authored field, aligning entities by ID rather than list position."""

    def definition(model: ModelSpec) -> JsonValue:
        value = model.model_dump(mode="json", exclude={"edges", "parameters"})
        value["constructs"] = {item.id: item.model_dump(mode="json") for item in model.constructs}
        value["parameters"] = {item.id: item.model_dump(mode="json") for item in model.parameters}
        value["edges"] = {
            item.id: {
                **item.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause": item.cause.id,
                "effect": item.effect.id,
            }
            for item in model.edges
        }
        for construct in value["constructs"].values():
            construct["indicators"] = {item["id"]: item for item in construct["indicators"]}
        return value

    changes = []

    def walk(before: JsonValue, after: JsonValue, path: str) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in before or key not in after:
                    changes.append(
                        ModelDefinitionChange(
                            path=pointer,
                            change="added" if key in after else "removed",
                            before=before.get(key),
                            after=after.get(key),
                        )
                    )
                else:
                    walk(before[key], after[key], pointer)
        elif before != after:
            changes.append(
                ModelDefinitionChange(path=path, change="revised", before=before, after=after)
            )

    walk(definition(left), definition(right), "")
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


def _model_revision(
    workspace_id: str, revision: GitOid
) -> tuple[ModelSpec, GitRef, InferenceReport | None, SimulationReport | None]:
    """Select an exact model tree or the model and recorded evidence at a Git commit."""
    import pygit2

    from nof1_causal_lab.machine.snapshots import ModelReader, SnapshotRevisionNotFound
    from nof1_causal_lab.machine.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    try:
        obj = store.repo[pygit2.Oid(hex=revision)]
    except KeyError as exc:
        raise SnapshotRevisionNotFound(f"Unknown model revision {revision}") from exc
    if isinstance(obj, pygit2.Tree):
        try:
            model = read_model(store, revision)
        except (KeyError, ValueError) as exc:
            raise SnapshotRevisionNotFound("The selected tree is not a model artifact") from exc
        return (
            model,
            GitRef(workspace_id=workspace_id, revision=revision, path="model.json"),
            None,
            None,
        )
    if not isinstance(obj, pygit2.Commit):
        raise SnapshotRevisionNotFound("Select a model artifact tree or a study commit")
    reader = ModelReader(workspace_id, at=revision)
    if reader.model is None:
        raise SnapshotRevisionNotFound("The selected checkpoint contains no model")
    fit, simulation = reader.inference_report, reader.simulation()
    return (
        reader.model,
        GitRef(workspace_id=workspace_id, revision=revision, path="artifacts/model/model.json"),
        fit.value if fit is not None and fit.source.validity == "fresh" else None,
        simulation.value
        if simulation is not None and simulation.source.validity == "fresh"
        else None,
    )


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffReport:
    """Inspect scientific definition changes and evidence without fitting or simulation."""
    from nof1_causal_lab.actions.checks import check_specification
    from nof1_causal_lab.models.model_inputs import input_fingerprints

    left, before, before_fit, before_simulation = _model_revision(workspace_id, before_id)
    right, after, after_fit, after_simulation = _model_revision(workspace_id, after_id)
    changes = compare_parameters(left, right)
    fingerprints = input_fingerprints(left)
    return ModelDiffReport(
        before=before,
        after=after,
        definition_changes=compare_model_definitions(left, right),
        parameters=changes,
        graph=compare_model_graph(left, right, changes),
        before_checks=check_specification(left),
        after_checks=check_specification(right),
        before_fit=before_fit,
        after_fit=after_fit,
        before_simulation=before_simulation,
        after_simulation=after_simulation,
        changed_inputs=[
            key for key, value in input_fingerprints(right).items() if value != fingerprints[key]
        ],
    )
