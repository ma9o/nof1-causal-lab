"""Operation dependencies and persisted results of one incrementally authored model."""

from __future__ import annotations

import graphlib
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.identity import ArtifactId, OperationId  # noqa: TC001

CreationClass = Literal["deterministic", "batch_llm", "judgment"]


@dataclass(frozen=True)
class Transition:
    operation_id: OperationId
    consumes: tuple[ArtifactId, ...]
    produces: tuple[ArtifactId, ...]
    creation_class: CreationClass
    optional_consumes: tuple[ArtifactId, ...] = ()
    produces_optional: tuple[ArtifactId, ...] = ()
    after: tuple[OperationId, ...] = ()
    writable: bool = False

    @property
    def all_produces(self) -> tuple[ArtifactId, ...]:
        return (*self.produces, *self.produces_optional)


@dataclass(frozen=True)
class Derivation:
    produces: ArtifactId
    from_: Annotated[tuple[ArtifactId, ...], Field(alias="from")]
    optional: bool = False


@dataclass(frozen=True)
class Root:
    artifact_id: ArtifactId
    write_pins: tuple[ArtifactId, ...] = ()


ARTIFACT_GRAPH: tuple[Transition, ...] = (
    Transition("raw_data", (), ("raw_data",), "batch_llm"),
    Transition("latent_structure", ("model",), ("model",), "judgment"),
    Transition(
        "measurement_structure",
        ("raw_data", "model"),
        ("model",),
        "judgment",
        after=("raw_data", "latent_structure"),
    ),
    Transition(
        "measurements",
        ("raw_data", "model"),
        (),
        "batch_llm",
        produces_optional=("panel",),
        after=("measurement_structure",),
    ),
    Transition(
        "statistical_model_spec",
        (
            "model",
            "identification_report",
            "panel",
            "validation_report",
        ),
        ("model",),
        "judgment",
        after=("measurements",),
    ),
    Transition(
        "posterior",
        ("model", "panel"),
        ("model",),
        "deterministic",
        after=("statistical_model_spec",),
    ),
    Transition("simulate", ("model",), (), "deterministic", optional_consumes=("panel",)),
)
DERIVATIONS: tuple[Derivation, ...] = (
    Derivation("identification_report", ("model",)),
    Derivation("data_profile", ("panel",)),
    Derivation("validation_report", ("panel", "model", "data_profile")),
)
ROOTS: tuple[Root, ...] = (Root("model"),)
ROOT_ARTIFACTS = tuple(root.artifact_id for root in ROOTS)
WRITABLE_ARTIFACTS = ROOT_ARTIFACTS + tuple(
    artifact for spec in ARTIFACT_GRAPH if spec.writable for artifact in spec.produces
)


def transition_spec(operation_id: OperationId) -> Transition:
    for spec in ARTIFACT_GRAPH:
        if spec.operation_id == operation_id:
            return spec
    raise KeyError(f"Unknown operation {operation_id!r}")


def topological_derivation_order() -> tuple[Derivation, ...]:
    by_id = {spec.produces: spec for spec in DERIVATIONS}
    graph = {spec.produces: set(spec.from_) & by_id.keys() for spec in DERIVATIONS}
    return tuple(by_id[key] for key in graphlib.TopologicalSorter(graph).static_order())


def topological_transition_order() -> tuple[OperationId, ...]:
    return tuple(
        graphlib.TopologicalSorter(
            {spec.operation_id: set(spec.after) for spec in ARTIFACT_GRAPH}
        ).static_order()
    )


def topological_artifact_order() -> tuple[ArtifactId, ...]:
    from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS

    dependencies = {key: set() for key in ARTIFACT_IDS}
    for spec in ARTIFACT_GRAPH:
        for key in spec.all_produces:
            if key not in ROOT_ARTIFACTS:
                dependencies[key].update(spec.consumes)
    for spec in DERIVATIONS:
        dependencies[spec.produces].update(spec.from_)
    return tuple(graphlib.TopologicalSorter(dependencies).static_order())


def _assert_graph_consistent() -> None:
    from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS

    operations = {spec.operation_id for spec in ARTIFACT_GRAPH}
    assert len(operations) == len(ARTIFACT_GRAPH), "Operation IDs must be unique"
    assert all(set(spec.after) <= operations for spec in ARTIFACT_GRAPH)
    produced = {key for spec in ARTIFACT_GRAPH for key in spec.all_produces}
    derived = {spec.produces for spec in DERIVATIONS}
    assert not produced & derived, "Derived artifacts cannot have operation producers"
    assert produced | derived | set(ROOT_ARTIFACTS) == set(ARTIFACT_IDS)
    topological_transition_order()
    topological_artifact_order()
    topological_derivation_order()


_assert_graph_consistent()
