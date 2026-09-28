"""Operation ordering is distinct from scientific artifact identity."""

import pytest

from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    DERIVATIONS,
    ROOT_ARTIFACTS,
    topological_artifact_order,
    topological_transition_order,
    transition_spec,
)

pytestmark = pytest.mark.contract


def test_model_is_authored_directly_and_conditioned_by_fitting():
    assert set(ROOT_ARTIFACTS) == {"model"}
    writers = [s.operation_id for s in ARTIFACT_GRAPH if "model" in s.produces]
    assert writers == ["posterior"]
    assert len({s.operation_id for s in ARTIFACT_GRAPH}) == len(ARTIFACT_GRAPH)
    assert set(topological_artifact_order()) == set(ARTIFACT_IDS)
    assert set(topological_transition_order()) == {s.operation_id for s in ARTIFACT_GRAPH}
    assert "baseline_report" not in ARTIFACT_IDS


def test_topological_order_respects_operation_prerequisites():
    positions = {key: i for i, key in enumerate(topological_transition_order())}
    for operation in ARTIFACT_GRAPH:
        assert all(
            positions[parent] < positions[operation.operation_id] for parent in operation.after
        )


def test_derived_outputs_depend_on_the_canonical_definition():
    parents = {d.produces: d.from_ for d in DERIVATIONS}
    assert parents == {
        "identification_report": ("model",),
        "data_profile": ("panel",),
        "validation_report": ("panel", "model", "data_profile"),
    }
    assert not {output for s in ARTIFACT_GRAPH for output in s.all_produces} & parents.keys()


def test_scientific_gates_are_declared():
    assert set(transition_spec("posterior").consumes) == {"model", "panel"}
    assert "panel" in transition_spec("posterior").consumes
    assert transition_spec("measurements").produces_optional == ("panel",)


@pytest.mark.parametrize(
    "key", ["model", "identification_report", "validation_report", "baseline_report"]
)
def test_artifacts_are_not_operation_names(key):
    with pytest.raises(KeyError, match="Unknown operation"):
        transition_spec(key)
