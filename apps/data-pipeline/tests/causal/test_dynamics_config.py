"""Scientific serialization and expression bindings retain the execution semantics."""

from pathlib import Path

import numpy as np
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import StructuralSelection, selected_state_ids
from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_to_dict
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
from tests.dynamics_fixtures import decay_term, hill_term, potential_term
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.contract


def test_description_retains_expression_constants_and_potential_semantics():
    spec = DynamicsSpec(
        2,
        (
            potential_term(
                0,
                center=0.5,
                stiffness=1.2,
                quartic=0.3,
            ),
            hill_term(0, 1, ec50=2.0, n=2.0),
        ),
    )
    description = dynamics_spec_to_dict(spec)
    assert [term["kind"] for term in description["components"]] == ["potential", "drift"]
    assert description["components"][0]["expression"] == spec.components[0].expression.model_dump(
        mode="json"
    )
    assert len(compile_dynamics(spec).site_registry) == 1


def test_scientific_model_roundtrip_preserves_derived_dynamics():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "dynamics_config/scientific_model_roundtrip_preserves_derived_dynamics_model_fixture.json"
        ).read_text()
    )
    restored = ModelSpec.model_validate_json(model.model_dump_json())
    assert dynamics_spec_to_dict(
        compile_model_fixture(model).dynamics.spec
    ) == dynamics_spec_to_dict(compile_model_fixture(restored).dynamics.spec)
    assert any(term.kind == "potential" for _, term in restored.iter_mechanisms())
    np.testing.assert_array_equal(
        selected_state_ids(StructuralSelection(model, None)),
        selected_state_ids(StructuralSelection(restored, None)),
    )


def test_expression_coordinates_must_fit_the_declared_states():
    with pytest.raises(ValueError, match="axes must fit"):
        compile_dynamics(DynamicsSpec(1, (decay_term(1),)))
