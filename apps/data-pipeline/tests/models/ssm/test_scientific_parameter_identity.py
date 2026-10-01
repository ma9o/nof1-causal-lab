"""Stable scientific subjects through authoring, compilation, and retained results."""

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.likelihood import ObservationLawSpec
from pathlib import Path

import numpyro.distributions as dist
import pytest

from nof1_causal_lab.actions.inference.subjects import reference_posterior_findings
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, MechanismRef
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.models.model_checks import check_execution
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from tests.helpers import make_model
from tests.model_fixtures import compile_fit_fixture






@pytest.mark.contract
def test_rename_preserves_parameter_and_element_identity():
    old_model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/rename_preserves_parameter_and_element_identity__compile.json').read_text())
    new_model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/rename_preserves_parameter_and_element_identity__compile_2.json').read_text())
    assert {b.parameter_id: set(b.elements) for b in parameter_bindings(old_model)[0]} == {
        b.parameter_id: set(b.elements) for b in parameter_bindings(new_model)[0]
    }
    assert {p.name for p in old_model.parameters} != {p.name for p in new_model.parameters}


@pytest.mark.contract
def test_scalar_identity_survives_reordered_execution_axes():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/scalar_identity_survives_reordered_execution_axes__compile.json').read_text())
    reordered = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/scalar_identity_survives_reordered_execution_axes__compile_2.json').read_text())
    assert model.state_order == tuple(reversed(reordered.state_order))
    decay = next(
        p
        for p in model.parameters
        if model.parameter_context(p.id).quantity == SiteKind.DYNAMICS_DECAY
    )
    old_binding = next(b for b in parameter_bindings(model)[0] if b.parameter_id == decay.id)
    new_binding = next(b for b in parameter_bindings(reordered)[0] if b.parameter_id == decay.id)
    assert old_binding.elements == new_binding.elements
    assert old_binding.coordinates != new_binding.coordinates


@pytest.mark.contract
def test_model_rejects_forged_owner_before_compilation():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/model_rejects_forged_owner_before_compilation__compile.json').read_text())
    payload = model.model_dump(mode="json")
    payload["parameters"][0]["owners"] = [{"kind": "construct", "id": "construct:forged"}]
    with pytest.raises(ValueError, match=r"Extra inputs|owner"):
        type(model).model_validate(payload)


@pytest.mark.contract
def test_posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate__compile.json').read_text())
    binding = parameter_bindings(model)[0][0]
    element, coordinate = next(iter(binding.coordinates.items()))
    row = {
        "parameter": "display only",
        "coordinate": coordinate.model_dump(mode="json"),
        "interval_kind": "hdi",
        "interval_mass": 0.94,
        "mean": 0.0,
        "sd": 1.0,
        "lower": -1.0,
        "upper": 1.0,
        "x_values": [],
        "density": [],
    }
    marginals, _ = reference_posterior_findings(compile_fit_fixture(model), [row], [])
    assert marginals[0]["subject"] == {"parameter_id": binding.parameter_id, "element_id": element}
    assert "coordinate" not in marginals[0]
    row["coordinate"] = {"site_name": "unknown", "indices": []}
    with pytest.raises(ValueError, match="unbound runtime coordinate"):
        reference_posterior_findings(compile_fit_fixture(model), [row], [])


@pytest.mark.contract
def test_ordinal_components_have_label_identity_and_padding_is_explicit():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/ordinal_components_have_label_identity_and_padding_is_explicit__compile.json').read_text())
    gaps = {
        p.id
        for p in model.parameters
        if model.parameter_context(p.id).quantity == SiteKind.OBS_ORDERED_GAPS
    }
    gap_bindings = [b for b in parameter_bindings(model)[0] if b.parameter_id in gaps]
    assert len(gap_bindings) == 1
    assert list(gap_bindings[0].elements.values()) == ["A_obs: gap low / mid / high"]
    assert parameter_bindings(model)[1]


@pytest.mark.contract
def test_shared_likelihood_parameter_owns_only_active_channels():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels__compile.json').read_text())
    first, second = model.constructs
    student = LikelihoodSpec(
        law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law.json').read_text()),
        reasoning="Test tails",
        standardized=True,
    )
    first = type(first).model_validate(
        {
            **first.model_dump(),
            "indicators": (
                type(first.indicators[0]).model_validate(
                    {**first.indicators[0].model_dump(), "likelihood": student}
                ),
            ),
        }
    )
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_complete_model.json').read_text())
    shared = next(
        p for p in model.parameters if model.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert {o.id for o in model.parameter_context(shared.id).owners} == {
        first.id,
        first.indicators[0].id,
    }
    second = type(second).model_validate(
        {
            **second.model_dump(),
            "indicators": (
                type(second.indicators[0]).model_validate(
                    {
                        **second.indicators[0].model_dump(),
                        "likelihood": LikelihoodSpec(
                            law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law_2.json').read_text()),
                            reasoning="Test tails",
                            standardized=True,
                        ),
                    }
                ),
            ),
        }
    )
    expanded = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_complete_model_2.json').read_text())
    newer = next(
        p
        for p in expanded.parameters
        if expanded.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert newer.id == shared.id
    assert newer.distribution is shared.distribution
    assert {o.id for o in expanded.parameter_context(newer.id).owners} == {
        first.id,
        second.id,
        first.indicators[0].id,
        second.indicators[0].id,
    }
    expanded.check_execution()


@pytest.mark.contract
def test_student_innovation_tail_is_explicit_and_shared_through_completion():

    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/student_innovation_tail_is_explicit_and_shared_through_completion_complete_model.json').read_text())
    parameter = next(p for p in model.parameters if p.name == "proc_df")
    for construct in model.constructs:
        assert parameter in model.parameters_for(construct.id)
    model.check_execution()
    first, second = model.constructs
    candidate = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                type(first).model_validate(
                    {
                        **first.model_dump(),
                        "coefficients": tuple(
                            operand
                            for operand in first.coefficients
                            if operand.role != "process_degrees_of_freedom"
                        ),
                    }
                ),
                second,
            ),
        )
    )
    with pytest.raises(ValueError, match="degrees_of_freedom requires a prior parameter"):
        candidate.check_execution()
    completed = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/student_innovation_tail_is_explicit_and_shared_through_completion_complete_model_2.json').read_text())
    assert completed.parameter(parameter.id) == parameter
    assert completed.get_construct(first.id).coefficients == first.coefficients


@pytest.mark.contract
@pytest.mark.parametrize(('retained_role', '_compile_payload', 'complete_model_payload'), [
    pytest.param(None, 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation__compile_none.json', 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_none.json', id='None'),
    pytest.param('initial_mean', 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation__compile_initial_mean.json', 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_initial_mean.json', id='initial_mean'),
    pytest.param('initial_scale', 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation__compile_initial_scale.json', 'scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_initial_scale.json', id='initial_scale'),
])
def test_initial_state_defaults_are_authored_before_compilation(retained_role, _compile_payload, complete_model_payload):
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / _compile_payload).read_text())

    free = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                type(c).model_validate(
                    {
                        **c.model_dump(),
                        "coefficients": tuple(
                            operand
                            for operand in c.coefficients
                            if not operand.role.startswith("initial_")
                            or operand.role == retained_role
                        ),
                    }
                )
                for c in model.constructs
            ),
        )
    )
    with pytest.raises(ValueError, match="initial-state coefficients"):
        free.check_execution()
    completed = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / complete_model_payload).read_text())
    before = completed.model_dump(mode="json")
    completed.check_execution()
    initial = [
        p
        for p in completed.parameters
        if completed.parameter_context(p.id).quantity in {SiteKind.T0_MEANS, SiteKind.T0_VAR_DIAG}
    ]
    assert initial
    assert all(p.distribution is not None for p in initial)
    assert {p.id for p in initial} <= {b.parameter_id for b in parameter_bindings(completed)[0]}
    assert completed.model_dump(mode="json") == before
    assert all(
        "elements" not in p and "role" not in p and "constraint" not in p
        for p in before["parameters"]
    )


@pytest.mark.inference(concern="sampling")
def test_parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws__compile.json').read_text())
    renamed = model.revised(
        parameters=tuple(
            type(p).model_validate({**p.model_dump(), "name": f"display {n}"})
            for n, p in enumerate(model.parameters)
        )
    )
    check_execution(renamed)
    _assert_same_prior_laws(model, renamed)
    assert {b.parameter_id: b.coordinates for b in parameter_bindings(model)[0]} == {
        b.parameter_id: b.coordinates for b in parameter_bindings(renamed)[0]
    }


@pytest.mark.inference(concern="sampling")
def test_additive_hill_and_linear_terms_survive_parameter_renaming():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/additive_hill_and_linear_terms_survive_parameter_renaming__compile.json').read_text())

    additive = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'scientific_parameter_identity/additive_hill_and_linear_terms_survive_parameter_renaming_with_parameter_distributions.json').read_text())
    check_execution(additive)
    renamed = additive.revised(
        parameters=tuple(
            type(p).model_validate({**p.model_dump(), "name": f"opaque {n}"})
            for n, p in enumerate(additive.parameters)
        )
    )
    check_execution(renamed)
    _assert_same_prior_laws(additive, renamed)
    components = numeric.dynamics_components(additive).components
    original_components = numeric.dynamics_components(model).components
    assert isinstance(components, tuple)
    assert isinstance(original_components, tuple)
    assert len(components) == len(original_components) + 1


def _assert_same_prior_laws(first, second):
    import numpy as np

    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

    first_laws = compile_priors(first)[0]
    second_laws = compile_priors(second)[0]
    assert first_laws.keys() == second_laws.keys()
    for name, law in first_laws.items():
        for value in (0.15, 0.5, 1.25):
            np.testing.assert_allclose(law.log_prob(value), second_laws[name].log_prob(value))
