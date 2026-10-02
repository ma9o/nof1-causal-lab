"""Stable scientific subjects through authoring, compilation, and retained results."""

from pathlib import Path

import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.actions.inference.subjects import parameter_references
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import (
    LikelihoodSpec,
    ObservationLawSpec,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.model_structure import selected_state_ids
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from tests.model_fixtures import compile_fit_fixture, compile_model_fixture


@pytest.mark.contract
def test_rename_preserves_parameter_and_element_identity():
    old_model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )
    new_model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/rename_preserves_parameter_and_element_identity__compile_2.json"
        ).read_text()
    )
    assert {
        b.parameter_id: set(b.elements)
        for b in parameter_bindings(compile_model_fixture(old_model))[0]
    } == {
        b.parameter_id: set(b.elements)
        for b in parameter_bindings(compile_model_fixture(new_model))[0]
    }
    assert {p.name for p in old_model.parameters} != {p.name for p in new_model.parameters}


@pytest.mark.contract
def test_scalar_identity_survives_reordered_execution_axes():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/scalar_identity_survives_reordered_execution_axes__compile.json"
        ).read_text()
    )
    reordered = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/scalar_identity_survives_reordered_execution_axes__compile_2.json"
        ).read_text()
    )
    assert selected_state_ids(model) == tuple(reversed(selected_state_ids(reordered)))
    decay = next(
        p
        for p in model.parameters
        if model.parameter_context(p.id).quantity == SiteKind.DYNAMICS_DECAY
    )
    old_binding = next(
        b for b in parameter_bindings(compile_model_fixture(model))[0] if b.parameter_id == decay.id
    )
    new_binding = next(
        b
        for b in parameter_bindings(compile_model_fixture(reordered))[0]
        if b.parameter_id == decay.id
    )
    assert old_binding.elements == new_binding.elements
    assert old_binding.coordinates != new_binding.coordinates


@pytest.mark.contract
def test_model_rejects_forged_owner_before_compilation():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )
    payload = model.model_dump(mode="json")
    payload["parameters"][0]["owners"] = [{"kind": "construct", "id": "construct:forged"}]
    with pytest.raises(ValueError, match=r"Extra inputs|owner"):
        type(model).model_validate(payload)


@pytest.mark.contract
def test_posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )
    binding = parameter_bindings(compile_model_fixture(model))[0][0]
    element, coordinate = next(iter(binding.coordinates.items()))
    from nof1_causal_lab.artifacts.identity import ParameterRef
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.models.ssm.inference.diagnostics_viz import compute_posterior_marginals

    references = parameter_references(compile_fit_fixture(model))
    reference = references[coordinate]
    assert reference is not None
    label, subject = reference
    assert subject == ParameterRef(parameter_id=binding.parameter_id, element_id=element)
    # The numerical producer attaches the reference directly; no serialized row is rebound.
    shape = tuple(i + 1 for i in coordinate.indices)
    samples = jnp.zeros((4, *shape))
    marginals = compute_posterior_marginals({coordinate.site_name: samples}, references, 5)
    assert next(row for row in marginals if row.subject == subject).parameter == label
    with pytest.raises(KeyError):
        compute_posterior_marginals({"unknown": jnp.zeros(4)}, references, 5)
    assert ParameterCoordinate(site_name="unknown", indices=()) not in references


@pytest.mark.contract
def test_ordinal_components_have_label_identity_and_padding_is_explicit():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/ordinal_components_have_label_identity_and_padding_is_explicit__compile.json"
        ).read_text()
    )
    gaps = {
        p.id
        for p in model.parameters
        if model.parameter_context(p.id).quantity == SiteKind.OBS_ORDERED_GAPS
    }
    gap_bindings = [
        b for b in parameter_bindings(compile_model_fixture(model))[0] if b.parameter_id in gaps
    ]
    assert len(gap_bindings) == 1
    assert list(gap_bindings[0].elements.values()) == ["A_obs: gap low / mid / high"]
    assert parameter_bindings(compile_model_fixture(model))[1]


@pytest.mark.contract
def test_shared_likelihood_parameter_owns_only_active_channels():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )
    first, second = model.constructs
    student = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law.json"
            ).read_text()
        ),
        reasoning="Test tails",
        standardized=True,
    )
    first = first.revised(indicators=(first.indicators[0].revised(likelihood=student),))
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_complete_model.json"
        ).read_text()
    )
    shared = next(
        p for p in model.parameters if model.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert {o.id for o in model.parameter_context(shared.id).owners} == {
        first.id,
        first.indicators[0].observation.id,
    }
    second = second.revised(
        indicators=(
            second.indicators[0].revised(
                likelihood=LikelihoodSpec(
                    law=TypeAdapter(ObservationLawSpec).validate_json(
                        (
                            Path(__file__).resolve().parents[2]
                            / "fixtures/models"
                            / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law_2.json"
                        ).read_text()
                    ),
                    reasoning="Test tails",
                    standardized=True,
                )
            ),
        )
    )
    expanded = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_complete_model_2.json"
        ).read_text()
    )
    newer = next(
        p
        for p in expanded.parameters
        if expanded.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert newer.id == shared.id
    assert newer.distribution == shared.distribution
    assert {o.id for o in expanded.parameter_context(newer.id).owners} == {
        first.id,
        second.id,
        first.indicators[0].observation.id,
        second.indicators[0].observation.id,
    }
    compile_model_fixture(expanded)


@pytest.mark.contract
def test_student_innovation_tail_is_explicit_and_shared_through_completion():

    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/student_innovation_model.json"
        ).read_text()
    )
    parameter = next(p for p in model.parameters if p.name == "proc_df")
    for construct in model.constructs:
        assert parameter in model.parameters_for(construct.id)
    compile_model_fixture(model)
    first, second = model.constructs
    candidate = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                first.revised(
                    coefficients=tuple(
                        operand
                        for operand in first.coefficients
                        if operand.role != "process_degrees_of_freedom"
                    )
                ),
                second,
            ),
        )
    )
    with pytest.raises(ValueError, match="degrees_of_freedom requires a prior parameter"):
        compile_model_fixture(candidate)
    completed = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/student_innovation_model.json"
        ).read_text()
    )
    assert completed.parameter(parameter.id) == parameter
    assert completed.get_construct(first.id).coefficients == first.coefficients


@pytest.mark.contract
@pytest.mark.parametrize(
    ("retained_role", "_compile_payload", "complete_model_payload"),
    [
        pytest.param(
            None,
            "common/additive_a_b_model.json",
            "scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_none.json",
            id="None",
        ),
        pytest.param(
            "initial_mean",
            "common/additive_a_b_model.json",
            "scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_initial_mean.json",
            id="initial_mean",
        ),
        pytest.param(
            "initial_scale",
            "common/additive_a_b_model.json",
            "scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_initial_scale.json",
            id="initial_scale",
        ),
    ],
)
def test_initial_state_defaults_are_authored_before_compilation(
    retained_role, _compile_payload, complete_model_payload
):
    model = ModelSpec.model_validate_json(
        (Path(__file__).resolve().parents[2] / "fixtures/models" / _compile_payload).read_text()
    )

    free = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.revised(
                    coefficients=tuple(
                        operand
                        for operand in c.coefficients
                        if not operand.role.startswith("initial_") or operand.role == retained_role
                    )
                )
                for c in model.constructs
            ),
        )
    )
    with pytest.raises(ValueError, match="initial-state coefficients"):
        compile_model_fixture(free)
    completed = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2] / "fixtures/models" / complete_model_payload
        ).read_text()
    )
    before = completed.model_dump(mode="json")
    compile_model_fixture(completed)
    initial = [
        p
        for p in completed.parameters
        if completed.parameter_context(p.id).quantity in {SiteKind.T0_MEANS, SiteKind.T0_VAR_DIAG}
    ]
    assert initial
    assert all(p.distribution is not None for p in initial)
    assert {p.id for p in initial} <= {
        b.parameter_id for b in parameter_bindings(compile_model_fixture(completed))[0]
    }
    assert completed.model_dump(mode="json") == before
    assert all(
        "elements" not in p and "role" not in p and "constraint" not in p
        for p in before["parameters"]
    )


@pytest.mark.inference(concern="sampling")
def test_parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )
    renamed = model.revised(
        parameters=tuple(p.revised(name=f"display {n}") for n, p in enumerate(model.parameters))
    )
    compile_model_fixture(renamed)
    _assert_same_prior_laws(model, renamed)
    assert {
        b.parameter_id: b.coordinates for b in parameter_bindings(compile_model_fixture(model))[0]
    } == {
        b.parameter_id: b.coordinates for b in parameter_bindings(compile_model_fixture(renamed))[0]
    }


@pytest.mark.inference(concern="sampling")
def test_additive_hill_and_linear_terms_survive_parameter_renaming():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/additive_a_b_model.json"
        ).read_text()
    )

    additive = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "scientific_parameter_identity/additive_hill_and_linear_terms_survive_parameter_renaming_with_parameter_distributions.json"
        ).read_text()
    )
    compile_model_fixture(additive)
    renamed = additive.revised(
        parameters=tuple(p.revised(name=f"opaque {n}") for n, p in enumerate(additive.parameters))
    )
    compile_model_fixture(renamed)
    _assert_same_prior_laws(additive, renamed)
    components = compile_model_fixture(additive).dynamics.spec.components
    original_components = compile_model_fixture(model).dynamics.spec.components
    assert isinstance(components, tuple)
    assert isinstance(original_components, tuple)
    assert len(components) == len(original_components) + 1


def _assert_same_prior_laws(first, second):
    import numpy as np

    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

    first_laws = compile_priors(compile_model_fixture(first), first)[0]
    second_laws = compile_priors(compile_model_fixture(second), second)[0]
    assert first_laws.keys() == second_laws.keys()
    for name, law in first_laws.items():
        for value in (0.15, 0.5, 1.25):
            np.testing.assert_allclose(law.log_prob(value), second_laws[name].log_prob(value))
