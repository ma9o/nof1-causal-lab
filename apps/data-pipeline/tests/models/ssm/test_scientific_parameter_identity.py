"""Stable scientific subjects through authoring, compilation, and retained results."""

import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    linear_effect,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, MechanismRef
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.flows.transitions.inference.subjects import reference_posterior_findings
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.model_checks import check_execution
from nof1_causal_lab.models.prior_planning import complete_model
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from tests.helpers import complete_test_model, make_model
from tests.slot_fixtures import fixture_parameter_id


def _model(*, rename=False, reverse=False, ordinal=False, edges=()):
    model = make_model(["A", "B"], edges or [("A", "B")])
    values = list(model.constructs)
    for n, construct in enumerate(values):
        indicator = construct.indicators[0]
        if ordinal:
            indicator = indicator.model_copy(
                update={
                    "measurement_dtype": "ordinal",
                    "aggregation": "last",
                    "ordinal_levels": ("low", "mid", "high") if n == 0 else ("absent", "present"),
                    "likelihood": LikelihoodSpec(
                        law=observation_law(construct.id, "ordered_logistic", "cumulative_logit"),
                        reasoning="Ordered test measurement",
                    ),
                }
            )
        if rename:
            indicator = indicator.model_copy(update={"name": f"renamed measurement {n}"})
        values[n] = construct.model_copy(
            update={
                "name": f"renamed construct {n}" if rename else construct.name,
                "indicators": (indicator,),
            }
        )
    return model.revised(
        edges=replace_constructs(tuple(reversed(model.edges)) if reverse else model.edges, values)
    )


def _compile(model):
    completed = complete_test_model(model)
    plan = completed
    return check_execution(completed), completed, plan


def test_rename_preserves_parameter_and_element_identity():
    _before, old_model, _ = _compile(_model())
    _after, new_model, _ = _compile(_model(rename=True))
    assert {b.parameter_id: set(b.elements) for b in parameter_bindings(old_model)[0]} == {
        b.parameter_id: set(b.elements) for b in parameter_bindings(new_model)[0]
    }
    assert {p.name for p in old_model.parameters} != {p.name for p in new_model.parameters}


def test_scalar_identity_survives_reordered_execution_axes():
    feedback = [("A", "B"), ("B", "A")]
    _before, model, _ = _compile(_model(edges=feedback))
    _after, reordered, _ = _compile(_model(edges=feedback, reverse=True))
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


def test_model_rejects_forged_owner_before_compilation():
    _, model, _ = _compile(_model())
    payload = model.model_dump(mode="json")
    payload["parameters"][0]["owners"] = [{"kind": "construct", "id": "construct:forged"}]
    with pytest.raises(ValueError, match=r"Extra inputs|owner"):
        type(model).model_validate(payload)


def test_posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate():
    _compiled, model, _ = _compile(_model())
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
    marginals, _ = reference_posterior_findings(model, [row], [])
    assert marginals[0]["subject"] == {"parameter_id": binding.parameter_id, "element_id": element}
    assert "coordinate" not in marginals[0]
    row["coordinate"] = {"site_name": "unknown", "indices": []}
    with pytest.raises(ValueError, match="unbound runtime coordinate"):
        reference_posterior_findings(model, [row], [])


def test_ordinal_components_have_label_identity_and_padding_is_explicit():
    _compiled, model, _ = _compile(_model(ordinal=True))
    gaps = {
        p.id
        for p in model.parameters
        if model.parameter_context(p.id).quantity == SiteKind.OBS_ORDERED_GAPS
    }
    gap_bindings = [b for b in parameter_bindings(model)[0] if b.parameter_id in gaps]
    assert len(gap_bindings) == 1
    assert list(gap_bindings[0].elements.values()) == ["A_obs: gap low / mid / high"]
    assert parameter_bindings(model)[1]


def test_shared_likelihood_parameter_owns_only_active_channels():
    _, model, _ = _compile(_model())
    first, second = model.constructs
    student = LikelihoodSpec(
        law=observation_law(first.id, "student_t", "identity"),
        reasoning="Test tails",
        standardized=True,
    )
    first = first.model_copy(
        update={"indicators": (first.indicators[0].model_copy(update={"likelihood": student}),)}
    )
    model = complete_model(model.revised(edges=replace_constructs(model.edges, (first, second))))
    shared = next(
        p for p in model.parameters if model.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert {o.id for o in model.parameter_context(shared.id).owners} == {
        first.id,
        first.indicators[0].id,
    }
    second = second.model_copy(
        update={
            "indicators": (
                second.indicators[0].model_copy(
                    update={
                        "likelihood": LikelihoodSpec(
                            law=observation_law(second.id, "student_t", "identity"),
                            reasoning="Test tails",
                            standardized=True,
                        )
                    }
                ),
            )
        }
    )
    expanded = model.revised(
        edges=replace_constructs(model.edges, (model.get_construct(first.id), second))
    )
    expanded = complete_model(expanded)
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


def test_student_innovation_tail_is_explicit_and_shared_through_completion():
    from nof1_causal_lab.models.model_parameters import referenced_parameter_ids

    _, model, _ = _compile(_model())
    model = complete_model(
        model.revised(
            edges=replace_constructs(
                model.edges,
                tuple(
                    construct.model_copy(
                        update={
                            "innovation": construct.innovation.model_copy(
                                update={"distribution": "student_t"}
                            )
                        }
                    )
                    for construct in model.constructs
                ),
            )
        )
    )
    parameter = next(p for p in model.parameters if p.name == "proc_df")
    for construct in model.constructs:
        assert construct.innovation is not None
        assert parameter.id in referenced_parameter_ids(construct.innovation)
    model.check_execution()
    first, second = model.constructs
    assert first.innovation is not None
    candidate = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                first.model_copy(
                    update={
                        "innovation": first.innovation.model_copy(
                            update={"degrees_of_freedom": None}
                        )
                    }
                ),
                second,
            ),
        )
    )
    with pytest.raises(ValueError, match="degrees_of_freedom requires a prior parameter"):
        candidate.check_execution()
    completed = complete_model(candidate)
    assert completed.parameter(parameter.id) == parameter
    assert completed.get_construct(first.id).innovation == first.innovation


def test_initial_state_defaults_are_authored_before_compilation():
    _, model, _ = _compile(_model())
    from nof1_causal_lab.models.parameter_planning import complete_component_slots

    free = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(c.model_copy(update={"initial_state": None}) for c in model.constructs),
        )
    )
    with pytest.raises(ValueError, match="initial-state coefficients"):
        free.check_execution()
    completed = complete_model(complete_component_slots(free, free_initial=True))
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


def test_parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws():
    before, model, _plan = _compile(_model())
    renamed = model.revised(
        parameters=tuple(
            p.model_copy(update={"name": f"display {n}"}) for n, p in enumerate(model.parameters)
        )
    )
    after = check_execution(renamed)
    assert before == after
    _assert_same_prior_laws(model, renamed)
    assert {b.parameter_id: b.coordinates for b in parameter_bindings(model)[0]} == {
        b.parameter_id: b.coordinates for b in parameter_bindings(renamed)[0]
    }


def test_additive_hill_and_linear_terms_survive_parameter_renaming():
    _, model, _plan = _compile(_model(edges=(("A", "B"),)))
    edge = model.edges[0]
    owners = (
        EdgeRef(id=edge.id),
        ConstructRef(id=edge.cause.id),
        ConstructRef(id=edge.effect.id),
        MechanismRef(id="mechanism:test-hill"),
    )
    peak = ParameterSpec(
        id=fixture_parameter_id(SiteKind.HILL_EMAX, owners),
        name="Peak effect",
        description="Test nonlinear contribution",
        distribution=dist.HalfNormal(1.0),
    )
    hill = DynamicsMechanism(
        id="mechanism:test-hill",
        expression=expr_hill(
            expr_state(edge.cause.id),
            emax=ParameterCoefficient(parameter_id=peak.id),
            ec50=FixedCoefficient(value=1.0),
            n=FixedCoefficient(value=2.0),
        ),
    )
    additive = model.revised(
        edges=(edge.model_copy(update={"mechanisms": (*edge.mechanisms, hill)}),),
        parameters=(*model.parameters, peak),
    )
    before = check_execution(additive)
    renamed = additive.revised(
        parameters=tuple(
            p.model_copy(update={"name": f"opaque {n}"}) for n, p in enumerate(additive.parameters)
        )
    )
    after = check_execution(renamed)
    assert before == after
    _assert_same_prior_laws(additive, renamed)
    components = numeric.dynamics_components(additive).components
    original_components = numeric.dynamics_components(model).components
    assert isinstance(components, tuple)
    assert isinstance(original_components, tuple)
    assert len(components) == len(original_components) + 1


def test_known_input_mechanism_cannot_silently_double_its_effect():
    from nof1_causal_lab.artifacts.construct import KnownInput

    model = _model(edges=(("A", "B"),))
    first, second = model.constructs
    first = first.model_copy(
        update={
            "role": "exogenous",
            "usage": KnownInput(source_indicator_id=first.indicators[0].id),
        }
    )
    _, model, _plan = _compile(
        model.revised(edges=replace_constructs(model.edges, (first, second)))
    )
    edge = model.edges[0]
    first = edge.mechanisms[0]
    weight = next(p for p in model.parameters_for(first.id))
    owners = (
        *(
            owner
            for owner in model.parameter_context(weight.id).owners
            if owner.kind != "mechanism"
        ),
        MechanismRef(id="mechanism:second-input-effect"),
    )
    second_weight = weight.model_copy(
        update={"id": fixture_parameter_id(model.parameter_context(weight.id).quantity, owners)}
    )
    second = first.model_copy(
        update={
            "id": "mechanism:second-input-effect",
            "expression": linear_effect(
                edge.cause.id, ParameterCoefficient(parameter_id=second_weight.id)
            ),
        }
    )
    duplicated = model.revised(
        edges=(edge.model_copy(update={"mechanisms": (first, second)}),),
        parameters=(*model.parameters, second_weight),
    )
    with pytest.raises(
        ValueError,
        match=r"Multiple scientific definitions|one linear expression per input matrix cell",
    ):
        check_execution(duplicated)


def _assert_same_prior_laws(first, second):
    import numpy as np

    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

    first_laws = compile_priors(first)[0]
    second_laws = compile_priors(second)[0]
    assert first_laws.keys() == second_laws.keys()
    for name, law in first_laws.items():
        for value in (0.15, 0.5, 1.25):
            np.testing.assert_allclose(law.log_prob(value), second_laws[name].log_prob(value))
