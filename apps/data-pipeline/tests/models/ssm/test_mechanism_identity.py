"""Independent additive terms survive revision without tying their free coefficients."""

from dataclasses import replace

import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, MechanismRef
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
    contribution_from_payload,
)
from nof1_causal_lab.models.model_structure import model_for_constructs
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors
from nof1_causal_lab.models.ssm.construct_admission import (
    AdmissionState,
    trial_admission_state,
)
from tests.helpers import complete_test_model, make_model
from tests.slot_fixtures import fixture_parameter_id


@pytest.fixture(scope="module")
def two_hills():
    model = complete_test_model(make_model(["A", "B"], [("A", "B")]))
    edge = model.edges[0]
    parameters = [
        p
        for p in model.parameters
        if not any(o.id == edge.id for o in model.parameter_context(p.id).owners)
    ]
    terms = []
    for identity, scale in (("mechanism:fast-response", 0.3), ("mechanism:slow-response", 1.5)):
        owners = (
            ConstructRef(id=edge.cause.id),
            ConstructRef(id=edge.effect.id),
            EdgeRef(id=edge.id),
            MechanismRef(id=identity),
        )
        coefficients = {}
        for slot, quantity in (
            ("emax", SiteKind.HILL_EMAX),
            ("ec50", SiteKind.HILL_EC50),
            ("n", SiteKind.HILL_N),
        ):
            parameter = ParameterSpec(
                id=fixture_parameter_id(quantity, owners),
                name=f"{identity} {slot}",
                description="Independent saturating response",
                distribution=dist.HalfNormal(scale),
            )
            parameters.append(parameter)
            coefficients[slot] = ParameterCoefficient(parameter_id=parameter.id)
        terms.append(
            DynamicsMechanism(
                id=identity,
                expression=expr_hill(
                    expr_state(edge.cause.id),
                    emax=coefficients["emax"],
                    ec50=coefficients["ec50"],
                    n=coefficients["n"],
                ),
            )
        )
    return model.revised(
        edges=(edge.model_copy(update={"mechanisms": tuple(terms)}),), parameters=tuple(parameters)
    )


def test_independent_hill_coefficients_survive_reorder_rename_and_submission(two_hills):
    model = two_hills
    edge = model.edges[0]
    model.check_execution()
    before = parameter_bindings(model)[0]
    before_priors = compile_priors(model)[0]
    revised = model.revised(
        edges=(edge.model_copy(update={"mechanisms": tuple(reversed(edge.mechanisms))}),),
        parameters=tuple(
            p.model_copy(update={"name": f"new label {i}"}) for i, p in enumerate(model.parameters)
        ),
    )
    revised.check_execution()
    after = parameter_bindings(revised)[0]
    after_priors = compile_priors(revised)[0]
    identities = {p.id for term in edge.mechanisms for p in model.parameters_for(term.id)}
    assert len(identities) == 6
    old = {b.parameter_id: b for b in before if b.parameter_id in identities}
    new = {b.parameter_id: b for b in after if b.parameter_id in identities}
    assert old.keys() == new.keys() == identities
    assert len({b.site_name for b in old.values()}) == 6
    for identity in identities:
        assert old[identity].elements.keys() == new[identity].elements.keys()
        assert old[identity].coordinates != new[identity].coordinates
        np.testing.assert_allclose(
            before_priors[old[identity].site_name].log_prob(0.7),
            after_priors[new[identity].site_name].log_prob(0.7),
        )
    # The authoring boundary accepts the exact canonical terms and definitions;
    # it does not require a single pre-enumerated Hill coefficient per edge.
    contribution = contribution_from_payload(
        model,
        {
            "construct": model.get_construct(edge.effect.id).model_dump(mode="json"),
            "edges": [edge.model_dump(mode="json")],
            "parameters": [p.model_dump(mode="json") for p in model.parameters_for(edge.effect.id)],
        },
    )
    assert contribution.edges[0].mechanisms == edge.mechanisms
    projected = model_for_constructs(model, {item.name for item in model.constructs})
    assert projected.parameters == model.parameters
    first = edge.mechanisms[0]
    removed_ids = {p.id for p in model.parameters_for(edge.mechanisms[1].id)}
    replacement = replace(
        contribution,
        edges=(edge.model_copy(update={"mechanisms": (first,)}),),
        parameters=tuple(p for p in contribution.parameters if p.id not in removed_ids),
    )
    accepted = trial_admission_state(AdmissionState(model=model), replacement).model
    assert {p.id for p in accepted.parameters} == {p.id for p in model.parameters} - removed_ids
    accepted.check_execution()


@pytest.mark.parametrize("change", ["duplicate", "wrong_coefficient", "delete"])
def test_term_identity_rejects_ambiguous_or_dangling_revisions(two_hills, change):
    model = two_hills
    edge = model.edges[0]
    first, second = edge.mechanisms
    if change == "duplicate":
        terms = (first, second.model_copy(update={"id": first.id}))
    elif change == "wrong_coefficient":
        from nof1_causal_lab.artifacts.expressions import (
            CoefficientExpression,
            expression_coefficients,
            map_expression,
        )

        emax = next(
            operand
            for operand in expression_coefficients(first.expression)
            if operand.role == "emax"
        )
        terms = (
            first,
            second.model_copy(
                update={
                    "expression": map_expression(
                        second.expression,
                        lambda node: (
                            emax
                            if isinstance(node, CoefficientExpression) and node.role == "emax"
                            else node
                        ),
                    )
                }
            ),
        )
    else:
        terms = (first,)
    with pytest.raises(ValueError, match=r"Duplicate mechanism|not referenced by component slots"):
        model.revised(edges=(edge.model_copy(update={"mechanisms": terms}),))
    assert model.mechanism(second.id) is second
