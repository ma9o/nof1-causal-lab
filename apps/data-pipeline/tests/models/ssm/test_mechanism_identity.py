"""Independent additive terms survive revision without tying their free coefficients."""

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from pathlib import Path

import numpy as np
import pytest

from nof1_causal_lab.artifacts.expressions import BinaryExpression
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors


@pytest.fixture(scope="module")
def two_hills():
    return ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'mechanism_identity/two_hills_model.json').read_text())


@pytest.mark.inference(concern="sampling")
def test_independent_hill_coefficients_survive_reorder_rename_and_submission(two_hills):
    model = two_hills
    edge = model.edges[0]
    model.check_execution()
    before = parameter_bindings(model)[0]
    before_priors = compile_priors(model)[0]
    revised = model.revised(
        edges=(
            type(edge).model_validate(
                {**edge.model_dump(), "mechanisms": tuple(reversed(edge.mechanisms))}
            ),
        ),
        parameters=tuple(
            type(p).model_validate({**p.model_dump(), "name": f"new label {i}"})
            for i, p in enumerate(model.parameters)
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
    # Incremental edits submit the whole authored model directly.

    submitted = ModelSpec.model_validate(revised.model_dump(mode="json"))
    assert submitted.edges[0].mechanisms == revised.edges[0].mechanisms
    removed_ids = {p.id for p in model.parameters_for(edge.mechanisms[1].id)}
    remaining = tuple(p for p in model.parameters if p.id not in removed_ids)
    edited = model.revised(
        edges=(
            type(edge).model_validate({**edge.model_dump(), "mechanisms": (edge.mechanisms[0],)}),
        ),
        parameters=remaining,
        distributions={
            key: law
            for key, law in model.distributions.items()
            if key in {p.distribution for p in remaining}
        },
    )
    assert {p.id for p in edited.parameters} == {p.id for p in model.parameters} - removed_ids
    edited.check_execution()


@pytest.mark.contract
@pytest.mark.parametrize("change", ["duplicate", "wrong_coefficient", "delete"])
def test_term_identity_rejects_ambiguous_or_dangling_revisions(two_hills, change):
    model = two_hills
    edge = model.edges[0]
    first, second = edge.mechanisms
    if change == "duplicate":
        terms = (first, type(second).model_validate({**second.model_dump(), "id": first.id}))
    elif change == "wrong_coefficient":


        terms = (
            first,
            type(second).model_validate(
                {
                    **second.model_dump(),
                    "expression": BinaryExpression.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'mechanism_identity/term_identity_rejects_ambiguous_or_dangling_revisions_map_expression.json').read_text()),
                }
            ),
        )
    else:
        terms = (first,)
    with pytest.raises(ValueError, match=r"Duplicate mechanism|not referenced by component slots"):
        model.revised(
            edges=(type(edge).model_validate({**edge.model_dump(), "mechanisms": terms}),)
        )
    assert model.mechanism(second.id) is second
