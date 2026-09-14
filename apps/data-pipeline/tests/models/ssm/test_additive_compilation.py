"""Numerical parity against retained arrays; never fit or simulate a model."""

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors
from nof1_causal_lab.models.ssm.construct_admission import (
    ConstructContribution,
    _incoming_edge_off_target,
)
from nof1_causal_lab.numpyro_json import decode_distribution
from tests.helpers import complete_test_model, make_model


@pytest.fixture(scope="module")
def retained():
    root = Path(__file__).resolve().parents[5]
    return (
        ModelSpec.model_validate_json(
            (root / "data/DEMO/fixture/artifacts/model.json").read_text()
        ),
        json.loads(
            (Path(__file__).resolve().parents[2] / "fixtures/numerical_reference.json").read_text()
        ),
    )


def test_numerical_views_preserve_arrays_without_a_second_spec(retained):
    model, expected = retained
    before = model.model_dump(mode="json")
    functions = {
        "diffusion_block": numeric.diffusion_block,
        "t0_means_block": numeric.initial_mean_block,
        "t0_chol_block": numeric.initial_covariance_block,
        "static_state_sd_block": numeric.static_scale_block,
        "loadings": numeric.loading_block,
        "means": numeric.observation_mean_block,
        "noise": numeric.observation_noise_block,
        "input_effect": numeric.input_effect_block,
    }
    states = [expected["state_ids"].index(key) for key in numeric.state_ids(model)]
    observations = [
        expected["observation_ids"].index(key) for key in numeric.observation_ids(model)
    ]
    inputs = [expected["input_ids"].index(key) for key in numeric.input_ids(model)]
    factors = [expected["static_factor_ids"].index(key) for key in numeric.static_factor_ids(model)]
    for name, function in functions.items():
        for field, value in asdict(function(model)).items():
            old = expected["blocks"][name][field]
            if hasattr(value, "shape"):
                previous = np.asarray(old).reshape(value.shape)
                axes = (
                    (observations, states)
                    if name == "loadings"
                    else (states, inputs)
                    if name == "input_effect"
                    else (factors,)
                    if name == "static_state_sd_block"
                    else (observations, observations)
                    if name in {"means", "noise"}
                    else (states, states)
                )
                for axis in range(previous.ndim):
                    previous = np.take(previous, axes[axis], axis=axis)
                np.testing.assert_array_equal(value, previous, err_msg=f"{name}.{field}")
            else:
                assert (value.value if isinstance(value, Enum) else value) == old
    np.testing.assert_array_equal(
        numeric.static_factor_loadings(model),
        np.asarray(expected["blocks"]["static_factor_loadings"])[np.ix_(states, factors)],
    )
    assert set(numeric.state_ids(model)) == set(expected["state_ids"])
    assert model.model_dump(mode="json") == before
    assert {anchor.construct_id for anchor in model.check_execution()} == set(
        numeric.state_ids(model)
    )
    assert model.model_dump(mode="json") == before


def test_scientific_subjects_and_native_prior_densities_are_unchanged(retained):
    model, expected = retained
    old = {b["parameter_id"]: b for b in expected["bindings"]}
    new = {b.parameter_id: b for b in parameter_bindings(model)[0]}
    priors = compile_priors(
        model,
        edge_lag_days=numeric.edge_lag_days(model),
    )[0]
    assert old.keys() == new.keys() == {p.id for p in model.parameters}
    for value in (0.15, 0.5, 1.25):
        for identity, binding in new.items():
            # Graph traversal determines the execution basis. Covariance element IDs
            # include that ordered basis; compare the retained scalar by its label.
            assert set(old[identity]["elements"].values()) == set(binding.elements.values())
            for element, coord in binding.coordinates.items():
                previous_element = next(
                    key
                    for key, label in old[identity]["elements"].items()
                    if label == binding.elements[element]
                )
                previous = old[identity]["coordinates"][previous_element]
                old_law = decode_distribution(expected["priors"][previous["site_name"]])
                law = priors[coord.site_name]
                np.testing.assert_allclose(
                    old_law.log_prob(jnp.full(old_law.batch_shape, value))[
                        tuple(previous["indices"])
                    ],
                    law.log_prob(jnp.full(law.batch_shape, value))[coord.indices],
                    rtol=1e-6,
                    atol=1e-6,
                )


def test_edge_off_targets_every_additive_contribution_without_running_a_simulation():
    model = complete_test_model(make_model(["A", "B"], [("A", "B")]))
    edge = model.edges[0]
    fixed_hill = DynamicsMechanism(
        id="mechanism:fixed-hill-a",
        expression=expr_hill(
            expr_state(edge.cause.id),
            emax=FixedCoefficient(value=0.4),
            ec50=FixedCoefficient(value=1),
            n=FixedCoefficient(value=2),
        ),
    )
    model = model.revised(
        edges=(
            edge.model_copy(
                update={
                    "mechanisms": (
                        *edge.mechanisms,
                        fixed_hill,
                        fixed_hill.model_copy(update={"id": "mechanism:fixed-hill-b"}),
                    )
                }
            ),
        )
    )
    native = model
    target = model.get_construct(edge.effect.id)
    source = model.get_construct(edge.cause.id)
    contribution = ConstructContribution(
        construct=target, edges=model.edges, edge_parents=(source.name,)
    )
    assert numeric.state_names(native) is not None
    off = _incoming_edge_off_target(
        native,
        contribution,
        numeric.state_names(native),
        numeric.state_names(native).index(target.name),
    )
    assert len(off.components) == 3
    assert off.input_effect_cells == ()
