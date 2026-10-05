"""ModelSpec identity, parameter draws, and native model execution."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import ObservationLawSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import (
    StructuralSelection,
)
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)
from tests.helpers import make_model
from tests.inference_fixtures import compile_model_fixture, model_draws, particle_posterior
from tests.model_fixtures import (
    load_model_fixture,
)


def _conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model() -> (
    ModelSpec
):
    return load_model_fixture(
        "model_spec_execution/conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model.json"
    )


def _model_model() -> ModelSpec:
    return load_model_fixture("model_spec_execution/model_model.json")


@pytest.fixture(scope="module")
def model():
    return _model_model()


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize("categorical", [False, True])
def test_conditioning_revises_the_same_type_and_retains_joint_uncertainty(
    model, tmp_path, monkeypatch, categorical
):
    from functools import cache

    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.models.model_inputs import input_fingerprints
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
    from nof1_causal_lab.numpyro_json import empirical_atoms, empirical_distribution
    from nof1_causal_lab.study.store import ArtifactStore, read_model
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    if categorical:
        model = make_model(["A", "B"], [("A", "B")])
        construct = model.constructs[1]
        indicator = construct.indicators[0].revised(
            observation=construct.indicators[0].observation.revised(
                measurement_dtype="categorical",
                categorical_levels=("low", "medium", "high"),
                aggregation="last",
            ),
            likelihood=LikelihoodSpec(
                law=TypeAdapter(ObservationLawSpec).validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "model_spec_execution/conditioning_revises_the_same_type_and_retains_joint_uncertainty_observation_law.json"
                    ).read_text()
                ),
                reasoning="Joint law with category-specific parameter elements",
            ),
        )
        model = (
            _conditioning_revises_the_same_type_and_retains_joint_uncertainty_complete_test_model()
        )
    count = 3
    samples = {
        site.name: 100 * (index + 1)
        + jnp.arange(count * np.prod(site.shape), dtype=float).reshape(count, *site.shape)
        for index, site in enumerate(build_site_registry(compile_model_fixture(model)))
    }
    bindings, auxiliary = parameter_bindings(compile_model_fixture(model))
    if categorical:
        assert any(len(binding.coordinates) > 1 for binding in bindings)
    for coordinate in auxiliary:
        samples[coordinate.site_name] = (
            samples[coordinate.site_name].at[(slice(None), *coordinate.indices)].set(0)
        )
    samples.update(assemble_deterministics_from_registry(samples, compile_model_fixture(model)))
    paths = jnp.arange(count * 4 * 2, dtype=float).reshape(count, 4, 2)
    result = particle_posterior(JointPosteriorDraws(samples, paths))
    conditioned, _ = condition_model(
        model,
        compile_model_fixture(model),
        result,
        times=jnp.arange(4),
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
    )
    assert type(conditioned) is ModelSpec
    assert model.time_points == ()
    assert len(model.distributions) == len(model.parameters)
    assert len(conditioned.distributions) == 1
    assert all(
        p.distribution == next(iter(conditioned.distributions)) for p in conditioned.parameters
    )
    payload = conditioned.model_dump(mode="json")
    assert not {"posterior", "prior", "provenance", "diagnostics", "result"} & payload.keys()
    assert "array_ref" in conditioned.model_dump_json()
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="fit",
        json_files={"model.json": payload},
    )
    loaded = read_model(store, info.revision)
    assert loaded == conditioned
    assert input_fingerprints(model)["compilation"] == input_fingerprints(loaded)["compilation"]
    restored = model_draws(compile_model_fixture(loaded))
    for name, values in samples.items():
        np.testing.assert_array_equal(restored.parameters[name], values)
    np.testing.assert_array_equal(restored.latent_paths, paths)
    sampled = sample_model_laws(compile_model_fixture(loaded), draws=12, key=jax.random.PRNGKey(4))
    assert sampled.latent_paths is not None
    for draw in range(12):
        atom = int(sampled.latent_paths[draw, 0, 0] // 8)
        np.testing.assert_array_equal(sampled.latent_paths[draw], paths[atom])
        for name, values in samples.items():
            np.testing.assert_array_equal(sampled.parameters[name][draw], values[atom])
    # Entity/parameter list order carries no joint distribution coordinates.
    reordered = loaded.revised(parameters=tuple(reversed(loaded.parameters)))
    for name, values in restored.parameters.items():
        np.testing.assert_array_equal(
            model_draws(compile_model_fixture(reordered)).parameters[name], values
        )
    assert (
        loaded.revised(edges=replace_constructs(loaded.edges, tuple(reversed(loaded.constructs))))
        == loaded
    )
    np.testing.assert_array_equal(
        empirical_atoms(next(iter(loaded.distributions.values()))),
        empirical_atoms(next(iter(conditioned.distributions.values()))),
    )
    identity, law = next(iter(loaded.distributions.items()))
    # A joint law's coordinates follow the outcome's scope, which checks them.
    with pytest.raises(ValueError, match="event width must match"):
        StructuralSelection(
            loaded.revised(
                distributions={identity: empirical_distribution(empirical_atoms(law)[:, :-1])}
            ),
            None,
        )
    with pytest.raises(ValueError, match="identity and event width must match"):
        StructuralSelection(
            loaded.revised(
                law_layouts={
                    identity: loaded.law_layouts[identity].revised(time_points=(0.0, 1.0, 2.0, 5.0))
                }
            ),
            None,
        )
    if categorical:
        construct = loaded.constructs[1]
        indicator = construct.indicators[0]
        renamed = loaded.revised(
            edges=replace_constructs(
                loaded.edges,
                (
                    construct.revised(
                        indicators=(
                            indicator.revised(
                                observation=indicator.observation.revised(
                                    categorical_levels=("medium", "low", "high")
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        assert renamed.law_layouts == loaded.law_layouts
