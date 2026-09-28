"""Shared model fixtures and async test helpers."""

import asyncio
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal, overload

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import ConstructId, EdgeId, IndicatorId, MechanismId
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction


@overload
def fixture_entity_id(kind: Literal["construct"], initial_identity: str) -> ConstructId: ...


@overload
def fixture_entity_id(kind: Literal["edge"], initial_identity: str) -> EdgeId: ...


@overload
def fixture_entity_id(kind: Literal["indicator"], initial_identity: str) -> IndicatorId: ...


@overload
def fixture_entity_id(kind: Literal["mechanism"], initial_identity: str) -> MechanismId: ...


@overload
def fixture_entity_id(kind: str, initial_identity: str) -> str: ...


def fixture_entity_id(kind: str, initial_identity: str) -> str:
    """Assign reproducible IDs when constructing a new test fixture."""
    return f"{kind}:{sha256(f'{kind}\0{initial_identity}'.encode()).hexdigest()[:20]}"


def run_async(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.run(coro)


def invalid_dict_payload(value: object) -> Any:
    return value


def make_prior_model(statistical_model_spec, priors):
    """Attach readable test prior inputs to their scientific parameter definitions."""
    return model_with_prior_payloads(
        statistical_model_spec, named_prior_payloads(statistical_model_spec, priors)
    )


def model_with_prior_payloads(model, payloads):
    """Attach explicit ID-keyed law inputs for compiler behavior tests."""
    from notebooks.predictive_support import model_with_prior_payloads as attach_priors

    if model is None:
        if payloads:
            raise ValueError("Prior inputs require a scientific model")
        return None
    return attach_priors(model, payloads)


def make_model(state_names: list[str], edges: Sequence[tuple[str, str]] = ()):
    """A connected test graph; uncoupled states share an unmeasured downstream outcome."""
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    constructs = {
        name: ConstructSpec.model_validate(
            {
                "id": fixture_entity_id("construct", name),
                "name": name,
                "description": name,
                "role": "endogenous",
                "temporal_status": "time_varying",
                "indicators": [
                    {
                        "id": fixture_entity_id("indicator", name + "_obs"),
                        "name": name + "_obs",
                        "construct_polarity": "positive",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            }
        )
        for name in state_names
    }
    if edges:
        assert set(state_names) == {name for pair in edges for name in pair}
    if not edges:
        outcome = "unmeasured_outcome"
        constructs[outcome] = ConstructSpec(
            id=fixture_entity_id("construct", outcome),
            name=outcome,
            description="A downstream response outside the numerical fixture's measured states.",
            role="endogenous",
            temporal_status="time_varying",
        )
        edges = [(name, outcome) for name in state_names]
    return ModelSpec(
        edges=tuple(
            CausalEdgeSpec(
                id=fixture_entity_id("edge", cause + "->" + effect),
                cause=constructs[cause],
                effect=constructs[effect],
                description=cause + " causes " + effect,
            )
            for cause, effect in edges
        ),
        measurement_clock="1d",
    )


def graph_constructs(payload):
    """Writable endpoint definitions in a serialized test graph, excluding shared references."""
    return [
        endpoint
        for edge in payload["edges"]
        for endpoint in (edge["cause"], edge["effect"])
        if "name" in endpoint
    ]


def complete_test_model(model, *, self_limiting=(), hill_edges=()):
    """Explicit Gaussian/continuous test choices followed by scientific completion."""
    from evaluation.fixtures.prior_planning import complete_model
    from notebooks.model_mechanisms import declare_dynamics

    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.models.likelihoods import observation_law

    manifest = set(model.manifest_indicator_order)
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.model_copy(
                    update={
                        "indicators": tuple(
                            i
                            if i.likelihood is not None or i.id not in manifest
                            else i.model_copy(
                                update={
                                    "likelihood": LikelihoodSpec(
                                        law=observation_law(
                                            c.id, DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY
                                        ),
                                        reasoning="Test Gaussian emission",
                                        standardized=True,
                                    )
                                }
                            )
                            for i in c.indicators
                        )
                    }
                )
                for c in model.constructs
            ),
        )
    )
    authored = declare_dynamics(model, self_limiting=self_limiting, hill_edges=hill_edges)
    return complete_model(authored)


def named_prior_payloads(model, proposals):
    """Build ID-keyed compiler inputs from readable labels in a test's prior table."""
    ids = {parameter.name: parameter.id for parameter in model.parameters}
    return {ids[label]: payload for label, payload in proposals.items()}


def native_axis_metadata(n_latent, n_manifest, metadata):
    """Declare stable scientific axes when constructing a new native test model."""
    latent_names = metadata.get("latent_names") or [f"latent_{index}" for index in range(n_latent)]
    manifest_names = metadata.get("manifest_names") or [
        f"manifest_{index}" for index in range(n_manifest)
    ]
    return {
        "latent_names": latent_names,
        "manifest_names": manifest_names,
        "latent_ids": [fixture_entity_id("construct", name) for name in latent_names],
        "manifest_ids": [fixture_entity_id("indicator", name) for name in manifest_names],
        "static_factor_ids": [],
        **metadata,
    }


def declare_test_dynamics(
    model, *, quartic_states=(), hill_edges=(), centered_states=(), additional_parameters=()
):
    """Author an explicit dynamics fixture with optional quartic, Hill, and center choices."""
    from notebooks.model_mechanisms import declare_dynamics

    declared = declare_dynamics(
        model,
        self_limiting=quartic_states,
        hill_edges=hill_edges,
        centered_states=centered_states,
    )
    replacements = {parameter.name: parameter for parameter in additional_parameters}
    return declared.revised(
        parameters=tuple(
            replacements.get(parameter.name, parameter) for parameter in declared.parameters
        )
    )
