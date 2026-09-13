"""StructuralPlan planning, closure, and persisted binding contracts."""

from __future__ import annotations

from typing import Any

import pytest

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
from nof1_causal_lab.artifacts.structural_plan import StructuralPlan  # noqa: TC001
from nof1_causal_lab.flows.transitions.model_spec.agentic.parameter_surfaces import (
    parameter_is_active_for_statistical_model_spec,
)
from nof1_causal_lab.flows.transitions.model_spec.agentic.skeleton import (
    derive_deterministic_spec,
)
from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
from nof1_causal_lab.models.structural import (
    StructuralPlanningError,
    build_structural_plan,
)
from nof1_causal_lab.utils.structural_plan import (
    get_edges,
    get_known_inputs,
    get_manifest_indicators,
    get_state_names,
)
from tests.helpers import fixture_entity_id, make_prior_plan


def _indicator(name: str, construct: str) -> dict[str, Any]:
    return {
        "id": fixture_entity_id("indicator", name),
        "construct_id": fixture_entity_id("construct", construct),
        "name": name,
        "construct_polarity": "positive",
        "how_to_measure": f"measure {name}",
        "measurement_dtype": "continuous",
        "aggregation": "mean",
    }


def _dynamic_design() -> dict[str, Any]:
    return {
        "latent": {
            "default_outcome": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
            "constructs": [
                {
                    "id": "construct:311c9047b5ede16a8f26",
                    "name": "X",
                    "description": "cause",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:d90c52e59b79004188dc",
                    "name": "Y",
                    "description": "outcome",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause_id": "construct:311c9047b5ede16a8f26",
                    "effect_id": "construct:d90c52e59b79004188dc",
                    "id": "edge:39ba80b774e02c409662",
                    "description": "X causes Y",
                    "lagged": True,
                }
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [_indicator("x_obs", "X"), _indicator("y_obs", "Y")],
        },
        "known_inputs": [],
        "scientific_only_constructs": [],
    }


def test_planner_rejects_retained_static_target_edge():
    design = _dynamic_design()
    design["latent"]["constructs"].insert(
        1,
        {
            "id": "construct:7cc3028e04224dd88c7d",
            "name": "Baseline",
            "description": "stable mediator",
            "role": "endogenous",
            "temporal_status": "time_invariant",
        },
    )
    design["latent"]["edges"] = [
        {
            "cause_id": "construct:311c9047b5ede16a8f26",
            "effect_id": "construct:7cc3028e04224dd88c7d",
            "id": "edge:378960547a3758865b3b",
            "description": "unsupported baseline equation",
            "lagged": False,
        },
        {
            "cause_id": "construct:7cc3028e04224dd88c7d",
            "effect_id": "construct:d90c52e59b79004188dc",
            "id": "edge:c1a32864c9c3a7264fb5",
            "description": "baseline predicts Y",
            "lagged": False,
        },
    ]
    design["latent"]["constructs"][0]["temporal_status"] = "time_invariant"
    design["measurement"]["indicators"].insert(1, _indicator("baseline_obs", "Baseline"))

    with pytest.raises(StructuralPlanningError, match="static-target edge"):
        build_structural_plan(CausalDesign.model_validate(design))


def test_planner_rejects_multiple_lag_classes_for_one_edge():
    design = _dynamic_design()
    design["latent"]["edges"].append(
        {
            "cause_id": "construct:311c9047b5ede16a8f26",
            "effect_id": "construct:d90c52e59b79004188dc",
            "id": "edge:4a936ebb99d108078f01",
            "description": "duplicate endpoint with a different lag class",
            "lagged": False,
        }
    )

    with pytest.raises(StructuralPlanningError, match="multiple lag classes"):
        build_structural_plan(CausalDesign.model_validate(design))


def test_planner_records_known_input_and_scientific_only_dispositions():
    design = _dynamic_design()
    design["latent"]["constructs"].insert(
        0,
        {
            "id": "construct:96800712f58a2ef2e759",
            "name": "Genotype",
            "description": "known stable driver",
            "role": "exogenous",
            "temporal_status": "time_invariant",
        },
    )
    design["latent"]["constructs"].insert(
        1,
        {
            "id": "construct:6440dd37ede553148b7d",
            "name": "History",
            "description": "scientific context",
            "role": "exogenous",
            "temporal_status": "time_invariant",
        },
    )
    design["latent"]["edges"].insert(
        0,
        {
            "cause_id": "construct:96800712f58a2ef2e759",
            "effect_id": "construct:d90c52e59b79004188dc",
            "id": "edge:a6713ef89213eec2f5c1",
            "description": "known baseline driver",
            "lagged": False,
        },
    )
    design["measurement"]["indicators"] = [
        _indicator("genotype_obs", "Genotype"),
        _indicator("history_obs", "History"),
        *design["measurement"]["indicators"],
    ]
    design["known_inputs"] = [
        {
            "construct_id": "construct:96800712f58a2ef2e759",
            "source_indicator_id": "indicator:e51f080bced00e19d205",
            "missing_policy": "forward_fill",
        }
    ]
    design["scientific_only_constructs"] = [
        {
            "construct_id": "construct:6440dd37ede553148b7d",
            "reason": "context, not an estimable state",
        }
    ]

    plan = build_structural_plan(CausalDesign.model_validate(design))

    assert get_state_names(plan) == ["X", "Y"]
    assert [item["construct"] for item in get_known_inputs(plan)] == ["Genotype"]
    assert [item["name"] for item in get_manifest_indicators(plan)] == [
        "x_obs",
        "y_obs",
    ]
    assert [(edge["cause"], edge["effect"]) for edge in get_edges(plan)] == [
        ("Genotype", "Y"),
        ("X", "Y"),
    ]
    dispositions = {item.source_id: item.disposition.value for item in plan.dispositions}
    construct_ids = {
        construct.name: source_id for source_id, construct in plan.semantics.constructs.items()
    }
    assert dispositions[construct_ids["Genotype"]] == "known_input"
    assert dispositions[construct_ids["History"]] == "identification_only"


def test_source_ids_are_stable_across_authoring_reordering():
    design = {
        "latent": {
            "constructs": [
                {
                    "id": "construct:96800712f58a2ef2e759",
                    "name": "Genotype",
                    "description": "known stable driver",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
                {
                    "id": "construct:6adb2ea5d7ef5f732601",
                    "name": "U",
                    "description": "unobserved common cause",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                *_dynamic_design()["latent"]["constructs"],
            ],
            "edges": [
                {
                    "cause_id": "construct:96800712f58a2ef2e759",
                    "effect_id": "construct:d90c52e59b79004188dc",
                    "id": "edge:a6713ef89213eec2f5c1",
                    "description": "known driver",
                    "lagged": False,
                },
                {
                    "cause_id": "construct:6adb2ea5d7ef5f732601",
                    "effect_id": "construct:311c9047b5ede16a8f26",
                    "id": "edge:a884ded3cb0d95aaa1a3",
                    "description": "confounding path one",
                    "lagged": True,
                },
                {
                    "cause_id": "construct:6adb2ea5d7ef5f732601",
                    "effect_id": "construct:d90c52e59b79004188dc",
                    "id": "edge:864e17d9483a6d797070",
                    "description": "confounding path two",
                    "lagged": True,
                },
                *_dynamic_design()["latent"]["edges"],
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                _indicator("genotype_obs", "Genotype"),
                *_dynamic_design()["measurement"]["indicators"],
            ],
        },
        "known_inputs": [
            {
                "construct_id": "construct:96800712f58a2ef2e759",
                "source_indicator_id": "indicator:e51f080bced00e19d205",
                "missing_policy": "forward_fill",
            }
        ],
        "scientific_only_constructs": [],
    }
    design["latent"]["constructs"][2]["role"] = "endogenous"
    original = build_structural_plan(CausalDesign.model_validate(design))
    assert original.known_inputs
    assert original.induced_dependencies

    reordered_design = {
        **design,
        "latent": {
            **design["latent"],
            "constructs": list(reversed(design["latent"]["constructs"])),
            "edges": list(reversed(design["latent"]["edges"])),
        },
        "measurement": {
            **design["measurement"],
            "indicators": list(reversed(design["measurement"]["indicators"])),
        },
    }
    reordered = build_structural_plan(CausalDesign.model_validate(reordered_design))

    def _ids_by_name(plan: StructuralPlan):
        construct_name_by_id = {
            source_id: item.name for source_id, item in plan.semantics.constructs.items()
        }
        indicator_name_by_id = {
            source_id: item.name for source_id, item in plan.semantics.indicators.items()
        }
        return {
            "constructs": {
                item.name: source_id for source_id, item in plan.semantics.constructs.items()
            },
            "indicators": {
                item.name: source_id for source_id, item in plan.semantics.indicators.items()
            },
            "edges": {
                (item.cause_id, item.effect_id, item.lagged): source_id
                for source_id, item in plan.semantics.edges.items()
            },
            "known_inputs": {
                (
                    construct_name_by_id[item.construct_id],
                    indicator_name_by_id[item.source_indicator_id],
                ): item.source_id
                for item in plan.known_inputs
            },
            "dependencies": {
                (
                    item.kind,
                    tuple(sorted(construct_name_by_id[source_id] for source_id in item.between)),
                    tuple(
                        sorted(
                            construct_name_by_id[source_id]
                            for source_id in item.source_confounder_ids
                        )
                    ),
                ): item.source_id
                for item in plan.induced_dependencies
            },
        }

    assert _ids_by_name(original) == _ids_by_name(reordered)


def test_compiled_artifact_has_total_structural_bindings_and_canonical_order():
    plan = build_structural_plan(CausalDesign.model_validate(_dynamic_design()))
    skeleton = derive_deterministic_spec(plan)
    likelihoods: list[dict[str, Any]] = [
        {
            "indicator_id": fixture_entity_id("indicator", variable),
            "distribution": "gaussian",
            "link": "identity",
            "standardized": True,
            "reasoning": "test",
        }
        for variable in ("x_obs", "y_obs")
    ]
    likelihood_by_variable: dict[str, dict[str, Any]] = {
        likelihood["indicator_id"]: likelihood for likelihood in likelihoods
    }
    parameters: list[dict[str, Any]] = []
    for candidate in skeleton.all_params:
        parameter = dict(candidate)
        if parameter_is_active_for_statistical_model_spec(
            parameter,
            likelihood_by_variable,
            initialization_policy="stationary",
            observation_intercept_policy="free",
            equilibrium_forcing=False,
        ):
            parameters.append(parameter)
    statistical_model_spec = StatisticalModelSpec.model_validate(
        {
            "mechanisms": skeleton.mechanisms,
            "likelihoods": list(reversed(likelihoods)),
            "parameters": parameters,
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
        }
    )
    artifact = compile_ssm_artifact(
        statistical_model_spec,
        make_prior_plan(statistical_model_spec, {}),
        structural_plan=plan,
    )

    payload = artifact.model_dump(mode="json")
    assert payload["schema_version"] == 2
    assert "structure" in payload
    assert "spec" not in payload
    assert artifact.spec.manifest_names == ["x_obs", "y_obs"]
    assert {
        (binding.source_kind, binding.target_kind) for binding in artifact.structure.bindings
    } == {
        ("state", "latent_state"),
        ("manifest", "manifest_channel"),
        ("edge", "dynamics_edge"),
    }
    assert [
        certificate.construct_name for certificate in artifact.structure.anchor_certificates
    ] == [
        "X",
        "Y",
    ]
