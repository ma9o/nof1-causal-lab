"""Owning modules, semantic roles, and descriptions for exported API contracts."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeAliasType

from pydantic import BaseModel

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue

# Group exported types by their owning subject.
CONCERNS = {
    "scientific_model": (
        "Scientific model",
        (
            "artifacts.model_spec",
            "artifacts.construct",
            "artifacts.evidence",
            "artifacts.indicator",
            "artifacts.data_preparation",
            "artifacts.observations",
            "artifacts.predictive_provenance",
            "artifacts.likelihood",
            "artifacts.mechanism",
            "measurement_types",
            "utils.observation_semantics",
            "utils.window_expressions",
            "artifacts.parameter_spec",
            "artifacts.expressions",
            "artifacts.parameter",
            "distributions",
            "numpyro_json",
            "artifacts.raw_data",
            "artifacts.measurements",
            "artifacts.validation_report",
            "artifacts.identification",
            "artifacts.prior",
            "artifacts.execution",
            "artifacts.posterior",
            "artifacts.posterior_diagnostics",
            "artifacts.effects",
            "artifacts.scenarios",
            "artifacts.simulation",
            "artifacts.checks",
            "artifacts.model_checks",
        ),
    ),
    "execution_history": (
        "Execution & history",
        (
            "study.artifact_files",
            "study.state",
            "actions.effects",
            "actions.status",
            "study.store",
            "study.records",
            "actions.progress",
        ),
    ),
    "read_models": (
        "Read models",
        ("study.snapshot_models", "study.view_models", "study.visual_models"),
    ),
    "api_tools": (
        "API & tool contracts",
        (
            "study_api",
            "actions.contracts",
            "actions.results",
            "actions.revisions",
            "actions.data_diff",
            "tool_contracts",
            "json_types",
            "utils.llm",
        ),
    ),
    "identity": ("Shared identities & references", ("artifacts.identity",)),
}

# Aliases and dataclasses need explicit role sentences: JSON Schema does not carry
# their Python docstrings. These describe concepts, never infer prose from field names.
ROLE_SENTENCES = {
    "ActionPoll": "A poll is either running labels or a completed typed attempt with its optional publication identity.",
    "ActionAttempt": "A closed action attempt pairs its request with only that action's successful result or failure outcome.",
    "ActionBody": "An action result carries its owned scientific payload before Git publication.",
    "FailedOutcome": "A failed outcome is an expected rejection or an opaque execution failure.",
    "RejectionReason": "A rejection reason identifies the expected input or publication condition that prevented the action.",
    "CheckGroup": "A group of model checks is selected by the inputs it consumes.",
    "PredictiveCheckReason": "A predictive check reason explains why a battery could not be evaluated for the selected model and observations.",
    "GitOid": "A native Git object identity for an immutable tree or commit.",
    "CoefficientRole": "A coefficient role identifies an expression operand’s scientific quantity and support.",
    "BinaryOperator": "A binary operator combines two scalar expression operands.",
    "ExpressionFunction": "An expression function transforms scalar operands or constructs structured observation arguments.",
    "Expression": "A scalar expression composes supported arithmetic with scientific state and coefficient references.",
    "NumPyroDistribution": "A native NumPyro probability distribution serialized by its constructor tree.",
    "DynamicsMechanismSpec": "A dynamics mechanism declares one contribution to continuous-time drift.",
    "ArtifactFreshness": "An artifact's presence and freshness are derived from the selected journal revision.",
    "ArtifactId": "An artifact identity selects one node in the study's artifact graph.",
    "ConstructId": "A persistent construct identity survives changes to its display name.",
    "EdgeId": "A persistent edge identity identifies one authored causal relationship.",
    "EffectTrajectoryPoint": "An effect trajectory point records a causal delta at one rollout time.",
    "EntityRef": "An entity reference identifies a construct, edge, indicator, or mechanism by its persistent identity.",
    "IndicatorId": "A persistent indicator identity survives changes to its measurement label.",
    "MechanismId": "A persistent mechanism identity distinguishes additive terms through reordering and revision.",
    "ParameterId": "A scientific parameter identity connects component coefficients to one parameter definition.",
    "ParameterElementId": "A parameter element identity identifies a logical scalar component across model revisions.",
    "JsonArray": "A JSON array transports an ordered collection of recursively typed values.",
    "JsonObject": "A JSON object transports string-keyed recursively typed values.",
    "JsonScalar": "A JSON scalar transports a string, number, boolean, or null.",
    "JsonValue": "A JSON value transports a scalar or a recursive array or object.",
    "MeasurementDtype": "A measurement dtype defines the observed value domain of an indicator.",
    "ProgressEvent": "A progress event records one running attempt's step status or extraction telemetry.",
    "ScientificActionId": "A scientific action identity selects model editing, data preparation, fitting, or simulation.",
    "SimulationReport": "A simulation report records forward histories, resolved execution settings, and certified effects when supported.",
    "Sourced": "A sourced value pairs one model finding with its supporting artifact revision.",
    "ToolError": "A tool error reports why a requested operation could not produce a result.",
    "ToolQuerySpec": "A tool query specification declares a context's callable query.",
}

ALIAS_MODULES = {
    "NumPyroDistribution": "numpyro_json",
    "EntityRef": "artifacts.identity",
    "SimulationReport": "artifacts.simulation",
    "ProgressEvent": "actions.progress",
}


def _module_for(name: str) -> str:
    if name in ALIAS_MODULES:
        return "nof1_causal_lab." + ALIAS_MODULES[name]
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("nof1_causal_lab.") or module is None:  # ty: ignore[redundant-condition-strict] - sys.modules can contain None import sentinels at runtime.
            continue
        value = vars(module).get(name)
        if value is not None and getattr(value, "__module__", None) == module_name:
            return module_name
        for value in tuple(vars(module).values()):
            if (
                isinstance(value, TypeAliasType)
                and value.__module__ == module_name
                and name.startswith(value.__name__ + "_")
            ):
                return module_name
            if (
                isinstance(value, type)
                and issubclass(value, BaseModel)
                and value.__module__ == module_name
                and value.__pydantic_generic_metadata__["parameters"]
                and name.startswith(value.__name__ + "_")
            ):
                return module_name
    raise ValueError(f"Exported type {name} has no declared Python owner")


def _layer_for(name: str, module: str) -> str:
    if module.endswith("artifacts.identity") or name == "ParameterCoordinate":
        return "identity"
    if module.endswith(("study.snapshot_models", "study.view_models")):
        return "read_models"
    if module.startswith("nof1_causal_lab.study.") or module.endswith("actions.progress"):
        return "study"
    if module.endswith(("study_api", "json_types", "utils.llm", "numpyro_json")):
        return "transport"
    if module.endswith("artifacts.scenarios"):
        return (
            "findings"
            if name.endswith(("Result", "Visualization", "Point", "Trajectory"))
            else "authored"
        )
    if module.endswith("artifacts.simulation"):
        return "authored" if name in {"SimulationSpec"} else "findings"
    if module.endswith(("artifacts.checks", "artifacts.model_checks")):
        return "findings"
    if name == "FitSettingsSpec":
        return "authored"
    if name.endswith("Artifact"):
        return "artifacts"
    if module.endswith(
        (
            "artifacts.posterior",
            "artifacts.posterior_diagnostics",
            "artifacts.effects",
            "artifacts.validation_report",
            "artifacts.prior",
            "artifacts.identification",
        )
    ) or name in {
        "IdentificationReport",
        "IdentifiedTreatmentStatus",
        "NonIdentifiableTreatmentStatus",
        "TreatmentIdentification",
        "InferenceMetadata",
        "ObservationRecord",
    }:
        return "findings"
    if module.startswith("nof1_causal_lab.artifacts.") or module.endswith(
        (
            "distributions",
            "measurement_types",
            "utils.observation_semantics",
            "utils.window_expressions",
        )
    ):
        return "authored"
    if module.startswith("nof1_causal_lab.actions."):
        return "transport"
    raise ValueError(f"Exported type {name} in {module} has no conceptual layer")


def _concern_for(name: str, module: str) -> str:
    if name == "ParameterCoordinate":
        return "identity"
    for concern, (_, modules) in CONCERNS.items():
        if module.removeprefix("nof1_causal_lab.") in modules:
            return concern
    raise ValueError(f"Exported type {name} in {module} has no declared concern")


def annotate_definitions(definitions: dict[str, dict[str, JsonValue]]) -> None:
    """Require an owning Python module, role, and concern for every exported type."""
    for name, definition in definitions.items():
        module = (
            str(definition["x-python-module"])
            if "x-python-module" in definition
            else _module_for(name)
        )
        definition["x-python-module"] = module
        definition["x-layer"] = _layer_for(name, module)
        definition["x-concern"] = _concern_for(name, module)
        if name in ROLE_SENTENCES:
            definition["description"] = ROLE_SENTENCES[name]
        # Pydantic's specialized Sourced[T] definitions do not carry its docstring.
        if module.endswith("study.snapshot_models") and name.startswith("Sourced_"):
            definition["description"] = ROLE_SENTENCES["Sourced"]
