"""Conceptual placement for every exported contract, including named scalar references."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject

LAYERS = {
    "identity": ("Identity", "#dcfce7"),
    "authored": ("Authored values", "#dbeafe"),
    "artifacts": ("Artifact payloads", "#ede9fe"),
    "findings": ("Derived findings", "#fef3c7"),
    "machine": ("Machine records", "#fee2e2"),
    "read_models": ("Read models", "#cffafe"),
    "transport": ("Transport", "#e2e8f0"),
}

# Sections group types by subject; layer colors continue to describe their role.
CONCERNS = {
    "scientific_model": (
        "Scientific model",
        (
            "artifacts.model_spec",
            "artifacts.construct",
            "artifacts.evidence",
            "artifacts.indicator",
            "artifacts.likelihood",
            "artifacts.mechanism",
            "measurement_types",
            "utils.observation_semantics",
            "artifacts.parameter_spec",
            "artifacts.expressions",
            "artifacts.parameter",
            "distributions",
            "numpyro_json",
            "artifacts.raw_data",
            "artifacts.measurements",
            "artifacts.validation_report",
            "artifacts.identification",
            "artifacts.prior_predictive",
            "artifacts.prior",
            "artifacts.execution",
            "artifacts.posterior",
            "artifacts.posterior_diagnostics",
            "artifacts.effects",
            "artifacts.scenarios",
        ),
    ),
    "execution_provenance": (
        "Execution & provenance",
        (
            "machine.artifact_files",
            "machine.artifacts",
            "machine.graph",
            "machine.hierarchy",
            "machine.moves",
            "machine.status",
            "machine.store",
            "flows.runtime_events",
        ),
    ),
    "read_models": (
        "Read models",
        ("machine.snapshot_models", "machine.view_models"),
    ),
    "api_tools": (
        "API & tool contracts",
        ("episode_api", "flows.transitions.analysis.contracts", "json_types", "utils.llm"),
    ),
    "identity": ("Shared identities & references", ("artifacts.identity",)),
}

# Diagram emphasis is curated by scientific meaning, independent of graph degree
# or contract layer. UI projections of the same central objects share the emphasis.
CORE_DOMAIN_OBJECTS = {
    "ModelSpec": "The canonical scientific definition, enriched through stable entities.",
    "ObservationRecord": "An observed measurement supplying evidence to the model.",
    "ConstructSpec": "A scientific variable whose causal relationships are modeled.",
    "CausalEdgeSpec": "A directed causal assumption between constructs.",
    "IndicatorSpec": "The measurement definition connecting a construct to observed data.",
    "IdentificationReport": "The identification evidence required for causal reporting.",
    "LikelihoodSpec": "The observation distribution linking model states to measurements.",
    "DynamicsMechanismSpec": "The explicit scientific contribution to continuous-time drift.",
    "ParameterSpec": "A referenced scientific parameter with a native prior law or fixed value.",
    "InferenceReport": "Inference telemetry and predictive findings recorded in the transition log.",
    "ScenarioClamp": "The intervention applied to a specific construct over time.",
    "ModelSnapshot": "Independently sourced canonical aggregates read at one committed revision.",
    "FitSummary": "The canonical posterior together with server-composed display findings.",
}

# Aliases and dataclasses need explicit role sentences: JSON Schema does not carry
# their Python docstrings. These describe concepts, never infer prose from field names.
ROLE_SENTENCES = {
    "CoefficientRole": "A coefficient role identifies an expression operand’s scientific quantity and support.",
    "BinaryOperator": "A binary operator combines two scalar expression operands.",
    "ExpressionFunction": "An expression function transforms scalar operands or constructs structured observation arguments.",
    "Expression": "A scalar expression composes supported arithmetic with scientific state and coefficient references.",
    "NumPyroDistribution": "A native NumPyro probability distribution serialized by its constructor tree.",
    "DynamicsMechanismSpec": "A dynamics mechanism declares one contribution to continuous-time drift.",
    "ActionSpec": "An action declares a machine operation and its interaction context.",
    "AggregationFunction": "An aggregation function summarizes observations within a measurement window.",
    "ArtifactFileSpec": "An artifact file specification declares its JSON payloads, tables, and executable binaries.",
    "ArtifactFreshness": "An artifact's presence and freshness are derived from the selected journal revision.",
    "ArtifactId": "An artifact identity selects one node in the machine's artifact graph.",
    "ConstructId": "A persistent construct identity survives changes to its display name.",
    "ContextSpec": "An interaction context declares the tools and machine actions available to an agent.",
    "EdgeId": "A persistent edge identity identifies one authored causal relationship.",
    "EffectTrajectoryPoint": "An effect trajectory point records a causal delta at one rollout time.",
    "EntityRef": "An entity reference identifies a construct, edge, indicator, or mechanism by its persistent identity.",
    "IndicatorId": "A persistent indicator identity survives changes to its measurement label.",
    "MechanismId": "A persistent mechanism identity distinguishes additive terms through reordering and revision.",
    "ParameterId": "A scientific parameter identity connects component coefficients to one parameter definition.",
    "ParameterElementId": "A parameter element identity identifies a logical scalar component across model revisions.",
    "JournalStatus": "A journal status distinguishes applied revisions from rejected or failed attempts.",
    "JsonArray": "A JSON array transports an ordered collection of recursively typed values.",
    "JsonObject": "A JSON object transports string-keyed recursively typed values.",
    "JsonScalar": "A JSON scalar transports a string, number, boolean, or null.",
    "JsonValue": "A JSON value transports a scalar or a recursive array or object.",
    "MeasurementDtype": "A measurement dtype defines the observed value domain of an indicator.",
    "Move": "A machine move requests computation or an authored artifact write.",
    "Provenance": "Artifact provenance records whether its content was computed, authored by a human, or proposed by an LLM.",
    "Derivation": "A derivation declares an artifact maintained atomically with its input versions.",
    "Root": "A root declares an independently writable artifact and any contextual input pins.",
    "RunOperation": "A run move invokes an authoring or computation operation.",
    "OperationId": "An operation identity selects an action independently of its output artifacts.",
    "RuntimeEvent": "A runtime event records transition progress, agent activity, or extraction telemetry.",
    "ScenarioQueryInput": "A scenario readout requests an estimand, forward horizon, and output scale.",
    "ScenarioRequest": "A simulation request declares the initial state, timed clamps, and requested outcome readout.",
    "SimulateScenarioToolResult": "A simulation tool response carries either a resolved scenario result or a reported tool error.",
    "Sourced": "A sourced value pairs one model finding with its supporting artifact version.",
    "ToolError": "A tool error reports why a requested operation could not produce a result.",
    "ToolQuerySpec": "A tool query specification declares a context's callable query.",
    "WriteArtifact": "A write move requests a validated authored artifact revision.",
}

ALIAS_MODULES = {
    "NumPyroDistribution": "numpyro_json",
    "EntityRef": "artifacts.identity",
    "Move": "machine.moves",
    "RuntimeEvent": "flows.runtime_events",
}


def _module_for(name: str) -> str:
    if name in ALIAS_MODULES:
        return "nof1_causal_lab." + ALIAS_MODULES[name]
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("nof1_causal_lab.") or module is None:
            continue
        value = vars(module).get(name)
        if value is not None and getattr(value, "__module__", None) == module_name:
            return module_name
    raise ValueError(f"Exported type {name} has no declared Python owner")


def _layer_for(name: str, module: str) -> str:
    if module.endswith("artifacts.identity") or name == "ParameterCoordinate":
        return "identity"
    if module.endswith(("machine.snapshot_models", "machine.view_models")):
        return "read_models"
    if module.startswith("nof1_causal_lab.machine.") or module.endswith("flows.runtime_events"):
        return "machine"
    if module.endswith(("episode_api", "json_types", "utils.llm", "numpyro_json")):
        return "transport"
    if module.endswith("artifacts.scenarios"):
        return "findings" if name.endswith(("Result", "Visualization", "Point")) else "authored"
    if name.endswith("Artifact"):
        return "artifacts"
    if module.endswith(
        (
            "artifacts.posterior",
            "artifacts.posterior_diagnostics",
            "artifacts.effects",
            "artifacts.validation_report",
            "artifacts.prior",
            "artifacts.prior_predictive",
            "artifacts.identification",
        )
    ) or name in {
        "IdentificationReport",
        "IdentifiedTreatmentStatus",
        "NonIdentifiableTreatmentStatus",
        "TreatmentIdentification",
        "PriorPredictiveDiagnostic",
        "InferenceMetadata",
        "ObservationRecord",
    }:
        return "findings"
    if module.startswith("nof1_causal_lab.artifacts.") or module.endswith(
        ("distributions", "measurement_types", "utils.observation_semantics")
    ):
        return "authored"
    if module.startswith("nof1_causal_lab.flows."):
        return "transport"
    raise ValueError(f"Exported type {name} in {module} has no conceptual layer")


def _concern_for(name: str, module: str) -> str:
    if name == "ParameterCoordinate":
        return "identity"
    for concern, (_, modules) in CONCERNS.items():
        if module.removeprefix("nof1_causal_lab.") in modules:
            return concern
    raise ValueError(f"Exported type {name} in {module} has no declared concern")


def annotate_definitions(definitions: dict[str, JsonObject]) -> None:
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
        if module.endswith("machine.snapshot_models") and name.startswith("Sourced_"):
            definition["description"] = ROLE_SENTENCES["Sourced"]
