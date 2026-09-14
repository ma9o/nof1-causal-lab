"""Temporal activities for the baseline-report transition."""

from __future__ import annotations

import json
import logging
from typing import Any

from temporalio import activity

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import json_filename
from nof1_causal_lab.machine.derivations import complete_computed_transition
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.model_contracts import project_model_fields
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.activity_errors import (
    as_non_retryable_application_error,
)
from nof1_causal_lab.machine.temporal.latent_structure_activities import _llm_backend_config
from nof1_causal_lab.machine.temporal.llm_subroutine_storage import subroutine_root
from nof1_causal_lab.machine.temporal.messages import (
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import storage

logger = logging.getLogger(__name__)


def _write_baseline_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


def _read_baseline_json(path: str) -> Any:
    return storage.read_json(path)


def _first_baseline_assistant_summary(trace: UncheckedJsonObject) -> str | None:
    for message in trace.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return None


@activity.defn
async def plan_baseline_report_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.flows.transitions.analysis.interventions import run_interventions
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.models.causal_proofs import (
        CertifiedCausalAnalysis,
        certify_identified_estimand,
    )
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("baseline_report")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"

    from nof1_causal_lab.machine.inference import inference_is_current, inference_record
    from nof1_causal_lab.machine.store import EpisodeJournal

    if not inference_is_current(input.state):
        raise ValueError("Baseline reporting requires a current conditioned model")
    record = inference_record(EpisodeJournal(input.workspace_id).read_all(), pins["model"])
    if record is None:
        raise ValueError("Baseline reporting requires the committed inference log")
    diagnostics = record.diagnostics["report"]
    identification = IdentificationReport.model_validate(
        store.read_json_file(
            "identification_report",
            pins["identification_report"],
            json_filename("identification_report", "identification_report"),
        )
    )
    model_revision = ModelRevision(workspace_id=input.workspace_id, version=pins["model"])
    model = read_model(store, model_revision.version)
    constructs = {construct.id: construct for construct in model.constructs}
    treatments = [constructs[cid].name for cid in identification.estimable_treatments]
    if identification.outcome is None:
        raise ValueError("A baseline report requires an identified outcome")
    outcome_name = constructs[identification.outcome].name
    estimands = tuple(
        certify_identified_estimand(
            model,
            identification,
            model_revision=model_revision,
            treatment=treatment,
            outcome=outcome_name,
        )
        for treatment in treatments
    )
    analysis = CertifiedCausalAnalysis(
        model=model,
        model_revision=ModelRevision(
            workspace_id=input.workspace_id, version=model_revision.version
        ),
        identification=identification,
        estimands=estimands,
        inference=record,
    )

    logger.info("=== analysis: Treatment Effects ===")
    logger.info("Estimating effects of %d treatments on %s", len(treatments), outcome_name)
    intervention_results = run_interventions(analysis)

    ppc_warnings = [
        {
            "indicator_id": warning["indicator_id"],
            "issue_type": warning.get("issue_type"),
            "severity": warning.get("severity"),
            "message": warning.get("message"),
        }
        for warning in diagnostics["assessment"].get("ppc", {}).get("per_variable_warnings", [])
    ][:5]
    top_results = [
        {
            "treatment": entry.get("treatment"),
            "summary": entry["summary"],
        }
        for entry in intervention_results[:5]
    ]
    commentary_input = {
        "outcome": outcome_name,
        "identifiable_treatments": treatments,
        "excluded_non_identifiable_treatments": sorted(
            constructs[cid].name for cid in identification.non_identifiable
        ),
        "top_ranked_effects": top_results,
        "ppc_warnings": ppc_warnings,
        "follow_up_capabilities": {
            "get_model_info": (
                "Inspect variables, measurement, identifiability, diagnostics, and baseline effects."
            ),
            "simulate_intervention": (
                "Run Pearl rung-2 intervention simulations on the fitted generative model."
            ),
            "simulate_counterfactual": (
                "Run Pearl rung-3 counterfactual simulations conditioned on an observed history window."
            ),
        },
    }
    system_prompt = (
        "You are writing the opening commentary for analysis of a causal state-space "
        "analysis. Comment on the treatment-effect results for a technical user. "
        "Be concise and grounded. Do not invent certainty. Mention the strongest "
        "effects, note warnings or identifiability limits, and end by stating that "
        "follow-up chat can inspect model details or run Pearl rung 2 and rung 3 "
        "simulations. Return plain Markdown only."
    )
    user_prompt = (
        "Comment the results of analysis.\n\n"
        f"{json.dumps(commentary_input, indent=2, sort_keys=True)}"
    )
    context_ref = storage.join(
        subroutine_root(input.workspace_id, run_id, "baseline-report"),
        "context.json",
    )
    _write_baseline_json(
        context_ref,
        {
            "system_prompt": system_prompt,
            "user_messages": [user_prompt],
            "intervention_results": intervention_results,
        },
    )

    config = get_config()
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.analysis_commentary.llm, config.llm, None),
        max_tool_turns=1,
    )


@activity.defn
async def finalize_baseline_report_activity(
    input: SingleLLMTransitionFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.artifacts.baseline_report import BaselineReportArtifact

    try:
        context = _read_baseline_json(input.context_ref)
        trace = storage.read_json(input.trace_ref)
        payload: UncheckedJsonObject = {
            "intervention_results": context["intervention_results"],
        }
        final_summary = _first_baseline_assistant_summary(trace)
        if final_summary:
            payload["final_summary"] = final_summary
        payload = project_model_fields(BaselineReportArtifact, payload)

        store = ArtifactStore(input.workspace_id)
        produced = [
            store.write_version(
                "baseline_report",
                provenance="computed",
                derived_from=input.pins,
                produced_by="run:baseline_report",
                json_files={json_filename("baseline_report", "baseline_report"): payload},
            )
        ]
        return complete_computed_transition(store, input.state, "baseline_report", produced)
    except Exception as exc:
        raise as_non_retryable_application_error(exc) from exc


BASELINE_REPORT_ACTIVITIES = [
    plan_baseline_report_activity,
    finalize_baseline_report_activity,
]
