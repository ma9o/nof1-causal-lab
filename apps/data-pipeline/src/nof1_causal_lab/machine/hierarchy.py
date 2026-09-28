"""Four scientific actions with optional implementation and authoring contexts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from nof1_causal_lab.artifacts.identity import ArtifactId, ScientificActionId  # noqa: TC001

ContextLayer = Literal["navigator", "registry", "machine", "delegated", "tool"]


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    layer: ContextLayer
    label: str
    parent_id: str | None = None
    owns: tuple[ArtifactId, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    runtime_state: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    action_id: ScientificActionId
    description: str
    consumes: tuple[ArtifactId, ...] = ()
    optional_consumes: tuple[ArtifactId, ...] = ()
    produces: tuple[ArtifactId, ...] = ()
    produces_optional: tuple[ArtifactId, ...] = ()
    derives: tuple[ArtifactId, ...] = ()


CONTEXTS: tuple[ContextSpec, ...] = (
    ContextSpec(
        context_id="scientific",
        layer="tool",
        label="Scientific actions",
        parent_id="action-registry",
        allowed_tools=("edit_model", "prepare_data", "fit", "simulate", "poll_action"),
    ),
    ContextSpec(
        context_id="navigator",
        layer="navigator",
        label="LLM navigator",
    ),
    ContextSpec(
        context_id="action-registry",
        layer="registry",
        label="Transport-independent action contracts",
        parent_id="navigator",
    ),
    ContextSpec(
        context_id="episode-machine",
        layer="machine",
        label="Durable scientific action executor",
        parent_id="action-registry",
    ),
    ContextSpec(
        context_id="latent-structure",
        layer="tool",
        label="Structural definition validation",
        parent_id="action-registry",
        allowed_tools=("validate_latent_structure",),
    ),
    ContextSpec(
        context_id="measurement-structure",
        layer="tool",
        label="Measurement definition validation",
        parent_id="action-registry",
        allowed_tools=("validate_measurement_structure",),
    ),
    ContextSpec(
        context_id="statistical-model-spec",
        layer="tool",
        label="Statistical definition tools",
        parent_id="action-registry",
        allowed_tools=("search_literature",),
    ),
    ContextSpec(
        context_id="ingestion",
        layer="delegated",
        label="Ingestion file/code loop",
        parent_id="episode-machine",
        owns=("raw_data",),
        allowed_tools=("list_files", "read_file_sample", "execute_python", "submit_table"),
        runtime_state=("prepared_input_dir", "sandbox", "result_df", "column_descriptions"),
    ),
    ContextSpec(
        context_id="measurement",
        layer="delegated",
        label="Indicator extraction worker fan-out",
        parent_id="episode-machine",
        owns=("panel",),
        allowed_tools=("validate_extractions",),
        runtime_state=("indicator_plan", "worker_statuses", "extracted_values"),
    ),
    ContextSpec(
        context_id="inference",
        layer="delegated",
        label="Exact nonlinear SSM inference job",
        parent_id="episode-machine",
        owns=("model",),
        runtime_state=("sampler_config", "diagnostics", "conditioned_model"),
    ),
    ContextSpec(
        context_id="analysis",
        layer="tool",
        label="Runtime causal queries",
        parent_id="episode-machine",
        allowed_tools=("get_model_info",),
        runtime_state=("identified_treatments", "effect_summaries"),
    ),
)


ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "edit_model",
        "Revise scientific definitions and current laws; run or reuse applicable specification, identification, compatibility and exact predictive checks.",
        optional_consumes=("model", "panel"),
        produces=("model",),
        derives=("identification_report", "validation_report"),
    ),
    ActionSpec(
        "prepare_data",
        "Prepare uploaded files with scoring instructions and semantic workers, or materialize one recorded simulation replicate; return model-independent observations, metadata and numerical data checks.",
        optional_consumes=("raw_data", "model"),
        produces_optional=("raw_data", "panel"),
        derives=("data_profile", "validation_report"),
    ),
    ActionSpec(
        "fit",
        "Condition the selected model on selected observations; return joint uncertainty and fitting diagnostics.",
        consumes=("model", "panel"),
        produces=("model",),
    ),
    ActionSpec(
        "simulate",
        "Generate requested quantities from current model uncertainty and measure the shared predictive batch.",
        consumes=("model",),
        optional_consumes=("panel",),
    ),
)

ACTIONS_BY_ID = {action.action_id: action for action in ACTIONS}


def describe_contexts() -> list[dict[str, object]]:
    return [asdict(context) for context in CONTEXTS]


def describe_actions() -> list[dict[str, object]]:
    return [asdict(action) for action in ACTIONS]
