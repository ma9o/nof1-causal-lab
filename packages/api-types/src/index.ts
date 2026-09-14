// ---------------------------------------------------------------------------
// Hand-written (frontend-only) — not generated from Python
// ---------------------------------------------------------------------------

import type { ArtifactViewId } from "./transitions";

export type { ArtifactStatus, ArtifactViewState, PipelineRun, RunStatus } from "./run";
export type {
  ArtifactId,
  ArtifactViewId,
  PipelineSectionId,
  TransitionId,
  TransitionLogScopePolicy,
  TransitionMeta,
} from "./transitions";
export { ARTIFACT_IDS, ARTIFACT_VIEW_IDS, TRANSITION_META, TRANSITIONS } from "./transitions";

// ---------------------------------------------------------------------------
// Generated from Python, using the same type names in both languages
// ---------------------------------------------------------------------------

export type * from "./generated/models";

export type ArtifactViewDataMap = {
  [K in keyof import("./generated/models").ArtifactViews]: NonNullable<
    import("./generated/models").ArtifactViews[K]
  >;
};
export type ArtifactViewData<K extends ArtifactViewId = ArtifactViewId> = ArtifactViewDataMap[K];

// Distribution catalog metadata (codegen'd from Python)
export type { ObservationHyperparameter } from "./generated/metadata";
export {
  ARTIFACT_FILE_SPECS,
  MACHINE_DESCRIPTION,
  OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION,
} from "./generated/metadata";
// Tool definitions (codegen'd from Python ToolDefinition)
export type { ToolDefinition } from "./generated/tools";
export { CONTEXT_TOOLS, INTERACTIVE_CONTEXTS } from "./generated/tools";

export interface ArtifactData<T = unknown> {
  artifactId: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export { createModelClient } from "./client";
