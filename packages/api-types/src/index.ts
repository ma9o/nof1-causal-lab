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

type Snapshot = import("./generated/models").ModelSnapshot;
type Value<T> = NonNullable<T> extends { value: infer V } ? V : never;
export type ArtifactViewDataMap = {
  raw_data: Value<Snapshot["data"]["raw_data"]>;
  model: Value<Snapshot["model"]>;
  measurements: Value<Snapshot["data"]["measurements"]>;
  validation_report: Value<Snapshot["findings"]["validation_report"]>;
  prior_predictive: Value<Snapshot["findings"]["prior_predictive"]>;
  model_diagnostics: NonNullable<Snapshot["findings"]["diagnostics"]>;
  inference_report: Value<Snapshot["findings"]["fit"]>["report"];
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
export type CellStatus = "ok" | "warning" | "error" | "not_evaluated";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export type { ScientificActionRequest } from "./client";
export { createModelClient } from "./client";
