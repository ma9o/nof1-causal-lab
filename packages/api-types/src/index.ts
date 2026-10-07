// ---------------------------------------------------------------------------
// Generated from Python, using the same type names in both languages
// ---------------------------------------------------------------------------

// Distribution catalog metadata (codegen'd from Python)
export type { ObservationHyperparameter } from "./generated/metadata";
export { ARTIFACT_IDS, OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION } from "./generated/metadata";
export type * from "./generated/models";

export interface ArtifactData<T = unknown> {
  artifactId: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error" | "not_evaluated";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export { readNumericalArray } from "./arrays";
export { createModelClient } from "./client";

export { decodeModelMessage } from "./msgpack";
