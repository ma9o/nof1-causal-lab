import {
  TRANSITION_META,
  type Move,
  type ArtifactId,
  type ConstructId,
  type EdgeId,
  type IndicatorId,
  type ParameterId,
} from "@nof1-causal-lab/api-types";

/** What the details pane is scoped to: the asset itself or one of its parts. */
export type ModelSelection =
  | { kind: "asset" }
  | { kind: "version"; seq: number }
  | { kind: "construct"; id: ConstructId }
  | { kind: "edge"; id: EdgeId }
  | { kind: "indicator"; id: IndicatorId }
  | { kind: "parameter"; id: ParameterId };

export type EditableSelection = Exclude<ModelSelection, { kind: "version" }>;

export const ASSET_SELECTION: ModelSelection = { kind: "asset" };

export const SCOPE_KIND_LABEL: Record<ModelSelection["kind"], string> = {
  asset: "asset",
  version: "action",
  construct: "construct",
  edge: "edge",
  indicator: "indicator",
  parameter: "parameter",
};

/** Short move labels for every artifact the machine can install. */
export const ARTIFACT_LABEL: Record<ArtifactId, string> = {
  raw_data: "Preprocess",
  model: "Model",
  identification_report: "Identification report",
  panel: "Panel",
  data_profile: "Data profile",
  validation_report: "Validation",
};

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

export function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export function formatPlain(value: number, digits = 2): string {
  return `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;
}

export function moveLabel(move: Move): string {
  if (move.kind === "write" && move.artifact_id === "model") return "Edit model";
  if (move.kind === "run") {
    if (move.operation_id === "posterior") return "Fit";
    if (move.operation_id === "simulate") return "Simulate";
  }
  return move.kind === "run"
    ? TRANSITION_META[move.operation_id].label
    : ARTIFACT_LABEL[move.artifact_id];
}
