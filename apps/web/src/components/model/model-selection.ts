import {
  TRANSITION_META,
  type Move,
  type ArtifactId,
  type ConstructId,
  type EdgeId,
  type IndicatorId,
} from "@nof1-causal-lab/api-types";

/** What the details pane is scoped to: the asset itself or one of its parts. */
export type ModelSelection =
  | { kind: "asset" }
  | { kind: "version"; seq: number }
  | { kind: "construct"; id: ConstructId }
  | { kind: "edge"; id: EdgeId }
  | { kind: "indicator"; id: IndicatorId }
  | { kind: "query"; key: string };

export const ASSET_SELECTION: ModelSelection = { kind: "asset" };

export const SCOPE_KIND_LABEL: Record<ModelSelection["kind"], string> = {
  asset: "asset",
  version: "version",
  construct: "construct",
  edge: "edge",
  indicator: "indicator",
  query: "query",
};

/** Short move labels for every artifact the machine can install. */
export const ARTIFACT_LABEL: Record<ArtifactId, string> = {
  raw_data: "Preprocess",
  model: "Model",
  identification_report: "Identification report",
  panel: "Panel",
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
  return move.kind === "run"
    ? TRANSITION_META[move.operation_id].label
    : ARTIFACT_LABEL[move.artifact_id];
}
