import {
  type ScientificActionId,
  type ArtifactId,
  type ConstructId,
  type EdgeId,
  type IndicatorId,
  type ParameterId,
  type ObservationSpec,
} from "@nof1-causal-lab/api-types";

/** The model entity or recorded version shown in the details pane. */
export type ModelSelection =
  | { kind: "revision"; seq: number }
  | { kind: "construct"; id: ConstructId }
  | { kind: "edge"; id: EdgeId }
  | { kind: "indicator"; id: IndicatorId }
  | { kind: "parameter"; id: ParameterId };

export type EntitySelection = Exclude<ModelSelection, { kind: "revision" }>;

/** Short action labels for every artifact the machine can install. */
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

export function formatFillNull(
  observation: Pick<ObservationSpec, "fill_null" | "fill_null_limit">,
): string {
  if (observation.fill_null == null) return "None";
  const method = String(observation.fill_null);
  return observation.fill_null_limit == null
    ? method
    : `${method} (limit ${observation.fill_null_limit})`;
}

export function formatSigned(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export function formatPlain(value: number, digits = 2): string {
  return `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;
}

export function actionLabel(action: ScientificActionId): string {
  const labels: Record<ScientificActionId, string> = {
    edit_model: "Edit model",
    prepare_data: "Prepare data",
    fit: "Fit",
    simulate: "Simulate",
  };
  return labels[action];
}
