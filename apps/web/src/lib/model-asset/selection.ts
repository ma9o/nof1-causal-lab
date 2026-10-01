import {
  type ConstructId,
  type EdgeId,
  type IndicatorId,
  type ObservationSpec,
} from "@nof1-causal-lab/api-types";

/** The graph entity shown in the details pane. */
export type EntitySelection =
  | { kind: "construct"; id: ConstructId }
  | { kind: "edge"; id: EdgeId }
  | { kind: "indicator"; id: IndicatorId };

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
