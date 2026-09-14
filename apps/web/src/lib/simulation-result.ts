import type { SimulationResult } from "@nof1-causal-lab/api-types";

/** Select current simulation responses from heterogeneous tool messages. */
export function parseSimulationResult(output: unknown): SimulationResult | null {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return null;
    }
  }
  if (typeof value !== "object" || value === null) return null;
  const candidate = value as Partial<SimulationResult>;
  return candidate.request != null &&
    Array.isArray(candidate.request.clamps) &&
    candidate.request.clamps.length > 0 &&
    candidate.model?.version != null &&
    Array.isArray(candidate.time_grid_days) &&
    candidate.labels != null &&
    candidate.trajectories != null &&
    candidate.summary != null
    ? (value as SimulationResult)
    : null;
}
