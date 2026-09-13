import type { SimulateScenarioResult } from "@nof1-causal-lab/api-types";

/** Read the same query/result envelope from a live tool output or serialized trace. */
export function parseSimulationResult(output: unknown): SimulateScenarioResult | null {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return null;
    }
  }
  if (typeof value !== "object" || value === null) return null;
  const candidate = value as Partial<SimulateScenarioResult>;
  const { query, evaluation, result } = candidate;
  return typeof query === "object" &&
    query !== null &&
    typeof query.id === "string" &&
    Array.isArray(query.clamps) &&
    query.clamps.length > 0 &&
    typeof evaluation === "object" &&
    evaluation !== null &&
    evaluation.query_id === query.id &&
    typeof evaluation.id === "string" &&
    typeof result === "object" &&
    result !== null &&
    result.evaluation_id === evaluation.id &&
    typeof result.outcome_label === "string" &&
    typeof result.summary === "object" &&
    result.summary !== null &&
    typeof result.start === "object" &&
    result.start !== null
    ? (value as SimulateScenarioResult)
    : null;
}
