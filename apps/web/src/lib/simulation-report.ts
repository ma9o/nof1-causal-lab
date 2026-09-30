import type { CausalEffectResult, SimulationReport } from "@nof1-causal-lab/api-types";

/** A presentation refinement of the canonical report after causal certification. */
export type SimulationWithEffects = SimulationReport & { causal_result: CausalEffectResult };

export function hasCausalEffects(report: SimulationReport): report is SimulationWithEffects {
  return report.causal_result != null;
}

/** Select materialized certified effects from heterogeneous tool messages. */
export function parseSimulationReport(output: unknown): SimulationWithEffects | null {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return null;
    }
  }
  if (typeof value !== "object" || value === null) return null;
  const report = value as Partial<SimulationReport>;
  const result = report.causal_result;
  return Array.isArray(report.design?.interventions) &&
    report.design.interventions.length > 0 &&
    report.model?.revision != null &&
    Array.isArray(report.times) &&
    Array.isArray(result?.effect_trajectory) &&
    result.outcome != null &&
    result.labels != null &&
    report.predictive?.states != null &&
    result.summary != null
    ? (value as SimulationWithEffects)
    : null;
}
