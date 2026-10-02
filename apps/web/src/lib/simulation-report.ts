import type { CausalEffectResult, SimulationReport } from "@nof1-causal-lab/api-types";

/** A presentation refinement of the canonical report after causal certification. */
export type SimulationWithEffects = SimulationReport & { causal_result: CausalEffectResult };

export function hasCausalEffects(report: SimulationReport): report is SimulationWithEffects {
  return report.causal_result != null;
}
