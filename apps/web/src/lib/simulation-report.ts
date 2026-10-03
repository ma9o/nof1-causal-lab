import type { Available, CausalEffectResult, SimulationReport } from "@nof1-causal-lab/api-types";

/** A presentation refinement of the canonical report after causal certification. */
export type SimulationWithEffects = SimulationReport & { causal: Available<CausalEffectResult> };

export function hasCausalEffects(report: SimulationReport): report is SimulationWithEffects {
  return report.causal.kind === "available";
}
