import type {
  Available,
  CausalEffectResult,
  ModelSnapshot,
  SimulationReport,
} from "@nof1-causal-lab/api-types";

/** A presentation refinement of the canonical report after causal certification. */
export type SimulationWithEffects = SimulationReport & { causal: Available<CausalEffectResult> };

export function hasCausalEffects(report: SimulationReport): report is SimulationWithEffects {
  return report.causal.kind === "available";
}

/** The simulation of the viewed model revision, certified or not; an older model's run is none. */
export function viewedSimulation(model: ModelSnapshot) {
  const simulation = model.simulation;
  return simulation?.evidence.model.revision === model.state.current.model?.revision
    ? simulation
    : null;
}
