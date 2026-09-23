import type { ScenarioClamp, ScenarioRequest } from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult } from "@/lib/dag/simulation-types";

/** Interactive scenarios always select an explicit outcome. */
export type SimulateInput = ScenarioRequest;

/**
 * Runs a scenario and returns its result. The interactive DAG is agnostic to
 * how: production injects a `POST /api/tools/dispatch` call (the non-LLM tool
 * seam). Absent this, the DAG is a read-only viewer.
 */
export type SimulateFn = (input: SimulateInput) => Promise<AnalysisSimulationResult>;

/** Re-run `base`'s scenario start with a new set of clamps over the same horizon. */
export function buildSimulateInput(
  base: AnalysisSimulationResult,
  clamps: [ScenarioClamp, ...ScenarioClamp[]],
  horizonDays: number,
): SimulateInput {
  return {
    start: { ...base.request.start },
    clamps,
    outcome: base.request.outcome,
    readout: { estimand: "trajectory", horizon_days: horizonDays, projection: "latent" },
  };
}
