import type { InterventionSpec, ScientificActionRequest } from "@nof1-causal-lab/api-types";
import type { SimulationWithEffects } from "@/lib/simulation-report";

export type SimulateInput = Extract<ScientificActionRequest, { action?: "simulate" }>;
export type SimulateFn = (input: SimulateInput) => Promise<SimulationWithEffects>;

/** Re-run the selected model and resolved start with dated interventions. */
export function buildSimulateInput(
  base: SimulationWithEffects,
  interventions: InterventionSpec[],
  end: number,
): SimulateInput {
  return {
    action: "simulate",
    model_revision: base.model.revision,
    start: base.times[0],
    end,
    interventions,
  };
}
