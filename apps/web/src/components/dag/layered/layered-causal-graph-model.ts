import type { ModelSnapshot, SimulateScenarioResult } from "@nof1-causal-lab/api-types";

export const CAUSAL_GRAPH_LAYER_ORDER = [
  "structure",
  "measurement",
  "design",
  "specification",
  "fit",
  "simulation",
] as const;
export type CausalGraphLayerId = (typeof CAUSAL_GRAPH_LAYER_ORDER)[number];

/** Layer visibility reflects facts in the selected revision, including partial models. */
export function availableGraphLayers(
  model: ModelSnapshot,
  simulation?: SimulateScenarioResult | null,
): CausalGraphLayerId[] {
  const available = {
    structure: model.latent_structure != null,
    measurement: model.measurement_structure != null,
    design: (model.dispositions?.value.length ?? 0) > 0,
    specification: model.state.current.statistical_model_spec != null,
    fit: model.fit != null,
    simulation: simulation != null,
  };
  return CAUSAL_GRAPH_LAYER_ORDER.filter((layer) => available[layer]);
}
