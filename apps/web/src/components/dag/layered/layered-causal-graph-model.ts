import { modelConstructs } from "@/lib/model-accessors";
import type { ModelSnapshot, SimulationResult } from "@nof1-causal-lab/api-types";

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
  simulation?: SimulationResult | null,
): CausalGraphLayerId[] {
  const available = {
    structure: (modelConstructs(model.model?.value).length ?? 0) > 0,
    measurement:
      modelConstructs(model.model?.value).some((construct) => construct.indicators.length > 0) ??
      false,
    design: (model.findings.dispositions?.value.length ?? 0) > 0,
    specification:
      modelConstructs(model.model?.value).some(
        (c) => c.dynamics.length > 0 || c.indicators.some((i) => i.likelihood != null),
      ) ?? false,
    fit: model.findings.fit != null,
    simulation: simulation != null,
  };
  return CAUSAL_GRAPH_LAYER_ORDER.filter((layer) => available[layer]);
}
