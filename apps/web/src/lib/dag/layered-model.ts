import { modelConstructs } from "@/lib/model-accessors";
import { indexModel } from "@/lib/model-asset/entities";
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

/** Resolve the backend's graph selection without changing the scientific definition. */
export function graphEntities(model: ModelSnapshot) {
  const indexed = indexModel(model.model?.value);
  const constructs = model.findings.graph.construct_ids.map((id) => indexed.constructById.get(id)!);
  return {
    constructs,
    edges: model.findings.graph.edge_ids.map((id) => indexed.edgeById.get(id)!),
    indicators: constructs.flatMap((construct) => construct.indicators),
  };
}

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
    fit: model.findings.fit?.source.validity === "fresh",
    simulation: simulation != null,
  };
  return CAUSAL_GRAPH_LAYER_ORDER.filter((layer) => available[layer]);
}

import type { DagLayoutNode } from "@/lib/utils/dag-graph-layout";

export interface GraphBand {
  key: "static" | "history" | "present";
  label: string;
  nodes: DagLayoutNode[];
}

export function boundsForBand(
  band: GraphBand,
): { x: number; y: number; width: number; height: number } | null {
  if (band.nodes.length === 0) return null;
  const minimumX = Math.min(...band.nodes.map((node) => node.x));
  const minimumY = Math.min(...band.nodes.map((node) => node.y));
  const maximumX = Math.max(...band.nodes.map((node) => node.x + node.width));
  const maximumY = Math.max(...band.nodes.map((node) => node.y + node.height));
  return {
    x: minimumX - 14,
    y: minimumY - 28,
    width: maximumX - minimumX + 28,
    height: maximumY - minimumY + 42,
  };
}
