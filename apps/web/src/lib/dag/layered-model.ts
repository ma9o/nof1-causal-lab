import type { ModelSnapshot, SimulationReport } from "@nof1-causal-lab/api-types";
import { modelConstructs } from "@/lib/model-accessors";
import type { ModelEntities } from "@/lib/model-asset/entities";

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
export function graphEntities(model: ModelSnapshot, indexed: ModelEntities) {
  const selectedConstructs = new Set(model.graph.construct_ids);
  const selectedEdges = new Set(model.graph.edge_ids);
  const constructs = indexed.constructs.filter((construct) => selectedConstructs.has(construct.id));
  return {
    constructs,
    dynamicConstructIds: model.graph.dynamic_construct_ids,
    edges: indexed.edges.filter((edge) => selectedEdges.has(edge.id)),
    indicators: constructs.flatMap((construct) => construct.indicators),
  };
}

/** Layer visibility reflects facts in the selected revision, including partial models. */
export function availableGraphLayers(
  model: ModelSnapshot,
  simulation?: SimulationReport | null,
): CausalGraphLayerId[] {
  const available = {
    structure: modelConstructs(model.model).length > 0,
    measurement: modelConstructs(model.model).some((construct) => construct.indicators.length > 0),
    design: (model.dispositions?.length ?? 0) > 0,
    specification: modelConstructs(model.model).some(
      (c) => c.dynamics.length > 0 || c.indicators.some((i) => i.likelihood != null),
    ),
    fit: model.fit !== null,
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
