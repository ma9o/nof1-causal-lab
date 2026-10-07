import type { ConstructId, ModelSnapshot, SimulationReport } from "@nof1-causal-lab/api-types";
import { modelConstructs, presentEntries } from "@/lib/model-accessors";
import type { ConstructStatus } from "./construct-statuses";
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

/** Display every construct and directed edge in the saved scientific model. */
export function graphEntities(indexed: ModelEntities) {
  return {
    constructs: indexed.constructs,
    dynamicConstructIds: indexed.constructs
      .filter((construct) => construct.temporal_status === "time_varying")
      .map((construct) => construct.id),
    edges: indexed.edges,
    indicators: indexed.indicators,
  };
}

/** Overlay saved identification findings on explicitly observed or latent constructs. */
export function graphStatus(model: ModelSnapshot, id: ConstructId): ConstructStatus | null {
  const construct = modelConstructs(model.model).find((item) => item.id === id);
  if (!construct) return null;
  const blocking = presentEntries(model.identification?.treatments ?? {}).some(
    ([target, finding]) =>
      finding.status === "not_identified" && (target === id || finding.confounders.includes(id)),
  );
  return blocking ? "blocking" : construct.indicators.length > 0 || construct.role === "exogenous" ? "observed" : "latent";
}

/** Layer visibility reflects facts in the selected revision, including partial models. */
export function availableGraphLayers(
  model: ModelSnapshot,
  simulation?: SimulationReport | null,
): CausalGraphLayerId[] {
  const available = {
    structure: modelConstructs(model.model).length > 0,
    measurement: modelConstructs(model.model).some((construct) => construct.indicators.length > 0),
    design: model.identification !== null,
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
