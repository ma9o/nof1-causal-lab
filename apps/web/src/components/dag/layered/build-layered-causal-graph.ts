import type { CausalEdge, Construct, ConstructId, EdgeId } from "@nof1-causal-lab/api-types";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { splitEdgesWithGlyphs, ghostId, isGhost } from "../unroll";

export const LAYERED_NODE_WIDTH = 250;
export const LAYERED_NODE_HEIGHT = 132;
export const LAYERED_HISTORY_WIDTH = 156;
export const LAYERED_HISTORY_HEIGHT = 54;
export const LAYERED_EDGE_SLOT_WIDTH = 78;
export const LAYERED_EDGE_SLOT_HEIGHT = 32;

export type LayeredGraphNodeMeta =
  | { kind: "construct"; construct: Construct }
  | { kind: "history"; construct: Construct }
  | { kind: "edge_slot"; edgeId: string };

export interface LayeredGraphEdgeMeta {
  id: EdgeId | `self:${ConstructId}`;
  cause: ConstructId;
  effect: ConstructId;
  source: string;
  target: string;
  lagged: boolean;
  isSelf: boolean;
  slotId: string;
}

export interface LayeredGraphSegmentMeta {
  edgeId: string;
  markerEnd: boolean;
}

export interface LayeredGraphBundle {
  graph: DagGraphInput;
  nodeMeta: Map<string, LayeredGraphNodeMeta>;
  edgeMeta: Map<string, LayeredGraphEdgeMeta>;
  segmentMeta: Map<string, LayeredGraphSegmentMeta>;
}

const partition = (value: 0 | 1 | 2): Record<string, string> => ({
  "elk.partitioning.partition": String(value),
});

function constructPartition(construct: Construct): 0 | 2 {
  return construct.temporal_status === "time_invariant" ? 0 : 2;
}

/**
 * Build the permanent graph geometry exclusively from LatentStructure.
 * Every later artifact layer receives this same graph and can only decorate it.
 */
export function buildLayeredCausalGraph(
  constructs: Construct[],
  edges: CausalEdge[],
): LayeredGraphBundle {
  const constructById = new Map(constructs.map((construct) => [construct.id, construct] as const));
  const historyById = new Map(constructs.map((construct) => [ghostId(construct.id), construct]));
  for (const edge of edges) {
    if (!constructById.has(edge.cause_id) || !constructById.has(edge.effect_id)) {
      throw new Error(
        `Causal edge '${edge.cause_id}→${edge.effect_id}' references an unknown construct.`,
      );
    }
  }

  const timeVaryingIds = new Set(
    constructs
      .filter((construct) => construct.temporal_status === "time_varying")
      .map((construct) => construct.id),
  );
  const selfDynamicConstructs = constructs.filter(
    (construct) => construct.role === "endogenous" && construct.temporal_status === "time_varying",
  );
  const causalLinks = edges
    .filter((edge) => edge.cause_id !== edge.effect_id)
    .map((edge) => ({
      ...edge,
      source:
        edge.lagged && timeVaryingIds.has(edge.cause_id) ? ghostId(edge.cause_id) : edge.cause_id,
      target: edge.effect_id,
    }));
  const ghosts = new Set([
    ...causalLinks.filter((edge) => isGhost(edge.source)).map((edge) => edge.source),
    ...selfDynamicConstructs.map((construct) => ghostId(construct.id)),
  ]);

  const edgeDefinitions: Array<Omit<LayeredGraphEdgeMeta, "slotId">> = [
    ...causalLinks.map((edge) => ({
      id: edge.id,
      cause: edge.cause_id,
      effect: edge.effect_id,
      source: edge.source,
      target: edge.target,
      lagged: edge.lagged,
      isSelf: false,
    })),
    ...selfDynamicConstructs.map((construct) => ({
      id: `self:${construct.id}` as const,
      cause: construct.id,
      effect: construct.id,
      source: ghostId(construct.id),
      target: construct.id,
      lagged: true,
      isSelf: true,
    })),
  ];

  const split = splitEdgesWithGlyphs(
    edgeDefinitions.map((edge) => ({
      a: edge.source,
      b: edge.target,
      isSelf: edge.isSelf,
      lagged: edge.lagged,
    })),
    { width: LAYERED_EDGE_SLOT_WIDTH, height: LAYERED_EDGE_SLOT_HEIGHT },
  );

  const nodeMeta = new Map<string, LayeredGraphNodeMeta>();
  const nodes: DagGraphInput["nodes"] = [];
  for (const construct of constructs) {
    nodeMeta.set(construct.id, { kind: "construct", construct });
    nodes.push({
      id: construct.id,
      width: LAYERED_NODE_WIDTH,
      height: LAYERED_NODE_HEIGHT,
      layoutOptions: partition(constructPartition(construct)),
    });
  }
  for (const ghost of ghosts) {
    const construct = historyById.get(ghost);
    if (!construct) {
      throw new Error(`Temporal copy '${ghost}' has no source construct.`);
    }
    nodeMeta.set(ghost, { kind: "history", construct });
    nodes.push({
      id: ghost,
      width: LAYERED_HISTORY_WIDTH,
      height: LAYERED_HISTORY_HEIGHT,
      layoutOptions: partition(1),
    });
  }

  const edgeMeta = new Map<string, LayeredGraphEdgeMeta>();
  const segmentMeta = new Map<string, LayeredGraphSegmentMeta>();
  split.glyphNodes.forEach((slot, index) => {
    const definition = edgeDefinitions[index];
    const sourceConstruct = constructById.get(definition.cause);
    const targetConstruct = constructById.get(definition.effect);
    if (!sourceConstruct || !targetConstruct) {
      throw new Error(`Edge slot '${slot.id}' has an unknown endpoint.`);
    }
    const sourcePartition = definition.source.endsWith("__p")
      ? 1
      : constructPartition(sourceConstruct);
    const targetPartition = constructPartition(targetConstruct);
    const slotPartition =
      sourcePartition < targetPartition
        ? Math.max(sourcePartition, targetPartition - 1)
        : sourcePartition;

    nodes.push({
      ...slot,
      layoutOptions: partition(slotPartition as 0 | 1 | 2),
    });
    nodeMeta.set(slot.id, { kind: "edge_slot", edgeId: definition.id });
    edgeMeta.set(definition.id, { ...definition, slotId: slot.id });
    segmentMeta.set(`e${index}s`, { edgeId: definition.id, markerEnd: false });
    segmentMeta.set(`e${index}t`, { edgeId: definition.id, markerEnd: true });
  });

  return {
    graph: {
      nodes,
      edges: split.edges,
      direction: "RIGHT",
      layoutOptions: {
        "elk.partitioning.activate": "true",
        "elk.layered.spacing.nodeNodeBetweenLayers": "54",
        "elk.spacing.nodeNode": "30",
        "elk.spacing.edgeNode": "24",
        "elk.spacing.edgeEdge": "14",
        "elk.layered.spacing.edgeNodeBetweenLayers": "26",
      },
    },
    nodeMeta,
    edgeMeta,
    segmentMeta,
  };
}
