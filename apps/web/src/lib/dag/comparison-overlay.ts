import type { ConstructId, ConstructSpec, ModelDiffReport } from "@nof1-causal-lab/api-types";
import type { DagLayoutNode, Point } from "@/lib/utils/dag-graph-layout";
import { humanize } from "@/lib/model-asset/selection";
import { ghostId } from "@/lib/dag/unroll";
import {
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  type LayeredGraphBundle,
} from "@/lib/dag/build-layered-causal-graph";

type Change = "added" | "removed" | "revised";

interface DifferenceMark {
  id: string;
  change: Change;
  x: number;
  y: number;
  title: string;
  detail: string;
}

/** Place additions around the selected layout. Existing coordinates are never recomputed. */
export function placeComparisonOverlay(
  comparison: ModelDiffReport | null,
  topology: LayeredGraphBundle,
  nodes: DagLayoutNode[],
  width: number,
  height: number,
) {
  const dynamicIds = new Set(comparison?.graph.after_dynamic_construct_ids);
  const previousDynamicIds = new Set(comparison?.graph.before_dynamic_construct_ids);
  const positions = new Map(nodes.map((node) => [node.id, node]));
  const constructs = new Map(comparison?.graph.constructs.map((item) => [item.construct_id, item]));
  const edgeChanges = new Map<string, Change>();
  const constructChanges = new Map<ConstructId, Change>();
  const addedNodes: Array<{ node: DagLayoutNode; construct: ConstructSpec; history: boolean }> = [];
  const addedEdges: Array<{ id: string; points: Point[]; crossSlice: boolean }> = [];
  const marks: DifferenceMark[] = [];
  let overlayWidth = width,
    overlayHeight = height;

  const addNode = (construct: ConstructSpec, history = false) => {
    const id = history ? ghostId(construct.id) : construct.id;
    if (positions.has(id)) return;
    const node = {
      id,
      x: width + 60,
      y: 20 + addedNodes.length * (LAYERED_NODE_HEIGHT + 40),
      width: history ? LAYERED_HISTORY_WIDTH : LAYERED_NODE_WIDTH,
      height: history ? LAYERED_HISTORY_HEIGHT : LAYERED_NODE_HEIGHT,
    };
    positions.set(id, node);
    addedNodes.push({ node, construct, history });
    overlayWidth = Math.max(overlayWidth, node.x + node.width);
    overlayHeight = Math.max(overlayHeight, node.y + node.height);
  };
  const addEdge = (id: string, sourceId: string, targetId: string, crossSlice: boolean) => {
    const source = positions.get(sourceId)!,
      target = positions.get(targetId)!;
    const start = { x: source.x + source.width, y: source.y + source.height / 2 };
    const end = { x: target.x, y: target.y + target.height / 2 };
    const middle = (start.x + end.x) / 2;
    addedEdges.push({
      id,
      crossSlice,
      points: [start, { x: middle, y: start.y }, { x: middle, y: end.y }, end],
    });
    return { id, x: middle, y: (start.y + end.y) / 2, width: 0, height: 0 };
  };
  for (const item of comparison?.graph.constructs ?? []) {
    if (item.change === "unchanged") continue;
    constructChanges.set(item.construct_id, item.change);
    if (item.after) {
      if (!item.before) addNode(item.after);
      if (dynamicIds.has(item.construct_id) && !previousDynamicIds.has(item.construct_id)) {
        addNode(item.after, true);
        addEdge(`self:${item.construct_id}`, ghostId(item.construct_id), item.construct_id, true);
      }
    }
    if (previousDynamicIds.has(item.construct_id) && !dynamicIds.has(item.construct_id)) {
      edgeChanges.set(`self:${item.construct_id}`, "removed");
    }
    const exclusion = item.change === "removed" ? item.after_disposition : null;
    const node = positions.get(item.construct_id)!;
    marks.push({
      id: item.construct_id,
      x: node.x + node.width,
      y: node.y,
      change: item.change,
      title: `${exclusion ? "Excluded" : item.change === "revised" ? "Changed" : humanize(item.change)} construct`,
      detail: exclusion
        ? `${humanize(exclusion.disposition)}: ${exclusion.reason}`
        : item.change === "revised"
          ? dynamicIds.has(item.construct_id)
            ? "History node and persistence added"
            : "History node and persistence removed"
          : humanize((item.after ?? item.before)!.name),
    });
  }
  for (const item of comparison?.graph.edges ?? []) {
    if (item.change === "unchanged") continue;
    edgeChanges.set(item.edge_id, item.change);
    const existing = topology.edgeMeta.get(item.edge_id);
    let anchor = existing ? positions.get(existing.slotId) : undefined;
    if (item.after) {
      const cause = constructs.get(item.after.cause.id)!.after!;
      const sourceId = dynamicIds.has(cause.id) ? ghostId(cause.id) : cause.id;
      if (sourceId !== cause.id) addNode(cause, true);
      if (!existing || existing.source !== sourceId || existing.target !== item.after.effect.id) {
        if (existing) edgeChanges.set(item.edge_id, "removed");
        anchor = addEdge(item.edge_id, sourceId, item.after.effect.id, dynamicIds.has(cause.id));
      }
    }
    if (anchor)
      marks.push({
        id: item.edge_id,
        x: anchor.x + anchor.width / 2,
        y: anchor.y + anchor.height / 2,
        change: item.change,
        title: `${item.change === "revised" ? "Rerouted" : humanize(item.change)} connection`,
        detail:
          item.change === "removed" && item.after_disposition
            ? `${humanize(item.after_disposition.disposition)}: ${item.after_disposition.reason}`
            : (item.after ?? item.before)!.description,
      });
  }
  return {
    width: overlayWidth,
    height: overlayHeight,
    addedNodes,
    addedEdges,
    edgeChanges,
    constructChanges,
    marks,
  };
}
