import type {
  ConstructId,
  ConstructSpec,
  ModelComparison,
  ParameterSpec,
} from "@nof1-causal-lab/api-types";
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

function decision(parameter: ParameterSpec | null): string {
  if (!parameter) return "absent";
  if (parameter.value != null) return `pinned ${parameter.value}`;
  return parameter.distribution ? "free" : "unspecified";
}

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
  comparison: ModelComparison | null,
  topology: LayeredGraphBundle,
  nodes: DagLayoutNode[],
  width: number,
  height: number,
) {
  const positions = new Map(nodes.map((node) => [node.id, node]));
  const constructs = new Map(comparison?.graph.constructs.map((item) => [item.construct_id, item]));
  const edgeChanges = new Map<string, Change>();
  const constructChanges = new Map<ConstructId, Change>();
  const addedNodes: Array<{ node: DagLayoutNode; construct: ConstructSpec; history: boolean }> = [];
  const addedEdges: Array<{ id: string; points: Point[]; lagged: boolean }> = [];
  const marks: DifferenceMark[] = [];
  const parameters = new Map(comparison?.parameters.map((item) => [item.parameter_id, item]));
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
  const addEdge = (id: string, sourceId: string, targetId: string, lagged: boolean) => {
    const source = positions.get(sourceId)!,
      target = positions.get(targetId)!;
    const start = { x: source.x + source.width, y: source.y + source.height / 2 };
    const end = { x: target.x, y: target.y + target.height / 2 };
    const middle = (start.x + end.x) / 2;
    addedEdges.push({
      id,
      lagged,
      points: [start, { x: middle, y: start.y }, { x: middle, y: end.y }, end],
    });
    return { id, x: middle, y: (start.y + end.y) / 2, width: 0, height: 0 };
  };
  const detail = (
    ids: ModelComparison["graph"]["edges"][number]["parameter_ids"],
    text: string,
  ) => {
    if (ids.length === 1) {
      const parameter = parameters.get(ids[0])!;
      return `${decision(parameter.before)} → ${decision(parameter.after)}`;
    }
    return ids.length ? `${ids.length} parameters changed` : text;
  };

  for (const item of comparison?.graph.constructs ?? []) {
    if (item.change === "unchanged") continue;
    constructChanges.set(item.construct_id, item.change);
    if (!item.before && item.after) {
      addNode(item.after);
      if (item.after.role === "endogenous" && item.after.temporal_status === "time_varying") {
        addNode(item.after, true);
        addEdge(`self:${item.construct_id}`, ghostId(item.construct_id), item.construct_id, true);
      }
    }
    if (item.change === "removed") edgeChanges.set(`self:${item.construct_id}`, "removed");
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
        : detail(
            item.parameter_ids,
            item.before && item.after && item.before.name !== item.after.name
              ? `${humanize(item.before.name)} → ${humanize(item.after.name)}`
              : humanize((item.after ?? item.before)!.name),
          ),
    });
  }
  for (const item of comparison?.graph.edges ?? []) {
    if (item.change === "unchanged") continue;
    edgeChanges.set(item.edge_id, item.change);
    const existing = topology.edgeMeta.get(item.edge_id);
    let anchor = existing ? positions.get(existing.slotId) : undefined;
    if (item.after) {
      const cause = constructs.get(item.after.cause.id)!.after!;
      const sourceId =
        item.after.lagged && cause.temporal_status === "time_varying"
          ? ghostId(cause.id)
          : cause.id;
      if (sourceId !== cause.id) addNode(cause, true);
      if (!existing || existing.source !== sourceId || existing.target !== item.after.effect.id) {
        if (existing) edgeChanges.set(item.edge_id, "removed");
        anchor = addEdge(item.edge_id, sourceId, item.after.effect.id, item.after.lagged);
      }
    }
    if (anchor)
      marks.push({
        id: item.edge_id,
        x: anchor.x + anchor.width / 2,
        y: anchor.y + anchor.height / 2,
        change: item.change,
        title: item.parameter_ids.length
          ? "Parameter change"
          : `${item.change === "revised" ? "Changed" : humanize(item.change)} connection`,
        detail:
          item.change === "removed" && item.after_disposition
            ? `${humanize(item.after_disposition.disposition)}: ${item.after_disposition.reason}`
            : detail(item.parameter_ids, (item.after ?? item.before)!.description),
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
