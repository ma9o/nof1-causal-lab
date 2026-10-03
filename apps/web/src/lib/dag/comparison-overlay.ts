import type { ConstructId, ConstructSpec } from "@nof1-causal-lab/api-types";
import type { ResolvedModelDiff } from "@/lib/hooks/use-model-diff";
import type { DagLayoutNode, Point } from "@/lib/utils/dag-graph-layout";
import { modelConstructs } from "@/lib/model-accessors";
import { humanize } from "@/lib/model-asset/selection";
import { ghostId, isGhost } from "@/lib/dag/unroll";
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
  comparison: ResolvedModelDiff | null,
  topology: LayeredGraphBundle,
  nodes: DagLayoutNode[],
  width: number,
  height: number,
) {
  const dynamicIds = new Set(comparison?.after_dynamic_construct_ids);
  const previousDynamicIds = new Set(comparison?.before_dynamic_construct_ids);
  const positions = new Map(nodes.map((node) => [node.id, node]));
  const beforeConstructs = new Map(
    (comparison ? modelConstructs(comparison.beforeModel) : []).map((item) => [item.id, item]),
  );
  const constructs = new Map(
    (comparison ? modelConstructs(comparison.afterModel) : []).map((item) => [item.id, item]),
  );
  const beforeEdges = new Map(comparison?.beforeModel.edges.map((item) => [item.id, item]));
  const afterEdges = new Map(comparison?.afterModel.edges.map((item) => [item.id, item]));
  const afterDispositions = new Map(
    comparison?.after_dispositions.map((item) => [item.target.id, item]),
  );
  const edgeChanges = new Map<string, Change>();
  const constructChanges = new Map<ConstructId, Change>();
  const addedNodes: Array<{ node: DagLayoutNode; construct: ConstructSpec; history: boolean }> = [];
  const addedEdges: Array<{ id: string; points: Point[]; crossSlice: boolean }> = [];
  const marks: DifferenceMark[] = [];
  let overlayWidth = width,
    overlayHeight = height;

  const addNode = (construct: ConstructSpec, history = false) => {
    const id = history ? ghostId(construct.id) : construct.id;
    const existing = positions.get(id);
    if (existing) return existing;
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
    return node;
  };
  const addEdge = (
    id: string,
    source: DagLayoutNode,
    target: DagLayoutNode,
    crossSlice: boolean,
  ) => {
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
  for (const change of comparison?.constructs ?? []) {
    const id = change.kind === "removed" ? change.before.id : change.after.id;
    if (change.kind === "unchanged") continue;
    constructChanges.set(id, change.kind);
    const definition = (change.kind === "removed" ? beforeConstructs : constructs).get(id);
    if (!definition) throw new Error(`Compared construct ${id} is missing from its model`);
    const node = change.kind === "added" ? addNode(definition) : positions.get(id);
    if (!node) continue;
    if (change.kind !== "removed") {
      if (dynamicIds.has(id) && !previousDynamicIds.has(id)) {
        const source = addNode(definition, true);
        addEdge(`self:${id}`, source, node, true);
      }
    }
    if (previousDynamicIds.has(id) && !dynamicIds.has(id)) {
      edgeChanges.set(`self:${id}`, "removed");
    }
    const exclusion = change.kind === "removed" ? afterDispositions.get(id) : null;
    marks.push({
      id: id,
      x: node.x + node.width,
      y: node.y,
      change: change.kind,
      title: `${exclusion ? "Excluded" : change.kind === "revised" ? "Changed" : humanize(change.kind)} construct`,
      detail: exclusion
        ? `${humanize(exclusion.disposition)}: ${exclusion.reason}`
        : change.kind === "revised"
          ? dynamicIds.has(id)
            ? "History node and persistence added"
            : "History node and persistence removed"
          : humanize(definition.name),
    });
  }
  for (const change of comparison?.edges ?? []) {
    const id = change.kind === "removed" ? change.before.id : change.after.id;
    if (change.kind === "unchanged") continue;
    edgeChanges.set(id, change.kind);
    const definition = (change.kind === "removed" ? beforeEdges : afterEdges).get(id);
    if (!definition) throw new Error(`Compared edge ${id} is missing from its model`);
    const existing = topology.edgeMeta.get(id);
    let anchor = existing ? positions.get(existing.slotId) : undefined;
    if (change.kind !== "removed") {
      const cause = constructs.get(definition.cause.id);
      const target = positions.get(definition.effect.id);
      const source = cause
        ? dynamicIds.has(cause.id)
          ? addNode(cause, true)
          : positions.get(cause.id)
        : undefined;
      if (
        source &&
        target &&
        (!existing || existing.source !== source.id || existing.target !== target.id)
      ) {
        if (existing) edgeChanges.set(id, "removed");
        anchor = addEdge(id, source, target, isGhost(source.id));
      }
    }
    const exclusion = change.kind === "removed" ? afterDispositions.get(id) : null;
    if (anchor)
      marks.push({
        id: id,
        x: anchor.x + anchor.width / 2,
        y: anchor.y + anchor.height / 2,
        change: change.kind,
        title: `${change.kind === "revised" ? "Rerouted" : humanize(change.kind)} connection`,
        detail: exclusion
          ? `${humanize(exclusion.disposition)}: ${exclusion.reason}`
          : definition.description,
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
