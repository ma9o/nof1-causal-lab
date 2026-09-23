"use client";

import { useCallback, useMemo, useState } from "react";
import type { CausalEdgeSpec, ConstructSpec, IndicatorSpec } from "@nof1-causal-lab/api-types";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { DagDirection, DagGraphInput } from "@/lib/utils/dag-graph-layout";
import type { ConstructStatus } from "./construct-statuses";
import { useGraphControls } from "./use-graph-controls";
import { selectedNeighbors } from "./selection";
import {
  baseId,
  buildGhostLinks,
  DAG_LAYOUT_OPTIONS,
  type GlyphPair,
  splitEdgesWithGlyphs,
  unrollCausalLinks,
} from "./unroll";
import { NODE_W, NODE_W_WITH_INDICATORS, HEADER_H, nodeHeight } from "./structure-model";

export interface StructureGraphOptions {
  constructs: ConstructSpec[];
  outcomeId?: string | null;
  edges: CausalEdgeSpec[];
  indicators?: IndicatorSpec[];
  /**
   * Per-construct backend disposition/identifiability status. Colors the node border and any
   * incident edge: blocking → destructive, marginalized → warning.
   */
  nodeStatuses?: Record<string, ConstructStatus>;
  onNodeClick?: (constructName: string) => void;
  /** Initial layout flow direction. Defaults to cause → effect, left to right. */
  direction?: "RIGHT" | "DOWN";
}

export function useStructureGraph({
  constructs,
  edges,
  indicators,
  nodeStatuses,
  onNodeClick,
  direction = "RIGHT",
}: StructureGraphOptions) {
  const [selected, setSelected] = useState<string | null>(null);
  const [dir, setDir] = useState<DagDirection>(direction);
  const {
    zoom,
    setZoom: setZoomClamped,
    hoveredEdge: hoverEdgeId,
    setHoveredEdge: setHoverEdgeId,
  } = useGraphControls(0.69, 0.4, 2.5);

  const byId = useMemo(
    () => new Map(constructs.map((construct) => [construct.id, construct])),
    [constructs],
  );
  const byName = useMemo(() => new Map(constructs.map((c) => [c.name, c])), [constructs]);

  const indicatorsByConstruct = useMemo(() => {
    const selected = new Set((indicators ?? []).map((indicator) => indicator.id));
    return new Map(
      constructs.flatMap((construct) => {
        const owned = construct.indicators.filter((indicator) => selected.has(indicator.id));
        return owned.length ? [[construct.name, owned] as const] : [];
      }),
    );
  }, [constructs, indicators]);

  const nodeWidth = indicatorsByConstruct.size > 0 ? NODE_W_WITH_INDICATORS : NODE_W;

  // Lagged cross-construct edges start at the cause's t−1 copy, while
  // contemporaneous edges stay within t. Endogenous time-varying self-dynamics
  // are added separately. The spacer preserves the established layout rhythm.
  const { graph, glyphs } = useMemo(() => {
    const timeVaryingNames = new Set(
      constructs
        .filter((construct) => construct.temporal_status === "time_varying")
        .map((construct) => construct.name),
    );
    const causalLinks = unrollCausalLinks(
      edges
        .filter((edge) => edge.cause.id !== edge.effect.id)
        .map((edge) => ({
          cause: byId.get(edge.cause.id)!.name,
          effect: byId.get(edge.effect.id)!.name,
          lagged: edge.lagged,
        })),
      timeVaryingNames,
    );
    const selfLinks = buildGhostLinks(
      constructs
        .filter(
          (construct) =>
            construct.role === "endogenous" && construct.temporal_status === "time_varying",
        )
        .map((construct) => ({ from: construct.name, to: construct.name })),
    );
    const ghosts = new Set([...causalLinks.ghosts, ...selfLinks.ghosts]);
    const pairs: GlyphPair[] = [
      ...causalLinks.edges.map((edge) => ({
        a: edge.source,
        b: edge.target,
        isSelf: false,
        lagged: edge.lagged,
      })),
      ...selfLinks.edges.map((edge) => ({
        a: edge.source,
        b: edge.target,
        isSelf: true,
        lagged: true,
      })),
    ];
    const split = splitEdgesWithGlyphs(pairs);
    const built: DagGraphInput = {
      nodes: [
        ...constructs.map((c) => ({
          id: c.name,
          width: nodeWidth,
          height: nodeHeight(indicatorsByConstruct.get(c.name)?.length ?? 0),
        })),
        ...[...ghosts].map((ghost) => ({ id: ghost, width: nodeWidth, height: HEADER_H })),
        ...split.glyphNodes,
      ],
      edges: split.edges,
      direction: dir,
      layoutOptions: DAG_LAYOUT_OPTIONS,
    };
    return { graph: built, glyphs: split.glyphs };
  }, [constructs, edges, indicatorsByConstruct, nodeWidth, dir, byId]);

  const { nodes, edges: routed, width: W, height: H, isLayouting } = useDagLayout(graph);

  // Routed segment by id, to stitch each edge's `e<i>s` + `e<i>t` halves back together.
  const routedById = useMemo(() => new Map(routed.map((e) => [e.id, e])), [routed]);

  // The clicked construct plus its direct neighbors — everything else dims.
  const connected = useMemo(
    () =>
      selectedNeighbors(
        selected,
        edges.map(
          (edge) => [byId.get(edge.cause.id)!.name, byId.get(edge.effect.id)!.name] as const,
        ),
      ),
    [selected, edges, byId],
  );

  const hoverEndpoints = useMemo(() => {
    const meta = hoverEdgeId ? glyphs.get(hoverEdgeId) : null;
    return meta ? new Set([baseId(meta.a), baseId(meta.b)]) : new Set<string>();
  }, [hoverEdgeId, glyphs]);

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      const base = baseId(nodeId);
      setSelected((prev) => (prev === base ? null : base));
      onNodeClick?.(base);
    },
    [onNodeClick],
  );

  const hasGhosts = constructs.some(
    (construct) =>
      construct.temporal_status === "time_varying" &&
      (construct.role === "endogenous" ||
        edges.some((edge) => edge.lagged && byId.get(edge.cause.id)!.name === construct.name)),
  );
  const hasCrossLagged = edges.some(
    (edge) =>
      edge.lagged && byName.get(byId.get(edge.cause.id)!.name)?.temporal_status === "time_varying",
  );
  const hasContemporaneous = edges.some((edge) => !edge.lagged);
  const statusValues = nodeStatuses ? Object.values(nodeStatuses) : [];
  const hasMarginalized = statusValues.includes("marginalized");
  const hasBlocking = statusValues.includes("blocking");

  return {
    selected,
    setSelected,
    hoverEdgeId,
    setHoverEdgeId,
    dir,
    setDir,
    zoom,
    setZoomClamped,
    byName,
    indicatorsByConstruct,
    glyphs,
    nodes,
    W,
    H,
    isLayouting,
    routedById,
    connected,
    hoverEndpoints,
    handleNodeClick,
    hasGhosts,
    hasCrossLagged,
    hasContemporaneous,
    hasMarginalized,
    hasBlocking,
  };
}
