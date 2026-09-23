"use client";

import { useCallback, useMemo, useState } from "react";
import type {
  CausalEdgeSpec,
  ConstructSpec,
  PosteriorEstimate,
  IndicatorSpec,
} from "@nof1-causal-lab/api-types";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { useGraphControls, usePlayback } from "./use-graph-controls";
import { baseId } from "./unroll";
import { buildSimulationGraph } from "./build-simulation-graph";
import { getSimulationDays } from "./simulation";
import type { AnalysisSimulationResult } from "./simulation-types";
import type { ConstructStatus } from "./construct-statuses";
import { buildSimulateInput, type SimulateFn } from "./simulate-input";

export interface InteractiveGraphOptions {
  constructs: ConstructSpec[];
  edges: CausalEdgeSpec[];
  indicators?: IndicatorSpec[];
  edgePosteriors?: Record<string, PosteriorEstimate>;
  persistencePosteriors?: Record<string, PosteriorEstimate>;
  identifiableTreatments?: string[];
  result: AnalysisSimulationResult;
  height?: number;
  onSimulate?: SimulateFn;
  /** Controlled indicator visibility. Omit to retain the DAG's local toggle. */
  indicatorsVisible?: boolean;
  /** Identification status keeps marginalized theory nodes visible in the fitted graph. */
  nodeStatuses?: Record<string, ConstructStatus>;
  onNodeClick?: (constructName: string) => void;
}

export function useInteractiveGraph({
  constructs,
  edges,
  indicators = [],
  edgePosteriors = {},
  persistencePosteriors = {},
  identifiableTreatments = [],
  result,
  onSimulate,
  indicatorsVisible,
}: InteractiveGraphOptions) {
  const outcome = result.labels[result.request.outcome];
  const [dir, setDir] = useState<"LR" | "TB">("LR");
  const [localShowIndicators, setLocalShowIndicators] = useState(false);
  const showIndicators = indicatorsVisible ?? localShowIndicators;
  const {
    zoom,
    setZoom: setZoomClamped,
    hoveredEdge: hoverEdge,
    setHoveredEdge: setHoverEdge,
  } = useGraphControls(1, 0.4, 2.5);

  // Derive-from-props: a new scenario (result) resets in-progress do() editing.
  const [prevResult, setPrevResult] = useState(result);
  const [currentResult, setCurrentResult] = useState(result);
  if (prevResult !== result) {
    setPrevResult(result);
    setCurrentResult(result);
  }

  const days = useMemo(() => getSimulationDays(currentResult), [currentResult]);
  const n = days.length;
  const { index: clampedDay, setIndex: setDay, playing, setPlaying } = usePlayback(n, 110, 12);

  const byName = useMemo(() => new Map(constructs.map((c) => [c.name, c])), [constructs]);
  const { graph, edgeMeta } = useMemo(
    () =>
      buildSimulationGraph(constructs, edges, {
        dir: dir === "LR" ? "RIGHT" : "DOWN",
        showIndicators,
        showUnroll: true,
        indicators,
        persistenceNodes: Object.keys(persistencePosteriors),
      }),
    [constructs, edges, dir, showIndicators, indicators, persistencePosteriors],
  );
  const { nodes, edges: routed, width: W, height: H, isLayouting } = useDagLayout(graph);

  const identifiableTreatmentSet = useMemo(
    () => new Set(identifiableTreatments),
    [identifiableTreatments],
  );
  // Active interventions belong to the current resolved query.
  const interventions = currentResult.request.clamps;
  const maximumPosteriorMean = useMemo(
    () =>
      Math.max(
        0,
        ...Object.values(edgePosteriors).map(({ mean }) => Math.abs(mean)),
        ...Object.values(persistencePosteriors).map(({ mean }) => Math.abs(mean)),
      ),
    [edgePosteriors, persistencePosteriors],
  );

  const setDo = useCallback(
    async (node: string, value: number) => {
      const fromDay = days[clampedDay];
      const horizonDay = days[n - 1];
      if (!onSimulate || fromDay == null || horizonDay == null) return;
      const horizonDays = Math.max(horizonDay, 1);
      const res = await onSimulate(
        buildSimulateInput(
          result,
          [
            {
              target: constructs.find((c) => c.name === node)!.id,
              mode: "set",
              value,
              from_day: fromDay,
            },
          ],
          horizonDays,
        ),
      );
      setCurrentResult(res);
    },
    [onSimulate, days, clampedDay, n, result, constructs],
  );
  const resetScenario = useCallback(() => setCurrentResult(result), [result]);

  // column bands — shade alternate real-node layers
  const columnBands = useMemo(() => {
    const reals = nodes.filter((nd) => !nd.id.startsWith("G__"));
    const cols = new Map<number, { min: number; max: number }>();
    for (const nd of reals) {
      const key = Math.round(dir === "LR" ? nd.x : nd.y);
      const span =
        dir === "LR" ? { lo: nd.x, hi: nd.x + nd.width } : { lo: nd.y, hi: nd.y + nd.height };
      const cur = cols.get(key);
      if (cur) {
        cur.min = Math.min(cur.min, span.lo);
        cur.max = Math.max(cur.max, span.hi);
      } else cols.set(key, { min: span.lo, max: span.hi });
    }
    return [...cols.entries()]
      .sort((p, q) => p[0] - q[0])
      .map(([, v]) => v)
      .filter((_, i) => i % 2 === 1);
  }, [nodes, dir]);

  const hoverEndpoints = hoverEdge ? hoverEdge.split(">").map(baseId) : [];

  return {
    outcome,
    dir,
    setDir,
    showIndicators,
    setLocalShowIndicators,
    zoom,
    setZoomClamped,
    hoverEdge,
    setHoverEdge,
    currentResult,
    days,
    n,
    clampedDay,
    setDay,
    playing,
    setPlaying,
    byName,
    edgeMeta,
    nodes,
    routed,
    W,
    H,
    isLayouting,
    identifiableTreatmentSet,
    interventions,
    maximumPosteriorMean,
    setDo,
    resetScenario,
    columnBands,
    hoverEndpoints,
  };
}
