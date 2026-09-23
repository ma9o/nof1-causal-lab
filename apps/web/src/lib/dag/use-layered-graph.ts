"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ConstructId,
  IndicatorSpec,
  ModelSnapshot,
  ModelComparison,
  SimulationResult,
} from "@nof1-causal-lab/api-types";
import type { DagLayoutNode } from "@/lib/utils/dag-graph-layout";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { useGraphControls, usePlayback } from "./use-graph-controls";
import { selectedNeighbors } from "./selection";
import { buildLayeredCausalGraph, type LayeredGraphEdgeMeta } from "./build-layered-causal-graph";
import {
  graphEntities,
  availableGraphLayers,
  type CausalGraphLayerId,
  type GraphBand,
} from "./layered-model";
import { placeComparisonOverlay } from "./comparison-overlay";
import { DAG_COLORS, COMPARISON_COLORS, BLOCKING, MARGINALIZED, signColor } from "./palette";
import { getSimulationDays } from "./simulation";

export interface LayeredGraphOptions {
  model: ModelSnapshot;
  simulation?: SimulationResult | null;
  comparison?: ModelComparison | null;
  selectedNode: ConstructId | null;
}

/** Derive graph display state from recorded model findings and user interaction. */
export function useLayeredGraph({
  model,
  simulation = null,
  comparison = null,
  selectedNode,
}: LayeredGraphOptions) {
  const available = useMemo(() => availableGraphLayers(model, simulation), [model, simulation]);
  const [hiddenLayers, setHiddenLayers] = useState<Set<CausalGraphLayerId>>(() => new Set());
  const { zoom, setZoom, hoveredEdge, setHoveredEdge } = useGraphControls(1, 0.08, 1.8);
  const paneRef = useRef<HTMLDivElement>(null);
  const [paneWidth, setPaneWidth] = useState(0);
  const visible = useMemo(
    () => new Set(available.filter((layer) => !hiddenLayers.has(layer))),
    [available, hiddenLayers],
  );

  const entities = useMemo(() => graphEntities(model), [model]);
  const topology = useMemo(
    () => buildLayeredCausalGraph(entities.constructs, entities.edges),
    [entities],
  );
  const { nodes, edges: routedSegments, width, height, isLayouting } = useDagLayout(topology.graph);
  const difference = useMemo(
    () => placeComparisonOverlay(isLayouting ? null : comparison, topology, nodes, width, height),
    [comparison, topology, nodes, width, height, isLayouting],
  );
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane || isLayouting) return;
    const measure = () => setPaneWidth(pane.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(pane);
    return () => observer.disconnect();
  }, [isLayouting]);

  const designVisible = visible.has("design");
  const specificationVisible = visible.has("specification");
  const fitVisible = visible.has("fit");
  const simulationVisible = visible.has("simulation");
  const nodeStatuses = new Map(
    entities.constructs.map((entity) => [
      entity.id,
      designVisible ? model.findings.graph_status[entity.id] : null,
    ]),
  );
  const edgeDispositions = new Map(
    entities.edges.map((entity) => [
      entity.id,
      designVisible
        ? model.findings.dispositions?.value.find((item) => item.target.id === entity.id)
            ?.disposition
        : undefined,
    ]),
  );
  const indicatorsByConstruct = new Map<ConstructId, IndicatorSpec[]>(
    visible.has("measurement")
      ? entities.constructs.map((construct) => [construct.id, construct.indicators])
      : [],
  );
  const likelihoodByVariable = new Map(
    specificationVisible
      ? entities.indicators.flatMap((indicator) =>
          indicator.likelihood ? [[indicator.id, indicator.likelihood] as const] : [],
        )
      : [],
  );
  const warningVariables = new Set(
    model.findings.simulation?.source.validity === "fresh"
      ? (model.findings.simulation?.value.predictive_checks?.per_variable_warnings ?? [])
          .filter((check) => !check.passed)
          .map((check) => check.indicator_id)
      : [],
  );
  const edgePosteriors = fitVisible ? (model.findings.fit?.value.edge_estimates ?? {}) : {};
  const persistencePosteriors = fitVisible ? (model.findings.fit?.value.decay_estimates ?? {}) : {};
  const maximumPosteriorMean = Math.max(
    0,
    ...Object.values(edgePosteriors).map((posterior) => Math.abs(posterior.mean)),
    ...Object.values(persistencePosteriors).map((posterior) => Math.abs(posterior.mean)),
  );

  const simulationResult = simulationVisible ? simulation : null;
  const days = useMemo(
    () => (simulationResult ? getSimulationDays(simulationResult) : []),
    [simulationResult],
  );
  const {
    index: clampedDayIndex,
    setIndex: setDayIndex,
    playing,
    setPlaying,
  } = usePlayback(days.length, 180);
  const currentDay = days[clampedDayIndex];
  const selectedNeighborhood = useMemo(
    () =>
      selectedNeighbors(
        selectedNode,
        entities.edges.map((edge) => [edge.cause.id, edge.effect.id] as const),
      ),
    [entities.edges, selectedNode],
  );

  const graphBands = useMemo<GraphBand[]>(() => {
    const staticNodes: DagLayoutNode[] = [];
    const historyNodes: DagLayoutNode[] = [];
    const presentNodes: DagLayoutNode[] = [];
    for (const node of nodes) {
      const meta = topology.nodeMeta.get(node.id);
      if (meta?.kind === "history") historyNodes.push(node);
      if (meta?.kind === "construct") {
        if (meta.construct.temporal_status === "time_invariant") staticNodes.push(node);
        else presentNodes.push(node);
      }
    }
    return [
      { key: "static", label: "stable context", nodes: staticNodes },
      { key: "history", label: "t−1", nodes: historyNodes },
      { key: "present", label: "t", nodes: presentNodes },
    ];
  }, [nodes, topology.nodeMeta]);

  const toggleLayer = (layer: CausalGraphLayerId) => {
    if (layer === "structure") return;
    setHiddenLayers((current) => {
      const next = new Set(current);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  };

  const edgeVisual = (meta: LayeredGraphEdgeMeta) => {
    const disposition = meta.isSelf
      ? nodeStatuses.get(meta.cause) === "marginalized"
        ? "projected_edge"
        : undefined
      : edgeDispositions.get(meta.id as import("@nof1-causal-lab/api-types").EdgeId);
    const posterior = meta.isSelf ? persistencePosteriors[meta.cause] : edgePosteriors[meta.id];
    const activeClamp =
      currentDay != null &&
      simulationResult?.request.clamps.some(
        (clamp) =>
          clamp.target === meta.effect &&
          clamp.from_day <= currentDay &&
          (clamp.to_day == null || currentDay < clamp.to_day),
      );
    const blocking =
      nodeStatuses.get(meta.cause) === "blocking" || nodeStatuses.get(meta.effect) === "blocking";
    const marginalized =
      nodeStatuses.get(meta.cause) === "marginalized" ||
      nodeStatuses.get(meta.effect) === "marginalized";
    const color = activeClamp
      ? DAG_COLORS.pruned
      : blocking
        ? BLOCKING
        : marginalized
          ? MARGINALIZED
          : posterior
            ? signColor(posterior.mean)
            : disposition === "projected_edge"
              ? DAG_COLORS.muted
              : meta.lagged
                ? DAG_COLORS.lagged
                : DAG_COLORS.contemporaneous;
    const width = posterior
      ? 1.5 +
        (maximumPosteriorMean > 0 ? (Math.abs(posterior.mean) / maximumPosteriorMean) * 3.5 : 0)
      : 1.7;
    const selectedEdge =
      selectedNode == null || meta.cause === selectedNode || meta.effect === selectedNode;
    const dimmed = !selectedEdge || (hoveredEdge != null && hoveredEdge !== meta.id);
    const opacity = dimmed
      ? 0.1
      : activeClamp
        ? 0.45
        : marginalized || disposition === "projected_edge"
          ? 0.38
          : posterior
            ? 0.92
            : 0.78;
    const change = difference.edgeChanges.get(meta.id);
    return {
      disposition,
      posterior,
      activeClamp: Boolean(activeClamp),
      color: change ? COMPARISON_COLORS[change] : color,
      width: change ? Math.max(width, 3 / zoom) : width,
      opacity: change ? 1 : opacity,
      dimmed: change ? false : dimmed,
      change,
    };
  };

  return {
    available,
    visible,
    hoveredEdge,
    setHoveredEdge,
    zoom,
    setZoom,
    paneRef,
    paneWidth,
    topology,
    nodes,
    routedSegments,
    width,
    isLayouting,
    difference,
    designVisible,
    specificationVisible,
    fitVisible,
    simulationVisible,
    nodeStatuses,
    indicatorsByConstruct,
    likelihoodByVariable,
    warningVariables,
    persistencePosteriors,
    simulationResult,
    days,
    clampedDayIndex,
    setDayIndex,
    playing,
    setPlaying,
    currentDay,
    selectedNeighborhood,
    graphBands,
    toggleLayer,
    edgeVisual,
  };
}
