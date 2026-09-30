"use client";

import type {
  ConstructId,
  EdgeId,
  IndicatorSpec,
  ModelDiffReport,
  ModelSnapshot,
  SimulationReport,
} from "@nof1-causal-lab/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { type LawCurve, lawCurves, ownLawUses } from "@/lib/model-asset/laws";
import type { DagLayoutNode } from "@/lib/utils/dag-graph-layout";
import { buildLayeredCausalGraph, type LayeredGraphEdgeMeta } from "./build-layered-causal-graph";
import { placeComparisonOverlay } from "./comparison-overlay";
import {
  availableGraphLayers,
  type CausalGraphLayerId,
  type GraphBand,
  graphEntities,
} from "./layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, MARGINALIZED } from "./palette";
import { selectedNeighbors } from "./selection";
import { useGraphControls, usePlayback } from "./use-graph-controls";

export interface LayeredGraphOptions {
  model: ModelSnapshot;
  /** A simulation of the viewed model revision; its node histories replace the law charts. */
  simulation?: SimulationReport | null;
  comparison?: ModelDiffReport | null;
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
    () =>
      buildLayeredCausalGraph(entities.constructs, entities.edges, entities.dynamicConstructIds),
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
      designVisible ? model.findings.graph.status[entity.id] : null,
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
    model.findings.predictive?.source.validity === "fresh"
      ? (model.findings.predictive?.value.predictive_checks?.per_variable_warnings ?? [])
          .filter((check) => !check.passed)
          .map((check) => check.indicator_id)
      : [],
  );
  const lawsVisible = specificationVisible || fitVisible;
  const constructLaws = useMemo(
    () =>
      new Map<ConstructId, LawCurve[]>(
        entities.constructs.map((construct) => [
          construct.id,
          lawCurves(model, ownLawUses(construct)),
        ]),
      ),
    [model, entities.constructs],
  );
  const edgeLaws = useMemo(
    () =>
      new Map<EdgeId, LawCurve[]>(
        entities.edges.map((edge) => [edge.id, lawCurves(model, ownLawUses(edge))]),
      ),
    [model, entities.edges],
  );
  const edgePosteriors = fitVisible ? (model.findings.fit?.value.edge_estimates ?? {}) : {};
  const persistencePosteriors = fitVisible ? (model.findings.fit?.value.decay_estimates ?? {}) : {};

  const simulationResult = simulationVisible ? simulation : null;
  const days = useMemo(() => simulationResult?.times ?? [], [simulationResult]);
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
    const dynamicIds = new Set(entities.dynamicConstructIds);
    const staticNodes: DagLayoutNode[] = [];
    const historyNodes: DagLayoutNode[] = [];
    const presentNodes: DagLayoutNode[] = [];
    for (const node of nodes) {
      const meta = topology.nodeMeta.get(node.id);
      if (meta?.kind === "history") historyNodes.push(node);
      if (meta?.kind === "construct") {
        if (!dynamicIds.has(meta.construct.id)) staticNodes.push(node);
        else presentNodes.push(node);
      }
    }
    return [
      { key: "static", label: "stable context", nodes: staticNodes },
      { key: "history", label: "t−1", nodes: historyNodes },
      { key: "present", label: "t", nodes: presentNodes },
    ];
  }, [nodes, topology.nodeMeta, entities.dynamicConstructIds]);

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
    const blocking =
      nodeStatuses.get(meta.cause) === "blocking" || nodeStatuses.get(meta.effect) === "blocking";
    const marginalized =
      nodeStatuses.get(meta.cause) === "marginalized" ||
      nodeStatuses.get(meta.effect) === "marginalized";
    const color = blocking
      ? BLOCKING
      : marginalized
        ? MARGINALIZED
        : disposition === "projected_edge"
          ? DAG_COLORS.muted
          : meta.crossSlice
            ? DAG_COLORS.crossSlice
            : DAG_COLORS.contemporaneous;
    // A coefficient's mean is not the strength or sign of a nonlinear state-dependent effect.
    const width = 1.7;
    const selectedEdge =
      selectedNode == null || meta.cause === selectedNode || meta.effect === selectedNode;
    const dimmed = !selectedEdge || (hoveredEdge != null && hoveredEdge !== meta.id);
    const opacity = dimmed
      ? 0.1
      : marginalized || disposition === "projected_edge"
        ? 0.38
        : posterior
          ? 0.92
          : 0.78;
    const change = difference.edgeChanges.get(meta.id);
    return {
      disposition,
      posterior,
      laws: meta.isSelf || !lawsVisible ? [] : (edgeLaws.get(meta.id as EdgeId) ?? []),
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
    constructLaws: lawsVisible ? constructLaws : new Map<ConstructId, LawCurve[]>(),
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
