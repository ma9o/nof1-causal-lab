"use client";

import type {
  ConstructSpec,
  ConstructId,
  IndicatorSpec,
  LikelihoodSpec,
  PosteriorEstimate,
} from "@nof1-causal-lab/api-types";
import { Pause, Play } from "lucide-react";
import type { KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { formatPosteriorIntervalLabel } from "@/lib/utils/format";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { COMPARISON_COLORS, DAG_COLORS, BLOCKING, MARGINALIZED } from "@/lib/dag/palette";
import { useLayeredGraph, type LayeredGraphOptions } from "@/lib/dag/use-layered-graph";
import { getNodeActionSeries, getNodeReferenceSeries } from "@/lib/dag/simulation";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import {
  LAYERED_EDGE_SLOT_HEIGHT,
  LAYERED_EDGE_SLOT_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  type LayeredGraphEdgeMeta,
} from "@/lib/dag/build-layered-causal-graph";
import { boundsForBand, type CausalGraphLayerId } from "@/lib/dag/layered-model";
import { LayeredComparisonOverlay } from "./layered-comparison-overlay";

const CANVAS_PADDING = 36;

const LAYER_LABELS: Record<CausalGraphLayerId, string> = {
  structure: "Structure",
  measurement: "Measurement",
  design: "Design",
  specification: "Specification",
  fit: "Fit",
  simulation: "Simulation",
};

export type LayeredCausalGraphVariant = "workbench" | "asset";

export interface LayeredCausalGraphProps extends LayeredGraphOptions {
  onSelectNode: (construct: ConstructId | null) => void;
  /**
   * `workbench` keeps the layer toggles, simulation timeline and legend around the canvas;
   * `asset` renders the canvas alone, filling its container, with only the zoom control.
   */
  variant?: LayeredCausalGraphVariant;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, Math.max(1, length - 1))}…` : value;
}

function activateOnKeyboard(event: KeyboardEvent<SVGGElement>, action: () => void): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function wrapDescription(value: string, lineLength = 40): [string, string] {
  const words = value.trim().split(/\s+/);
  let first = "";
  let index = 0;
  for (; index < words.length; index += 1) {
    const candidate = first ? `${first} ${words[index]}` : words[index];
    if (candidate.length > lineLength && first) break;
    first = candidate;
  }
  return [truncate(first, lineLength + 1), truncate(words.slice(index).join(" "), lineLength + 1)];
}

function statusAccent(status: ConstructStatus | undefined): string | undefined {
  if (status === "blocking") return BLOCKING;
  if (status === "marginalized") return MARGINALIZED;
  return undefined;
}

function statusLabel(status: ConstructStatus | undefined): string | null {
  if (status === "blocking") return "blocking";
  if (status === "marginalized") return "marginalized";
  return null;
}

function LayerPill({ x, label, color }: { x: number; label: string; color: string }) {
  const width = Math.max(38, label.length * 5.3 + 14);
  return (
    <g transform={`translate(${x - width},8)`}>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

function pathForSeries(
  series: number[],
  x: number,
  y: number,
  width: number,
  height: number,
  minimum: number,
  maximum: number,
): string {
  const range = maximum - minimum || 1;
  const denominator = Math.max(1, series.length - 1);
  return series
    .map((value, index) => {
      const px = x + (index / denominator) * width;
      const py = y + height - ((value - minimum) / range) * height;
      return `${index === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join("");
}

function MiniTrajectory({
  days,
  reference,
  action,
  dayIndex,
}: {
  days: number[];
  reference: number[];
  action: number[];
  dayIndex: number;
}) {
  if (reference.length !== days.length) {
    throw new Error("A simulation reference trajectory is not aligned to its day axis.");
  }
  if (action.length > 0 && action.length !== days.length) {
    throw new Error("A simulation action trajectory is not aligned to its day axis.");
  }
  if (reference.length === 0) return null;

  const values = [...reference, ...action];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const x = 14;
  const y = 77;
  const width = LAYERED_NODE_WIDTH - 28;
  const height = 27;
  const index = Math.max(0, Math.min(reference.length - 1, dayIndex));
  const markerX = x + (index / Math.max(1, reference.length - 1)) * width;

  return (
    <g>
      <line x1={x} x2={x + width} y1={y + height / 2} y2={y + height / 2} stroke="#edf0f3" />
      <path
        d={pathForSeries(reference, x, y, width, height, minimum, maximum)}
        fill="none"
        stroke={DAG_COLORS.slate}
        strokeWidth={1.35}
        strokeOpacity={0.7}
      />
      {action.length > 0 ? (
        <path
          d={pathForSeries(action, x, y, width, height, minimum, maximum)}
          fill="none"
          stroke={DAG_COLORS.intervention}
          strokeWidth={1.8}
        />
      ) : null}
      <line
        x1={markerX}
        x2={markerX}
        y1={y - 2}
        y2={y + height + 2}
        stroke={DAG_COLORS.ink}
        strokeOpacity={0.18}
      />
    </g>
  );
}

function measurementSummary(
  indicators: IndicatorSpec[],
  likelihoodByVariable: ReadonlyMap<string, LikelihoodSpec>,
  warningVariables: ReadonlySet<string>,
): string {
  const rendered = indicators.slice(0, 2).map((indicator) => {
    const likelihood = likelihoodByVariable.get(indicator.id);
    const suffix = likelihood
      ? `:${likelihood.law.distribution}`
      : `:${indicator.measurement_dtype}`;
    return `${warningVariables.has(indicator.id) ? "!" : "•"} ${truncate(humanize(indicator.name), 15)}${suffix}`;
  });
  if (indicators.length > 2) rendered.push(`+${indicators.length - 2}`);
  return rendered.join("  ");
}

function ConstructCard({
  construct,
  isOutcome,
  indicators,
  likelihoodByVariable,
  warningVariables,
  status,
  persistence,
  days,
  reference,
  action,
  dayIndex,
  clampLabel,
  selected,
  dimmed,
  onSelect,
}: {
  construct: ConstructSpec;
  isOutcome: boolean;
  indicators: IndicatorSpec[];
  likelihoodByVariable: ReadonlyMap<string, LikelihoodSpec>;
  warningVariables: ReadonlySet<string>;
  status?: ConstructStatus;
  persistence?: PosteriorEstimate;
  days: number[];
  reference: number[];
  action: number[];
  dayIndex: number;
  clampLabel?: string;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
}) {
  const [descriptionLine1, descriptionLine2] = wrapDescription(construct.description);
  const label = statusLabel(status);
  const accent = clampLabel ? DAG_COLORS.intervention : statusAccent(status);
  const summary = measurementSummary(indicators, likelihoodByVariable, warningVariables);
  const badge = clampLabel ?? label ?? null;
  const badgeColor = clampLabel
    ? DAG_COLORS.intervention
    : status === "blocking"
      ? BLOCKING
      : status === "marginalized"
        ? MARGINALIZED
        : DAG_COLORS.slate;

  return (
    <g
      opacity={dimmed ? 0.18 : status === "marginalized" ? 0.62 : 1}
      role="button"
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_NODE_WIDTH}
        height={LAYERED_NODE_HEIGHT}
        title={`${isOutcome ? "★ " : ""}${truncate(humanize(construct.name), 29)}`}
        subtitle={`${construct.role} · ${construct.temporal_status === "time_varying" ? "varying" : "invariant"}`}
        accent={selected ? "var(--primary)" : accent}
        dashed={status === "marginalized"}
        highlighted={selected}
        outcome={isOutcome}
      >
        {badge ? <LayerPill x={LAYERED_NODE_WIDTH - 8} label={badge} color={badgeColor} /> : null}
        <text x={14} y={61} fontSize={8.2} fill="var(--muted-foreground)">
          {descriptionLine1}
        </text>
        <text x={14} y={72} fontSize={8.2} fill="var(--muted-foreground)">
          {descriptionLine2}
        </text>
        {days.length > 0 && reference.length > 0 ? (
          <MiniTrajectory days={days} reference={reference} action={action} dayIndex={dayIndex} />
        ) : persistence ? (
          <text
            x={14}
            y={94}
            fontSize={8}
            fontFamily="ui-monospace, monospace"
            fill={DAG_COLORS.muted}
          >
            {`decay ${persistence.mean.toFixed(2)} [${persistence.lower.toFixed(2)}, ${persistence.upper.toFixed(2)}] ${formatPosteriorIntervalLabel(persistence)}`}
          </text>
        ) : null}
        {indicators.length > 0 ? (
          <g>
            <line x1={12} x2={LAYERED_NODE_WIDTH - 12} y1={110} y2={110} stroke="var(--border)" />
            <text
              x={14}
              y={125}
              fontSize={7.2}
              fontFamily="ui-monospace, monospace"
              fill={DAG_COLORS.muted}
            >
              {truncate(summary, 61)}
            </text>
          </g>
        ) : null}
      </DagNodeShell>
    </g>
  );
}

function HistoryCard({
  construct,
  status,
  persistence,
  dimmed,
  selected,
  onSelect,
}: {
  construct: ConstructSpec;
  status?: ConstructStatus;
  persistence?: PosteriorEstimate;
  dimmed: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <g
      opacity={dimmed ? 0.13 : 0.45}
      role="button"
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_HISTORY_WIDTH}
        height={LAYERED_HISTORY_HEIGHT}
        title={`${truncate(humanize(construct.name), 20)} · t−1`}
        subtitle={
          persistence ? `fitted decay rate ${persistence.mean.toFixed(2)}` : "previous-time state"
        }
        accent={selected ? "var(--primary)" : statusAccent(status)}
        dashed
        highlighted={selected}
      />
    </g>
  );
}

function EdgeSlot({
  meta,
  disposition,
  posterior,
  color,
  pruned,
  specificationVisible,
  dimmed,
}: {
  meta: LayeredGraphEdgeMeta;
  disposition?: import("@nof1-causal-lab/api-types").StructuralItemDisposition["disposition"];
  posterior?: PosteriorEstimate;
  color: string;
  pruned: boolean;
  specificationVisible: boolean;
  dimmed: boolean;
}) {
  const top = meta.isSelf ? "AR(1)" : meta.lagged ? "lag 1" : "same t";
  const bottom = pruned
    ? "cut by do()"
    : posterior
      ? `${posterior.mean >= 0 ? "+" : ""}${posterior.mean.toFixed(2)}`
      : disposition === "projected_edge"
        ? "projected"
        : specificationVisible
          ? meta.isSelf
            ? "ρ prior"
            : "β prior"
          : disposition === "retained_edge"
            ? "retained"
            : null;

  return (
    <g opacity={dimmed ? 0.12 : 1}>
      <rect
        width={LAYERED_EDGE_SLOT_WIDTH}
        height={LAYERED_EDGE_SLOT_HEIGHT}
        rx={8}
        fill="var(--card)"
        stroke={color}
        strokeOpacity={0.55}
        strokeDasharray={disposition === "projected_edge" ? "4,3" : undefined}
      />
      <text
        x={LAYERED_EDGE_SLOT_WIDTH / 2}
        y={bottom ? 12 : 19}
        textAnchor="middle"
        fontSize={7.2}
        fontWeight={650}
        fill={color}
      >
        {top}
      </text>
      {bottom ? (
        <text
          x={LAYERED_EDGE_SLOT_WIDTH / 2}
          y={24}
          textAnchor="middle"
          fontSize={7.2}
          fontFamily="ui-monospace, monospace"
          fill={color}
        >
          {bottom}
        </text>
      ) : null}
    </g>
  );
}

function LayerControls({
  available,
  visible,
  onToggle,
}: {
  available: CausalGraphLayerId[];
  visible: ReadonlySet<CausalGraphLayerId>;
  onToggle: (layer: CausalGraphLayerId) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label="Causal graph layers">
      {available.map((layer, index) => {
        const active = visible.has(layer);
        return (
          <button
            key={layer}
            type="button"
            disabled={layer === "structure"}
            aria-pressed={active}
            onClick={() => onToggle(layer)}
            className={`rounded-full border px-2.5 py-1 text-[10px] font-medium transition-colors ${
              active
                ? "border-slate-500 bg-slate-800 text-white"
                : "border-slate-200 bg-white text-slate-400"
            } disabled:cursor-default disabled:opacity-100`}
          >
            {`${index + 1} · ${LAYER_LABELS[layer]}`}
          </button>
        );
      })}
    </div>
  );
}

export function LayeredCausalGraph({
  model,
  simulation = null,
  comparison = null,
  selectedNode,
  onSelectNode,
  variant = "workbench",
}: LayeredCausalGraphProps) {
  const {
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
  } = useLayeredGraph({ model, simulation, comparison, selectedNode });

  const canvas = (
    <DagCanvasFrame fill={variant === "asset"}>
      {isLayouting ? (
        <div
          className={`animate-pulse rounded-xl bg-slate-100 ${variant === "asset" ? "h-full" : "h-[560px]"}`}
        />
      ) : (
        <DagSvg
          contentWidth={difference.width + CANVAS_PADDING * 2}
          contentHeight={difference.height + CANVAS_PADDING * 2}
          zoom={zoom}
          style={
            variant === "asset"
              ? {
                  marginLeft: Math.max(
                    0,
                    (paneWidth - 14 - (width + CANVAS_PADDING * 2) * zoom) / 2,
                  ),
                  overflow: "visible",
                }
              : undefined
          }
          role="img"
          aria-label="Layered causal graph"
        >
          <g transform={`translate(${CANVAS_PADDING},${CANVAS_PADDING})`}>
            {graphBands.map((band) => {
              const bounds = boundsForBand(band);
              if (!bounds) return null;
              return (
                <g key={band.key}>
                  <rect
                    {...bounds}
                    rx={14}
                    fill={band.key === "present" ? "#f8fafc" : "#fbfcfd"}
                    stroke="#e7ebef"
                    strokeDasharray={band.key === "history" ? "5,4" : undefined}
                  />
                  <text
                    x={bounds.x + 10}
                    y={bounds.y + 16}
                    fontSize={8}
                    fontWeight={700}
                    letterSpacing={0.7}
                    fill={DAG_COLORS.muted}
                  >
                    {band.label.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {routedSegments.map((segment) => {
              const segmentMeta = topology.segmentMeta.get(segment.id);
              if (!segmentMeta) return null;
              const meta = topology.edgeMeta.get(segmentMeta.edgeId);
              if (!meta) return null;
              const visual = edgeVisual(meta);
              return (
                <g key={segment.id} data-edge-id={meta.id} data-change={visual.change}>
                  <DagEdge
                    points={segment.points}
                    color={visual.color}
                    width={visual.width}
                    dashed={visual.disposition === "projected_edge"}
                    opacity={visual.opacity}
                    markerEnd={segmentMeta.markerEnd && !visual.activeClamp}
                    highlighted={hoveredEdge === meta.id}
                    onHoverChange={(hovered) => setHoveredEdge(hovered ? meta.id : null)}
                  />
                </g>
              );
            })}

            {nodes.map((node) => {
              const meta = topology.nodeMeta.get(node.id);
              if (!meta) return null;
              if (meta.kind === "edge_slot") {
                const edge = topology.edgeMeta.get(meta.edgeId);
                if (!edge) return null;
                const visual = edgeVisual(edge);
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    transform={`translate(${node.x},${node.y})`}
                    onPointerEnter={() => setHoveredEdge(edge.id)}
                    onPointerLeave={() => setHoveredEdge(null)}
                  >
                    <EdgeSlot
                      meta={edge}
                      disposition={visual.disposition}
                      posterior={visual.posterior}
                      color={visual.color}
                      pruned={visual.activeClamp}
                      specificationVisible={specificationVisible}
                      dimmed={visual.dimmed}
                    />
                  </g>
                );
              }

              const construct = meta.construct;
              const dimmed =
                selectedNeighborhood != null && !selectedNeighborhood.has(construct.id);
              const selected = selectedNode === construct.id;
              const select = () =>
                onSelectNode(selectedNode === construct.id ? null : construct.id);
              if (meta.kind === "history") {
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    transform={`translate(${node.x},${node.y})`}
                  >
                    <HistoryCard
                      construct={construct}
                      status={nodeStatuses.get(construct.id) ?? undefined}
                      persistence={persistencePosteriors[construct.id]}
                      dimmed={dimmed}
                      selected={selected}
                      onSelect={select}
                    />
                  </g>
                );
              }

              const nodeIndicators = indicatorsByConstruct.get(construct.id) ?? [];
              const reference = simulationResult
                ? (getNodeReferenceSeries(simulationResult, construct.id) ?? [])
                : [];
              const action = simulationResult
                ? (getNodeActionSeries(simulationResult, construct.id) ?? [])
                : [];
              const clamp =
                currentDay == null
                  ? undefined
                  : simulationResult?.request.clamps.find(
                      (candidate) =>
                        candidate.target === construct.id &&
                        candidate.from_day <= currentDay &&
                        (candidate.to_day == null || currentDay < candidate.to_day),
                    );
              const clampLabel = clamp ? `do(${clamp.mode})` : undefined;
              return (
                <g
                  key={node.id}
                  data-node-id={node.id}
                  data-change={difference.constructChanges.get(construct.id)}
                  transform={`translate(${node.x},${node.y})`}
                >
                  <ConstructCard
                    construct={construct}
                    isOutcome={
                      construct.id ===
                      (simulation?.request.outcome ?? model.model?.value.default_outcome)
                    }
                    indicators={nodeIndicators}
                    likelihoodByVariable={likelihoodByVariable}
                    warningVariables={warningVariables}
                    status={nodeStatuses.get(construct.id) ?? undefined}
                    persistence={persistencePosteriors[construct.id]}
                    days={days}
                    reference={reference}
                    action={action}
                    dayIndex={clampedDayIndex}
                    clampLabel={clampLabel}
                    selected={selected}
                    dimmed={dimmed}
                    onSelect={select}
                  />
                  {difference.constructChanges.has(construct.id) && (
                    <rect
                      x={-3}
                      y={-3}
                      width={LAYERED_NODE_WIDTH + 6}
                      height={LAYERED_NODE_HEIGHT + 6}
                      rx={12}
                      fill="none"
                      stroke={COMPARISON_COLORS[difference.constructChanges.get(construct.id)!]}
                      strokeWidth={2 / zoom}
                    />
                  )}
                </g>
              );
            })}
            {comparison && <LayeredComparisonOverlay overlay={difference} zoom={zoom} />}
          </g>
        </DagSvg>
      )}
    </DagCanvasFrame>
  );

  if (variant === "asset") {
    return (
      <div ref={paneRef} className="relative flex h-full min-h-0 flex-col">
        {canvas}
        <div className="absolute right-3 bottom-2 flex items-center gap-1.5 rounded-lg bg-white/85 px-1.5 py-0.5">
          <DagZoomControls zoom={zoom} onZoomChange={setZoom} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-0 space-y-3 bg-slate-50/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-white px-3 py-2.5">
        <div>
          <div className="text-xs font-semibold text-slate-800">Causal model</div>
          <div className="text-[10px] text-muted-foreground">
            One structural topology; each materialized artifact adds one visual layer.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <LayerControls available={available} visible={visible} onToggle={toggleLayer} />
          <DagZoomControls zoom={zoom} onZoomChange={setZoom} />
        </div>
      </div>

      {canvas}

      {simulationVisible && days.length > 0 ? (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white px-3 py-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 w-7 p-0"
            onClick={() => setPlaying((current) => !current)}
            aria-label={playing ? "Pause simulation timeline" : "Play simulation timeline"}
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </Button>
          <input
            type="range"
            min={0}
            max={days.length - 1}
            value={clampedDayIndex}
            onChange={(event) => setDayIndex(Number(event.target.value))}
            className="min-w-52 flex-1 accent-blue-600"
            aria-label="Simulation day"
          />
          <span className="min-w-16 text-right font-mono text-[10px] text-slate-600">
            day {currentDay}
          </span>
          <span className="text-[10px] text-muted-foreground">
            gray reference · blue intervention
          </span>
        </div>
      ) : null}

      <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-[9px] text-muted-foreground">
        <span>Lagged effects physically originate in t−1.</span>
        <span>Contemporaneous effects remain inside t.</span>
        {designVisible ? <span>Dashed edge slots are projected by design.</span> : null}
        {fitVisible ? <span>Edge color and weight show fitted sign and magnitude.</span> : null}
      </div>
    </div>
  );
}
