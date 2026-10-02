"use client";

import type {
  ConstructSpec,
  IndicatorEmpiricalProfile,
  IndicatorSpec,
  InterventionSpec,
  PosteriorEstimate,
  ActionId,
  DataDiffReport,
  SimulationPaths,
  StructuralItemDisposition,
} from "@nof1-causal-lab/api-types";
import { scaleLinear } from "d3-scale";
import { Pause, Play } from "lucide-react";
import type { KeyboardEvent, ReactNode } from "react";
import { LawPlot, lawExtent } from "@/components/charts/law-density";
import { Button } from "@/components/ui/button";
import { HistoryPlot } from "@/components/charts/history-plot";
import { dataComparisonHistory, pathLines } from "@/components/model/scopes/recorded-history";
import {
  LAYERED_EDGE_SLOT_HEIGHT,
  LAYERED_EDGE_SLOT_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  type LayeredGraphEdgeMeta,
} from "@/lib/dag/build-layered-causal-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { boundsForBand, type CausalGraphLayerId } from "@/lib/dag/layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, MARGINALIZED } from "@/lib/dag/palette";
import { type LayeredGraphOptions, useLayeredGraph } from "@/lib/dag/use-layered-graph";
import { entityFailures } from "@/lib/model-asset/inspector";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { type LawCurve, lawLabel } from "@/lib/model-asset/laws";
import {
  formatModelDate,
  formatPosteriorIntervalLabel,
  formatSignificant,
} from "@/lib/utils/format";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";
import { DagZoomControls } from "../core/dag-zoom-controls";
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
  simulationPaths?: SimulationPaths | null;
  dataDiff?: DataDiffReport | null;
  onSelect: (selection: EntitySelection | null) => void;
  /** The action whose version is viewed; a data preparation shows each node's prepared data. */
  step?: ActionId | null;
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

function LayerPill({
  x,
  label,
  color,
  description,
}: {
  x: number;
  label: string;
  color: string;
  description?: string;
}) {
  const width = Math.max(38, label.length * 5.3 + 14);
  return (
    <g transform={`translate(${x - width},8)`} role="img" aria-label={description ?? label}>
      <title>{description ?? label}</title>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

function assignmentLabel(event: InterventionSpec, timeOrigin: string | null): string {
  const when = timeOrigin
    ? `${formatModelDate(event.time, timeOrigin)} (day ${event.time})`
    : `Day ${event.time}`;
  return `${when}: set to ${event.value}`;
}

/** The card's chart strip, below its title. */
const STRIP = { x: 14, top: 36, width: LAYERED_NODE_WIDTH - 28, height: 64 } as const;
/** One row per law or indicator: label, value, then its chart. */
const STRIP_ROW = { height: 21, value: 108, plot: 114 } as const;
const STRIP_ROWS = 3;
const rowTop = (index: number) => STRIP.top + index * STRIP_ROW.height;

function lawTitle(curve: LawCurve): string {
  const posterior = curve.posteriors.length === 1 ? curve.posteriors[0] : null;
  const summary = posterior
    ? `posterior ${formatSignificant(posterior.mean)} [${formatSignificant(posterior.lower)}, ${formatSignificant(posterior.upper)}] ${formatPosteriorIntervalLabel(posterior)}`
    : curve.posteriors.length > 1
      ? `${curve.posteriors.length} posterior elements`
      : `authored ${curve.family ? `${curve.family} ` : ""}prior`;
  return [
    lawLabel(curve),
    humanize(curve.parameter.name),
    summary,
    ...(curve.stale ? ["fitted on an earlier panel"] : []),
  ].join(" · ");
}

/** One labelled row of a card's chart strip; its chart sits right of the value. */
function StripRow({
  index,
  title,
  label,
  value,
  valueTone,
  range,
  children,
}: {
  index: number;
  title: string;
  label: string;
  value: string;
  valueTone: string;
  /** The chart's value range, labelled under its ends. */
  range?: readonly [number, number];
  children: ReactNode;
}) {
  const top = rowTop(index);
  const bottom = top + STRIP_ROW.height - 0.5;
  return (
    <g role="img" aria-label={title}>
      <title>{title}</title>
      <rect x={STRIP.x} y={top} width={STRIP.width} height={STRIP_ROW.height} fill="transparent" />
      <text x={STRIP.x} y={top + 11} fontSize={7.5} fontWeight={600} fill={DAG_COLORS.slate}>
        {truncate(label, 17)}
      </text>
      <text
        x={STRIP.x + STRIP_ROW.value}
        y={top + 11}
        textAnchor="end"
        fontSize={7.5}
        fontFamily="ui-monospace, monospace"
        fill={valueTone}
      >
        {value}
      </text>
      {children}
      {range ? (
        <>
          <text x={STRIP.x + STRIP_ROW.plot} y={bottom} fontSize={5.5} fill={DAG_COLORS.muted}>
            {formatSignificant(range[0])}
          </text>
          <text
            x={STRIP.x + STRIP.width}
            y={bottom}
            textAnchor="end"
            fontSize={5.5}
            fill={DAG_COLORS.muted}
          >
            {formatSignificant(range[1])}
          </text>
        </>
      ) : null}
    </g>
  );
}

/** Rows past the strip stay in the inspector. */
function StripOverflow({ hidden }: { hidden: number }) {
  return hidden > 0 ? (
    <text
      x={LAYERED_NODE_WIDTH - 8}
      y={LAYERED_NODE_HEIGHT - 4}
      textAnchor="end"
      fontSize={6.5}
      fill={DAG_COLORS.muted}
    >
      {`+${hidden} more`}
    </text>
  ) : null;
}

/** Each own law's backend curves: the prior outlined, posteriors filled after a fit. */
function LawStrip({ laws }: { laws: LawCurve[] }) {
  const shown = laws.slice(0, STRIP_ROWS);
  return (
    <g>
      {shown.map((curve, index) => {
        const posterior = curve.posteriors.length === 1 ? curve.posteriors[0] : null;
        return (
          <StripRow
            key={curve.parameter.id}
            index={index}
            title={lawTitle(curve)}
            label={lawLabel(curve)}
            value={
              posterior
                ? formatSignificant(posterior.mean)
                : curve.posteriors.length > 1
                  ? `×${curve.posteriors.length}`
                  : truncate(curve.family ?? "prior", 9)
            }
            valueTone={posterior && !curve.stale ? DAG_COLORS.ink : DAG_COLORS.muted}
            range={lawExtent(curve)}
          >
            <LawPlot
              curve={curve}
              x={STRIP.x + STRIP_ROW.plot}
              y={rowTop(index) + 2}
              width={STRIP.width - STRIP_ROW.plot}
              height={STRIP_ROW.height - 8}
            />
          </StripRow>
        );
      })}
      <StripOverflow hidden={laws.length - shown.length} />
    </g>
  );
}

/** A node's prepared indicators: observation counts and recorded range, quartiles and median. */
function DataStrip({
  rows,
}: {
  rows: Array<{ indicator: IndicatorSpec; profile: IndicatorEmpiricalProfile }>;
}) {
  const shown = rows.slice(0, STRIP_ROWS);
  return (
    <g>
      {shown.map(({ indicator, profile }, index) => {
        const { min, q25, q50, q75, max } = profile;
        const quartiles =
          min != null && q25 != null && q50 != null && q75 != null && max != null
            ? { min, q25, q50, q75, max }
            : null;
        const name = humanize(indicator.name);
        const counted = `${name} · ${profile.n_obs.toLocaleString()} observations`;
        const sx = scaleLinear()
          .domain(
            quartiles && quartiles.min !== quartiles.max
              ? [quartiles.min, quartiles.max]
              : [(min ?? 0) - 0.5, (max ?? 0) + 0.5],
          )
          .range([STRIP.x + STRIP_ROW.plot, STRIP.x + STRIP.width]);
        const mid = rowTop(index) + 8.5;
        return (
          <StripRow
            key={indicator.id}
            index={index}
            title={
              quartiles
                ? `${counted} · min ${formatSignificant(quartiles.min)} · quartiles ${formatSignificant(quartiles.q25)}–${formatSignificant(quartiles.q75)} · median ${formatSignificant(quartiles.q50)} · max ${formatSignificant(quartiles.max)}`
                : counted
            }
            label={name}
            value={`n ${formatSignificant(profile.n_obs)}`}
            valueTone={DAG_COLORS.ink}
            {...(quartiles ? { range: [quartiles.min, quartiles.max] as const } : {})}
          >
            {quartiles ? (
              <>
                <line
                  x1={sx(quartiles.min)}
                  x2={sx(quartiles.max)}
                  y1={mid}
                  y2={mid}
                  stroke={DAG_COLORS.muted}
                />
                <rect
                  x={sx(quartiles.q25)}
                  y={mid - 4}
                  width={Math.max(1.2, sx(quartiles.q75) - sx(quartiles.q25))}
                  height={8}
                  rx={1.5}
                  fill={DAG_COLORS.intervention}
                  fillOpacity={0.14}
                  stroke={DAG_COLORS.intervention}
                  strokeWidth={0.8}
                />
                <line
                  x1={sx(quartiles.q50)}
                  x2={sx(quartiles.q50)}
                  y1={mid - 4}
                  y2={mid + 4}
                  stroke={DAG_COLORS.intervention}
                  strokeWidth={1.4}
                />
              </>
            ) : null}
          </StripRow>
        );
      })}
      <StripOverflow hidden={rows.length - shown.length} />
    </g>
  );
}

function ConstructCard({
  construct,
  isOutcome,
  failures,
  status,
  children,
  assignments,
  timeOrigin,
  selected,
  dimmed,
  onSelect,
}: {
  construct: ConstructSpec;
  isOutcome: boolean;
  failures: string[];
  status: ConstructStatus | undefined;
  children: ReactNode;
  assignments: InterventionSpec[];
  timeOrigin: string | null;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
}) {
  const label = statusLabel(status);
  const hasAssignments = assignments.length > 0;
  const accent = hasAssignments ? DAG_COLORS.intervention : statusAccent(status);
  const shellAccent = selected ? "var(--primary)" : accent;
  const badge = hasAssignments
    ? `${assignments.length} assignment${assignments.length === 1 ? "" : "s"}`
    : label;
  const badgeColor = hasAssignments
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
      aria-label={humanize(construct.name)}
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_NODE_WIDTH}
        height={LAYERED_NODE_HEIGHT}
        title={`${isOutcome ? "★ " : ""}${truncate(humanize(construct.name), 29)}`}
        {...(shellAccent === undefined ? {} : { accent: shellAccent })}
        dashed={status === "marginalized"}
        highlighted={selected}
        outcome={isOutcome}
      >
        {badge ? (
          <LayerPill
            x={LAYERED_NODE_WIDTH - 8}
            label={badge}
            color={badgeColor}
            {...(hasAssignments
              ? {
                  description: assignments
                    .map((event) => assignmentLabel(event, timeOrigin))
                    .join("; "),
                }
              : {})}
          />
        ) : null}
        {children}
        {failures.length > 0 ? (
          <g role="img" aria-label={failures.join("; ")}>
            <title>{failures.join("\n")}</title>
            <path
              d="M235 37 L241 26 L247 37 Z"
              fill="none"
              stroke="var(--warning-foreground)"
              strokeWidth={1.2}
            />
            <text x={241} y={35} textAnchor="middle" fontSize={8} fill="var(--warning-foreground)">
              !
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
  dimmed,
  selected,
  onSelect,
}: {
  construct: ConstructSpec;
  status: ConstructStatus | undefined;
  dimmed: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const accent = selected ? "var(--primary)" : statusAccent(status);
  return (
    <g
      opacity={dimmed ? 0.13 : 0.45}
      role="button"
      aria-label={`${humanize(construct.name)} at the previous time`}
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_HISTORY_WIDTH}
        height={LAYERED_HISTORY_HEIGHT}
        title={truncate(humanize(construct.name), 20)}
        {...(accent === undefined ? {} : { accent })}
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
  laws,
  color,
  dimmed,
}: {
  meta: LayeredGraphEdgeMeta;
  disposition: StructuralItemDisposition["disposition"] | undefined;
  posterior: PosteriorEstimate | undefined;
  /** The edge mechanism's own laws; the first is drawn in the slot. */
  laws: LawCurve[];
  color: string;
  dimmed: boolean;
}) {
  const timing = meta.isSelf
    ? "Intrinsic dynamics"
    : meta.crossSlice
      ? "State evolution"
      : "Contemporaneous effect";
  const law = laws.at(0);
  if (law) {
    const effect = law.posteriors.length === 1 ? law.posteriors[0] : null;
    const tone = law.stale ? DAG_COLORS.muted : DAG_COLORS.slate;
    return (
      <g opacity={dimmed ? 0.12 : 1}>
        <title>
          {[
            `${timing}${disposition === "projected_edge" ? " · projected" : ""}`,
            ...laws.map(lawTitle),
          ].join("\n")}
        </title>
        <rect
          width={LAYERED_EDGE_SLOT_WIDTH}
          height={LAYERED_EDGE_SLOT_HEIGHT}
          rx={8}
          fill="var(--card)"
          stroke={color}
          strokeOpacity={0.55}
          strokeDasharray={disposition === "projected_edge" ? "4,3" : undefined}
        />
        <LawPlot
          curve={law}
          x={7}
          y={5}
          width={LAYERED_EDGE_SLOT_WIDTH - 14}
          height={LAYERED_EDGE_SLOT_HEIGHT - 19}
          color={tone}
        />
        <text
          x={LAYERED_EDGE_SLOT_WIDTH / 2}
          y={LAYERED_EDGE_SLOT_HEIGHT - 5}
          textAnchor="middle"
          fontSize={7.2}
          fontWeight={650}
          fill={tone}
        >
          {`${lawLabel(law)} · ${effect ? `μ ${formatSignificant(effect.mean)}` : (law.family ?? "prior")}${laws.length > 1 ? ` +${laws.length - 1}` : ""}`}
        </text>
      </g>
    );
  }
  const bottom = posterior
    ? `${posterior.mean >= 0 ? "+" : ""}${formatSignificant(posterior.mean)}`
    : null;

  if (bottom === null)
    return (
      <g opacity={dimmed ? 0.12 : 1}>
        <title>{`${timing}${disposition === "projected_edge" ? " · projected" : ""}`}</title>
        <line
          x1={0}
          x2={LAYERED_EDGE_SLOT_WIDTH}
          y1={LAYERED_EDGE_SLOT_HEIGHT / 2}
          y2={LAYERED_EDGE_SLOT_HEIGHT / 2}
          stroke={color}
          strokeWidth={1.2}
          strokeDasharray={disposition === "projected_edge" ? "4,3" : undefined}
        />
      </g>
    );

  return (
    <g opacity={dimmed ? 0.12 : 1}>
      <title>{timing}</title>
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
        y={LAYERED_EDGE_SLOT_HEIGHT / 2 + 2.5}
        textAnchor="middle"
        fontSize={7.2}
        fontWeight={650}
        fill={color}
      >
        {bottom}
      </text>
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

/**
 * The state the selected action left, one part at a glance. Each construct and edge shows its
 * current laws or posteriors, its prepared data or its simulated draws, and a single mark when a
 * check on it failed. The details pane holds the depth. A data_diff leaf overlays every saved
 * replicate and the observations on indicator rows, or marks added, removed and revised
 * observations in comparison colours. Latent-only constructs and edges recede.
 */
export function LayeredCausalGraph({
  model,
  entities,
  simulation = null,
  simulationPaths = null,
  dataDiff = null,
  comparison = null,
  selection,
  onSelect,
  step = null,
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
    fitVisible,
    simulationVisible,
    nodeStatuses,
    indicatorsByConstruct,
    constructLaws,
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
  } = useLayeredGraph({ model, entities, simulation, comparison, selection });
  const selectedNode = selection?.kind === "construct" ? selection.id : null;

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
                    opacity={dataDiff ? 0.15 : visual.opacity}
                    markerEnd={segmentMeta.markerEnd}
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
                const edge = meta.edge;
                const visual = edgeVisual(edge);
                const owner = entities.edges.find((item) => item.id === edge.id);
                const target: EntitySelection = edge.isSelf
                  ? { kind: "construct", id: edge.cause.id }
                  : { kind: "edge", id: edge.id };
                const select = () => onSelect(selection?.id === target.id ? null : target);
                const failures = owner && !dataDiff ? entityFailures(model, owner) : [];
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    transform={`translate(${node.x},${node.y})`}
                    role="button"
                    aria-label={
                      edge.isSelf
                        ? `${humanize(edge.cause.name)} intrinsic dynamics`
                        : `${humanize(edge.cause.name)} → ${humanize(edge.effect.name)}`
                    }
                    tabIndex={0}
                    style={{ cursor: "pointer" }}
                    onClick={select}
                    onKeyDown={(event) => activateOnKeyboard(event, select)}
                    onPointerEnter={() => setHoveredEdge(edge.id)}
                    onPointerLeave={() => setHoveredEdge(null)}
                  >
                    <rect
                      width={LAYERED_EDGE_SLOT_WIDTH}
                      height={LAYERED_EDGE_SLOT_HEIGHT}
                      fill="transparent"
                    />
                    <EdgeSlot
                      meta={edge}
                      disposition={visual.disposition}
                      posterior={visual.posterior}
                      laws={visual.laws}
                      color={visual.color}
                      dimmed={dataDiff !== null || visual.dimmed}
                    />
                    {failures.length > 0 && (
                      <g role="img" aria-label={failures.join("; ")}>
                        <title>{failures.join("\n")}</title>
                        <path
                          d="M79 13 L85 2 L91 13 Z"
                          fill="var(--card)"
                          stroke="var(--warning-foreground)"
                        />
                        <text
                          x={85}
                          y={11}
                          textAnchor="middle"
                          fontSize={8}
                          fill="var(--warning-foreground)"
                        >
                          !
                        </text>
                      </g>
                    )}
                  </g>
                );
              }

              const construct = meta.construct;
              const constructChange = difference.constructChanges.get(construct.id);
              const dimmed =
                (selectedNeighborhood != null && !selectedNeighborhood.has(construct.id)) ||
                (dataDiff !== null &&
                  (meta.kind === "history" ||
                    !construct.indicators.some((indicator) =>
                      dataDiff.variables.some((variable) => variable.indicator_id === indicator.id),
                    )));
              const selected = selectedNode === construct.id;
              const select = () =>
                onSelect(
                  selectedNode === construct.id ? null : { kind: "construct", id: construct.id },
                );
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
                      dimmed={dimmed}
                      selected={selected}
                      onSelect={select}
                    />
                  </g>
                );
              }

              const nodeIndicators = indicatorsByConstruct.get(construct.id) ?? [];
              const series = simulationPaths?.states[construct.id];
              const prepared =
                step === "prepare_data"
                  ? nodeIndicators.flatMap((indicator) => {
                      const profile = model.data.profile?.value.indicators[indicator.id]?.profile;
                      return profile ? [{ indicator, profile }] : [];
                    })
                  : [];
              const assignments =
                simulationResult?.design.interventions.filter(
                  (event) => event.target === construct.id,
                ) ?? [];
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
                      (simulation?.causal_result?.outcome ?? model.model?.value.default_outcome)
                    }
                    failures={dataDiff ? [] : entityFailures(model, construct)}
                    status={nodeStatuses.get(construct.id) ?? undefined}
                    assignments={assignments}
                    timeOrigin={simulationResult?.time_origin ?? null}
                    selected={selected}
                    dimmed={dimmed}
                    onSelect={select}
                  >
                    {dataDiff ? (
                      <>
                        {nodeIndicators.slice(0, STRIP_ROWS).map((indicator, index) => {
                          const variable = dataDiff.variables.find(
                            (item) => item.indicator_id === indicator.id,
                          );
                          const failures =
                            variable?.predictive_checks?.per_variable_warnings.filter(
                              (finding) =>
                                finding.kind !== "evaluated" || finding.outcome !== "passed",
                            ) ?? [];
                          const selectIndicator = () =>
                            onSelect({ kind: "indicator", id: indicator.id });
                          return (
                            <g
                              key={indicator.id}
                              role="button"
                              tabIndex={0}
                              aria-label={`${humanize(indicator.name)} comparison`}
                              onClick={(event) => {
                                event.stopPropagation();
                                selectIndicator();
                              }}
                              onKeyDown={(event) => {
                                event.stopPropagation();
                                activateOnKeyboard(event, selectIndicator);
                              }}
                            >
                              <StripRow
                                index={index}
                                title={
                                  failures
                                    .map((finding) =>
                                      finding.kind === "evaluated"
                                        ? finding.evidence.note
                                        : finding.detail,
                                    )
                                    .join("; ") || humanize(indicator.name)
                                }
                                label={humanize(indicator.name)}
                                value={failures.length ? "⚠" : ""}
                                valueTone="var(--warning-foreground)"
                              >
                                {variable && (
                                  <foreignObject
                                    x={STRIP.x + STRIP_ROW.plot}
                                    y={rowTop(index)}
                                    width={STRIP.width - STRIP_ROW.plot}
                                    height={STRIP_ROW.height - 2}
                                    pointerEvents="none"
                                  >
                                    <HistoryPlot {...dataComparisonHistory(variable)} compact />
                                  </foreignObject>
                                )}
                              </StripRow>
                            </g>
                          );
                        })}
                        <StripOverflow hidden={nodeIndicators.length - STRIP_ROWS} />
                      </>
                    ) : series ? (
                      <foreignObject
                        x={STRIP.x}
                        y={STRIP.top}
                        width={STRIP.width}
                        height={STRIP.height}
                        pointerEvents="none"
                      >
                        <HistoryPlot
                          compact
                          times={simulationPaths.times}
                          series={pathLines(series)}
                          label={`${humanize(series.label)}: ${series.action.length} individual simulation draws`}
                          markers={[
                            ...assignments.map((event) => ({
                              time: event.time,
                              label: assignmentLabel(event, simulationResult?.time_origin ?? null),
                            })),
                            ...(variant === "workbench" && currentDay !== undefined
                              ? [{ time: currentDay, label: `Viewed day ${currentDay}` }]
                              : []),
                          ]}
                        />
                      </foreignObject>
                    ) : prepared.length > 0 ? (
                      <DataStrip rows={prepared} />
                    ) : (
                      <LawStrip laws={constructLaws.get(construct.id) ?? []} />
                    )}
                  </ConstructCard>
                  {constructChange !== undefined && (
                    <rect
                      x={-3}
                      y={-3}
                      width={LAYERED_NODE_WIDTH + 6}
                      height={LAYERED_NODE_HEIGHT + 6}
                      rx={12}
                      fill="none"
                      stroke={COMPARISON_COLORS[constructChange]}
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
        <span>Dynamic causes connect successive state slices.</span>
        {simulationVisible ? <span>Dashed blue markers show dated assignments.</span> : null}
        {designVisible ? <span>Dashed edge slots are projected by design.</span> : null}
        {fitVisible ? <span>Edge color and weight show fitted sign and magnitude.</span> : null}
      </div>
    </div>
  );
}
