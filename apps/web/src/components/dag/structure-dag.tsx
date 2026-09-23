"use client";

import type { ConstructSpec, IndicatorSpec } from "@nof1-causal-lab/api-types";
import { useStructureGraph, type StructureGraphOptions } from "@/lib/dag/use-structure-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { baseId, GHOST_OPACITY, isGhost } from "@/lib/dag/unroll";
import {
  SEPARATOR_Y,
  INDICATOR_ROW_H,
  SELF_EDGE_OPACITY,
  edgeStatusColor,
  cleanGlyphPath,
} from "@/lib/dag/structure-model";
import { DagCanvasFrame, DagSvg } from "./core/dag-canvas";
import { DagDirectionToggle } from "./core/dag-direction-toggle";
import { DagEdge } from "./core/dag-edge";
import { DagNodeShell } from "./core/dag-node";
import { DagZoomControls } from "./core/dag-zoom-controls";

const labelize = (text: string): string => text.replace(/_/g, " ");

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, Math.max(1, max - 1))}…` : text;
}

/** Border accent: status wins, then selection/hover, else the default border. */
function nodeAccent(status: ConstructStatus | undefined, lit: boolean): string | undefined {
  if (status === "blocking") return "var(--destructive)";
  if (status === "marginalized") return "var(--warning)";
  if (lit) return "var(--primary)";
  return undefined;
}

interface StructureNodeProps {
  width: number;
  height: number;
  construct: ConstructSpec;
  isOutcome: boolean;
  indicators: IndicatorSpec[];
  status?: ConstructStatus;
  lit: boolean;
  /** Rendered as the faded t−1 ghost (no indicators, no status). */
  isPrev: boolean;
}

/** One construct card: name + role·temporal subtitle + its indicator rows. */
function StructureNode({
  width,
  height,
  construct,
  isOutcome,
  indicators,
  status,
  lit,
  isPrev,
}: StructureNodeProps) {
  const isExo = construct.role === "exogenous";
  const vary = construct.temporal_status === "time_varying" ? "varying" : "invariant";
  const subtitle = `${isExo ? "theory exo" : "theory endo"} · ${vary}`;

  const reserved = (isOutcome ? 44 : 28) + (isPrev ? 36 : 0);
  const title = `${isOutcome ? "★ " : ""}${truncate(labelize(construct.name), Math.floor((width - reserved) / 6.6))}${isPrev ? " · t−1" : ""}`;
  const indNameMax = Math.floor((width - 72) / 5.8);
  const showIndicators = !isPrev && indicators.length > 0;

  return (
    <DagNodeShell
      width={width}
      height={height}
      title={title}
      subtitle={subtitle}
      accent={nodeAccent(isPrev ? undefined : status, lit)}
      highlighted={lit}
      outcome={isOutcome}
    >
      <title>{construct.name}</title>
      {showIndicators ? (
        <>
          <line
            x1={12}
            y1={SEPARATOR_Y}
            x2={width - 12}
            y2={SEPARATOR_Y}
            stroke="var(--border)"
            strokeDasharray="3,3"
          />
          {indicators.map((ind, i) => {
            const y = SEPARATOR_Y + 14 + i * INDICATOR_ROW_H;
            return (
              <g key={ind.name}>
                <text x={14} y={y} fontSize={10} fill="var(--muted-foreground)">
                  {truncate(ind.name, indNameMax)}
                </text>
                <text
                  x={width - 12}
                  y={y}
                  fontSize={8.5}
                  textAnchor="end"
                  fill="var(--muted-foreground)"
                >
                  {ind.measurement_dtype}
                </text>
              </g>
            );
          })}
        </>
      ) : null}
    </DagNodeShell>
  );
}

function LegendSwatch({ border, faded }: { border?: string; faded?: boolean }) {
  return (
    <span
      className="inline-block h-3 w-3 rounded-sm border-2 bg-card align-[-2px]"
      style={{ borderColor: border ?? "var(--foreground)", opacity: faded ? GHOST_OPACITY : 1 }}
    />
  );
}

/**
 * Static causal-structure DAG for Stages 1a/1b. Uses the bespoke core's node /
 * edge primitives (`DagNodeShell`, `DagEdge`, ELK routing) on the same scrollable,
 * content-sized canvas as the analysis interactive DAG — full-size cards you scroll
 * and zoom, not a fit-to-box viewport — so the structural and intervention stages
 * read identically.
 *
 * It shows every signal the old React Flow `CausalDag` did: construct cards
 * (name, role·temporal, outcome ★, indicator rows, status-colored borders),
 * status-colored edges (red touching a blocking node, amber touching a marginalized
 * one), click-to-focus (highlight a node's neighborhood, dim the rest) with
 * `onNodeClick`, and edge hover.
 *
 * Topology AND layout mirror the analysis interactive DAG (the reference "good
 * state"): the ELK graph is built the same way — every causal edge is split
 * a → [glyph] → b (the glyph node per edge gives the layered layout its column
 * rhythm; see `splitEdgesWithGlyphs`), lagged causal arrows originate at faded
 * t−1 copies, contemporaneous arrows remain within t, and self-dynamics use
 * separate t−1 → matching-t edges. The same ELK spacing
 * (`DAG_LAYOUT_OPTIONS`) is used — so nodes land in the same columns. The
 * structural DAG leaves each glyph slot empty (drawing the edge straight through)
 * and uses compact cards instead of analysis's trajectory cards. It does not
 * cone-restrict: the structural stages show the full proposed model.
 */
export function StructureDag({
  constructs,
  outcomeId,
  edges,
  indicators,
  nodeStatuses,
  onNodeClick,
  direction = "RIGHT",
}: StructureGraphOptions) {
  const {
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
  } = useStructureGraph({ constructs, edges, indicators, nodeStatuses, onNodeClick, direction });

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      {/* toolbar — mirrors the analysis interactive DAG */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <DagDirectionToggle
          direction={dir === "RIGHT" ? "horizontal" : "vertical"}
          onDirectionChange={(direction) => setDir(direction === "horizontal" ? "RIGHT" : "DOWN")}
        />
        <DagZoomControls zoom={zoom} onZoomChange={setZoomClamped} />
      </div>

      {/* scrollable, content-sized canvas */}
      <DagCanvasFrame>
        {isLayouting ? null : (
          <DagSvg
            contentWidth={W}
            contentHeight={H}
            zoom={zoom}
            role="img"
            aria-label="Causal structure graph"
            onClick={(e) => {
              if (e.target === e.currentTarget) setSelected(null);
            }}
          >
            {[...glyphs.entries()].map(([gid, meta]) => {
              const i = gid.slice(3); // "G__<i>"
              const head = routedById.get(`e${i}s`);
              const tail = routedById.get(`e${i}t`);
              if (!head || !tail) return null;
              const points = cleanGlyphPath([...head.points, ...tail.points]);
              const aBase = baseId(meta.a);
              const bBase = baseId(meta.b);
              const { color, flagged } = edgeStatusColor(
                nodeStatuses?.[aBase],
                nodeStatuses?.[bBase],
              );
              const incident = selected == null || aBase === selected || bBase === selected;
              const isHover = hoverEdgeId === gid;
              const dimmed = selected != null && !incident && !isHover;
              return (
                <DagEdge
                  key={gid}
                  points={points}
                  color={color}
                  width={flagged ? 2.6 : 2}
                  opacity={dimmed ? 0.12 : meta.isSelf ? SELF_EDGE_OPACITY : 0.95}
                  highlighted={isHover}
                  onHoverChange={(h) => setHoverEdgeId(h ? gid : null)}
                />
              );
            })}
            {nodes.map((nd) => {
              if (nd.id.startsWith("G__")) return null; // empty glyph slot
              const base = baseId(nd.id);
              const construct = byName.get(base);
              if (!construct) return null;
              const prev = isGhost(nd.id);
              const lit = selected === base || hoverEndpoints.has(base);
              const dimmed = selected != null && !(connected?.has(base) ?? false);
              const baseOpacity = prev ? GHOST_OPACITY : 1;
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x} ${nd.y})`}
                  opacity={dimmed && !lit ? baseOpacity * 0.35 : baseOpacity}
                  style={{ cursor: "pointer", transition: "opacity 150ms" }}
                  onClick={() => handleNodeClick(nd.id)}
                >
                  <StructureNode
                    width={nd.width}
                    height={nd.height}
                    construct={construct}
                    isOutcome={construct.id === outcomeId}
                    indicators={indicatorsByConstruct.get(base) ?? []}
                    status={nodeStatuses?.[base]}
                    lit={lit}
                    isPrev={prev}
                  />
                </g>
              );
            })}
          </DagSvg>
        )}
      </DagCanvasFrame>

      {/* legend */}
      {hasGhosts || hasMarginalized || hasBlocking ? (
        <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1 text-xs text-muted-foreground">
          {hasGhosts ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--foreground)" faded />
              t−1 construct
            </span>
          ) : null}
          {hasCrossLagged ? <span>Lagged: cause(t−1) → effect(t)</span> : null}
          {hasContemporaneous ? <span>Contemporaneous: cause(t) → effect(t)</span> : null}
          {hasGhosts ? <span>Self-dynamics: construct(t−1) → construct(t)</span> : null}
          {hasMarginalized ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--warning)" />
              marginalized
            </span>
          ) : null}
          {hasBlocking ? (
            <span className="flex items-center gap-1.5">
              <LegendSwatch border="var(--destructive)" />
              blocking
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
