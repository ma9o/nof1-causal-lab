"use client";

import { useInteractiveGraph, type InteractiveGraphOptions } from "@/lib/dag/use-interactive-graph";
import { baseId } from "@/lib/dag/unroll";
import { CARD_H, CARD_W } from "@/lib/dag/build-simulation-graph";
import { getNodeActionSeries, getNodeReferenceSeries } from "@/lib/dag/simulation";
import { DAG_COLORS, signColor } from "@/lib/dag/palette";
import { orthoPath } from "@/lib/dag/ortho-path";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagDirectionToggle } from "../core/dag-direction-toggle";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { IndicatorStack } from "./indicator-stack";
import { TrajectoryCard } from "./trajectory-card";

const {
  positive: TEAL,
  negative: RED,
  neutral: NEUTRAL,
  muted: MUTED,
  intervention: BLUE,
  ink: INK,
} = DAG_COLORS;
const markerFor = (col: string) => (col === TEAL ? "arrPos" : col === RED ? "arrNeg" : "arrZero");

/**
 * Layer the backend's fitted and simulated artifacts over the full scientific
 * DAG. Theory-only structure remains visible; posterior summaries style fitted
 * edges; simulation results materialize reference and action node trajectories.
 */
export function InteractiveDag({
  constructs,
  edges,
  indicators = [],
  edgePosteriors = {},
  persistencePosteriors = {},
  identifiableTreatments = [],
  result,
  onSimulate,
  indicatorsVisible,
  nodeStatuses,
  onNodeClick,
}: InteractiveGraphOptions) {
  const {
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
  } = useInteractiveGraph({
    constructs,
    edges,
    indicators,
    edgePosteriors,
    persistencePosteriors,
    identifiableTreatments,
    result,
    onSimulate,
    indicatorsVisible,
    nodeStatuses,
    onNodeClick,
  });

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif", color: INK, fontSize: 14 }}>
      {/* toolbar */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <DagDirectionToggle
          direction={dir === "LR" ? "horizontal" : "vertical"}
          onDirectionChange={(direction) => setDir(direction === "horizontal" ? "LR" : "TB")}
        />
        {indicatorsVisible === undefined ? (
          <label
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              fontSize: 13,
              color: "#4a4f57",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={showIndicators}
              onChange={(e) => setLocalShowIndicators(e.target.checked)}
            />{" "}
            Indicators
          </label>
        ) : null}
        <DagZoomControls zoom={zoom} onZoomChange={setZoomClamped} />
      </div>

      {/* output */}
      <DagCanvasFrame>
        {isLayouting ? null : (
          <DagSvg contentWidth={W} contentHeight={H} zoom={zoom}>
            <defs>
              {(
                [
                  ["arrPos", TEAL],
                  ["arrNeg", RED],
                  ["arrZero", NEUTRAL],
                ] as const
              ).map(([id, col]) => (
                <marker
                  key={id}
                  id={id}
                  viewBox="0 0 10 10"
                  refX={9}
                  refY={5}
                  markerUnits="userSpaceOnUse"
                  markerWidth={11}
                  markerHeight={11}
                  orient="auto-start-reverse"
                >
                  <path d="M0,1 L9,5 L0,9" fill="none" stroke={col} strokeWidth={1.4} />
                </marker>
              ))}
            </defs>

            {columnBands.map((b, i) =>
              dir === "LR" ? (
                <rect
                  key={i}
                  x={b.min - 10}
                  y={0}
                  width={b.max - b.min + 20}
                  height={Math.ceil(H)}
                  fill={DAG_COLORS.col}
                />
              ) : (
                <rect
                  key={i}
                  x={0}
                  y={b.min - 10}
                  width={Math.ceil(W)}
                  height={b.max - b.min + 20}
                  fill={DAG_COLORS.col}
                />
              ),
            )}

            {/* edges */}
            {routed.map((e) => {
              const meta = edgeMeta.get(e.id);
              if (!meta) return null;
              const { a, b, isSelf, lagged } = meta;
              const posterior = isSelf
                ? persistencePosteriors[baseId(b)]
                : edgePosteriors[`${baseId(a)}→${baseId(b)}`];
              const theoryOnly = !isSelf && posterior == null;
              const col = posterior ? signColor(posterior.mean) : NEUTRAL;
              const sourceStatus = nodeStatuses?.[baseId(a)];
              const targetStatus = nodeStatuses?.[baseId(b)];
              const contextOnly =
                sourceStatus === "marginalized" || targetStatus === "marginalized";
              const currentDay = days[clampedDay];
              const pruned =
                currentDay != null &&
                interventions.some(
                  (clamp) =>
                    currentResult.labels[clamp.target] === baseId(b) &&
                    clamp.from_day <= currentDay &&
                    (clamp.to_day == null || currentDay < clamp.to_day),
                );
              const key = `${a}>${b}`;
              const hl = hoverEdge === key;
              const d = orthoPath(e.points);
              const pts = e.points;
              const ep = pts[pts.length - 1];
              const fittedWidth = posterior
                ? 1.4 +
                  (maximumPosteriorMean > 0
                    ? (Math.abs(posterior.mean) / maximumPosteriorMean) * 3.4
                    : 0)
                : 1.2;
              return (
                <g key={e.id}>
                  <path
                    d={d}
                    fill="none"
                    stroke={contextOnly ? MUTED : pruned ? DAG_COLORS.pruned : col}
                    strokeWidth={pruned ? 1.4 : hl ? fittedWidth + 1.4 : fittedWidth}
                    strokeOpacity={contextOnly ? 0.3 : pruned ? 0.5 : theoryOnly ? 0.35 : 0.9}
                    strokeDasharray={
                      contextOnly || pruned || theoryOnly || lagged ? "5,4" : undefined
                    }
                    markerEnd={
                      !pruned ? `url(#${markerFor(contextOnly ? NEUTRAL : col)})` : undefined
                    }
                    style={hl ? { filter: "drop-shadow(0 0 2px rgba(20,25,30,.28))" } : undefined}
                  />
                  {pruned && ep ? (
                    <text
                      x={ep.x - 9}
                      y={ep.y - 4}
                      textAnchor="middle"
                      fontSize={10}
                      fill={DAG_COLORS.scissors}
                    >
                      ✂
                    </text>
                  ) : null}
                  <path
                    d={d}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={16}
                    pointerEvents="stroke"
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoverEdge(key)}
                    onMouseLeave={() => setHoverEdge(null)}
                  />
                </g>
              );
            })}

            {/* node cards */}
            {nodes.map((nd) => {
              const base = baseId(nd.id);
              const isPrev = nd.id !== base;
              const construct = byName.get(base);
              if (!construct) return null;
              const referenceSeries = getNodeReferenceSeries(currentResult, construct.id) ?? [];
              const actionSeries = getNodeActionSeries(currentResult, construct.id) ?? [];
              const nodeInterventions = isPrev
                ? []
                : interventions.filter((clamp) => currentResult.labels[clamp.target] === base);
              const cardHl = hoverEndpoints.includes(base);
              const status = nodeStatuses?.[base];
              const contextOnly = status === "marginalized";
              return (
                <g
                  key={nd.id}
                  transform={`translate(${nd.x},${nd.y})`}
                  role={!isPrev && onNodeClick ? "button" : undefined}
                  tabIndex={!isPrev && onNodeClick ? 0 : undefined}
                  style={{
                    cursor: !isPrev && onNodeClick ? "pointer" : undefined,
                    ...(cardHl ? { filter: "drop-shadow(0 0 5px rgba(20,25,30,.22))" } : {}),
                  }}
                  onClick={() => {
                    if (!isPrev) onNodeClick?.(base);
                  }}
                  onKeyDown={(event) => {
                    if (!isPrev && (event.key === "Enter" || event.key === " ")) {
                      event.preventDefault();
                      onNodeClick?.(base);
                    }
                  }}
                >
                  <TrajectoryCard
                    width={CARD_W}
                    height={CARD_H}
                    name={base}
                    kind={construct.role === "endogenous" ? "endo" : "exo"}
                    vary={construct.temporal_status === "time_varying" ? "varying" : "invariant"}
                    isTarget={construct.id === result.request.outcome}
                    isPrev={isPrev}
                    days={days}
                    reference={referenceSeries}
                    action={actionSeries}
                    timeIndex={clampedDay}
                    interventions={nodeInterventions}
                    status={status}
                    persistence={persistencePosteriors[base]}
                    interactive={
                      !!onSimulate &&
                      n > 0 &&
                      referenceSeries.length === n &&
                      !isPrev &&
                      !contextOnly &&
                      identifiableTreatmentSet.has(base)
                    }
                    onSetDo={(v) => void setDo(base, v)}
                    onRemoveDo={currentResult !== result ? resetScenario : undefined}
                  />
                  {showIndicators && !isPrev ? (
                    <IndicatorStack
                      indicators={indicators.filter((indicator) =>
                        construct.indicators.some((owned) => owned.id === indicator.id),
                      )}
                    />
                  ) : null}
                </g>
              );
            })}
          </DagSvg>
        )}
      </DagCanvasFrame>

      {/* graph note */}
      <div style={{ fontSize: 11.5, color: MUTED, margin: "8px 4px 0" }}>
        {`Showing the full scientific DAG for ${outcome} (${constructs.length} nodes); ★ marks the outcome.`}
        {Object.values(nodeStatuses ?? {}).includes("marginalized")
          ? "  ·  Marginalized confounders remain visible as subdued causal context."
          : null}
        {
          "  ·  Solid colored edges have fitted posterior coefficients; subdued dashed edges are theory-only context."
        }
      </div>

      {/* scrubber */}
      {n > 0 ? (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 14,
              marginTop: 12,
              background: "#fff",
              border: `1px solid ${DAG_COLORS.line}`,
              borderRadius: 12,
              padding: "12px 16px",
            }}
          >
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              style={iconBtn}
              title="play / pause"
            >
              {playing ? "⏸" : "▶"}
            </button>
            <button
              type="button"
              onClick={() => {
                setPlaying(false);
                setDay(0);
              }}
              style={iconBtn}
              title="reset"
            >
              ↺
            </button>
            <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
              <input
                type="range"
                min={0}
                max={Math.max(0, n - 1)}
                step={1}
                value={clampedDay}
                onChange={(e) => setDay(Number(e.target.value))}
                style={{ width: "100%", accentColor: INK }}
              />
              <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
                {interventions.map((iv, index) => (
                  <span
                    key={`${iv.target}-${iv.from_day}-${index}`}
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      left: `${(iv.from_day / Math.max(days[n - 1], 1)) * 100}%`,
                    }}
                  >
                    <i
                      style={{
                        position: "absolute",
                        top: 2,
                        bottom: 2,
                        left: -1.25,
                        width: 2.5,
                        background: BLUE,
                        borderRadius: 2,
                      }}
                    />
                    <span
                      style={{
                        position: "absolute",
                        top: -16,
                        left: 0,
                        transform: "translateX(-50%)",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 3,
                      }}
                    >
                      <b
                        style={{
                          fontSize: 9,
                          fontWeight: 600,
                          color: "#fff",
                          background: BLUE,
                          borderRadius: 4,
                          padding: "1px 5px",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {`do · ${currentResult.labels[iv.target].replace(/_/g, " ")} @d${iv.from_day}`}
                      </b>
                      {currentResult !== result ? (
                        <span
                          title="Reset to the selected scenario"
                          onClick={resetScenario}
                          style={{
                            cursor: "pointer",
                            pointerEvents: "auto",
                            fontSize: 9,
                            fontWeight: 700,
                            color: "#fff",
                            background: RED,
                            borderRadius: 4,
                            padding: "1px 4px",
                          }}
                        >
                          ↺
                        </span>
                      ) : null}
                    </span>
                  </span>
                ))}
              </div>
            </div>
            <span
              style={{
                fontVariantNumeric: "tabular-nums",
                minWidth: 96,
                textAlign: "right",
                color: "#3a3f47",
              }}
            >
              {`day ${days[clampedDay]}`}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              color: MUTED,
              margin: "6px 8px 0",
            }}
          >
            <span>{`${days[0]}d`}</span>
            <span>{`${days[Math.floor((n - 1) / 2)]}d`}</span>
            <span>{`${days[n - 1]}d`}</span>
          </div>
        </>
      ) : (
        <div
          style={{
            marginTop: 12,
            background: "#fff",
            border: `1px solid ${DAG_COLORS.line}`,
            borderRadius: 12,
            padding: "12px 16px",
            color: MUTED,
            fontSize: 11.5,
          }}
        >
          {`End-state result · effect ${currentResult.summary.mean.toFixed(3)} [${currentResult.summary.lower_95.toFixed(3)}, ${currentResult.summary.upper_95.toFixed(3)}] · no trajectory projection requested.`}
        </div>
      )}
    </div>
  );
}

const iconBtn: React.CSSProperties = {
  border: `1px solid ${DAG_COLORS.line2}`,
  background: "#fff",
  width: 34,
  height: 34,
  borderRadius: 9,
  cursor: "pointer",
  fontSize: 15,
  display: "grid",
  placeItems: "center",
};
