import type { DensityCurve, PosteriorMarginal } from "@nof1-causal-lab/api-types";
import { scaleLinear } from "d3-scale";
import { area, curveLinear, curveStep, line } from "d3-shape";
import type { ReactNode } from "react";
import { type LawCurve, lawLabel } from "@/lib/model-asset/laws";
import { humanize } from "@/lib/model-asset/selection";
import { formatPosteriorIntervalLabel, formatSignificant } from "@/lib/utils/format";

type Point = readonly [number, number];

const PRIOR_COLOR = "var(--muted-foreground)";
/** Stable identity color; a nonlinear response's direction cannot be read from a mean. */
export const POSTERIOR_COLOR = "var(--chart-3)";

const densityPoints = (curve: DensityCurve): Point[] =>
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- DensityCurve owns aligned x and density columns at its backend constructor.
  curve.x.map((x, index) => [x, curve.density[index]!]);
const priorPoints = densityPoints;
const posteriorPoints = (marginal: PosteriorMarginal): Point[] =>
  densityPoints(marginal.density_curve);

/** The value range shared by every backend curve drawn for one law. */
export function lawExtent(curve: LawCurve): [number, number] {
  const values = [priorPoints(curve.prior), ...curve.posteriors.map(posteriorPoints)]
    .flat()
    .map(([x]) => x);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = low === high ? 0.5 : 0;
  return [low - pad, high + pad];
}

const densityPeak = (curve: LawCurve) =>
  Math.max(
    0,
    ...curve.prior.density,
    ...curve.posteriors.flatMap((marginal) => marginal.density_curve.density),
  ) || 1;

/**
 * Prior and posterior share both value and probability-density scales. Histograms retain
 * their bin heights; no spline can invent modes or bridge a low-density gap.
 */
export function LawPlot({
  curve,
  x,
  y,
  width,
  height,
  color = POSTERIOR_COLOR,
}: {
  curve: LawCurve;
  x: number;
  y: number;
  width: number;
  height: number;
  color?: string;
}) {
  const extent = lawExtent(curve);
  const sx = scaleLinear()
    .domain(extent)
    .range([x, x + width]);
  const top = densityPeak(curve);
  const shapes = (points: Point[], histogram = false) => {
    const sy = (density: number) => y + height - (density / top) * height;
    return {
      fill:
        area<Point>()
          .x(([value]) => sx(value))
          .y0(y + height)
          .y1(([, density]) => sy(density))
          .curve(histogram ? curveStep : curveLinear)(points) ?? "",
      stroke:
        line<Point>()
          .x(([value]) => sx(value))
          .y(([, density]) => sy(density))
          .curve(histogram ? curveStep : curveLinear)(points) ?? "",
    };
  };
  const prior = curve.prior.x.length > 0 ? shapes(priorPoints(curve.prior)) : null;
  const tone = curve.stale ? PRIOR_COLOR : color;
  const single = curve.posteriors.length === 1;
  const fitted = curve.posteriors.length > 0;
  return (
    <g>
      <line
        x1={x}
        x2={x + width}
        y1={y + height}
        y2={y + height}
        stroke="var(--border)"
        strokeWidth={0.8}
      />
      {extent[0] < 0 && extent[1] > 0 && (
        <line
          x1={sx(0)}
          x2={sx(0)}
          y1={y}
          y2={y + height}
          stroke="var(--muted-foreground)"
          strokeOpacity={0.45}
          strokeWidth={0.6}
          strokeDasharray="2 2"
        />
      )}
      {prior && (
        <>
          <path d={prior.fill} fill={PRIOR_COLOR} fillOpacity={fitted ? 0.07 : 0.16} />
          <path
            d={prior.stroke}
            fill="none"
            stroke={PRIOR_COLOR}
            strokeWidth={0.9}
            strokeDasharray={fitted ? "2.5 1.5" : undefined}
          />
        </>
      )}
      {curve.posteriors.map((marginal) => {
        const posterior = shapes(posteriorPoints(marginal), true);
        return (
          <g key={marginal.subject.element_id}>
            {single && <path d={posterior.fill} fill={tone} fillOpacity={0.24} />}
            <path
              d={posterior.stroke}
              fill="none"
              stroke={tone}
              strokeWidth={single ? 1.3 : 0.8}
              strokeOpacity={single ? 1 : 0.55}
            />
          </g>
        );
      })}
    </g>
  );
}

const CHART_WIDTH = 256;
const CHART_HEIGHT = 92;
const PLOT = { top: 4, bottom: 72 };

function LegendSwatch({ dashed, color }: { dashed?: boolean; color: string }) {
  return (
    <svg width={14} height={6} aria-hidden="true">
      <line
        x1={0}
        x2={14}
        y1={3}
        y2={3}
        stroke={color}
        strokeWidth={dashed ? 1 : 2}
        strokeDasharray={dashed ? "2.5 1.5" : undefined}
      />
    </svg>
  );
}

/** A law's recorded curves with axis, credible interval and scale legend for the inspector. */
export function LawChart({
  curve,
  color = POSTERIOR_COLOR,
  caption,
}: {
  curve: LawCurve;
  color?: string;
  /** Replaces the law's label and parameter name, e.g. with a link to the parameter. */
  caption?: ReactNode;
}) {
  const sx = scaleLinear().domain(lawExtent(curve)).range([0, CHART_WIDTH]);
  const ticks = sx.ticks(4);
  const format = sx.tickFormat(4);
  const posterior = curve.posteriors.length === 1 ? curve.posteriors[0] : null;
  const tone = curve.stale ? PRIOR_COLOR : color;
  return (
    <figure className="m-0 flex w-[256px] flex-none flex-col gap-1">
      <figcaption className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="min-w-0">
          {caption ?? (
            <>
              <span className="font-medium">{lawLabel(curve)}</span>
              <span className="block truncate text-[10px] text-muted-foreground">
                {humanize(curve.parameter.name)}
              </span>
            </>
          )}
        </span>
        {posterior ? (
          <span
            className="flex-none text-right font-mono text-[10px]"
            title={formatPosteriorIntervalLabel(posterior)}
          >
            <span style={{ color: tone }}>{formatSignificant(posterior.mean)}</span>{" "}
            <span className="text-muted-foreground">
              [{formatSignificant(posterior.lower)}, {formatSignificant(posterior.upper)}]
            </span>
          </span>
        ) : curve.posteriors.length > 1 ? (
          <span className="flex-none text-[10px] text-muted-foreground">
            {curve.posteriors.length} elements
          </span>
        ) : null}
      </figcaption>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        className="w-full overflow-visible"
        role="img"
        aria-label={`${lawLabel(curve)}: ${curve.posteriors.length ? "prior and posterior densities" : "prior density"}`}
      >
        <LawPlot
          curve={curve}
          x={0}
          y={PLOT.top}
          width={CHART_WIDTH}
          height={PLOT.bottom - PLOT.top}
          color={color}
        />
        {posterior && (
          <g>
            <line
              x1={sx(posterior.lower)}
              x2={sx(posterior.upper)}
              y1={PLOT.bottom + 4}
              y2={PLOT.bottom + 4}
              stroke={tone}
              strokeWidth={2}
            />
            <circle cx={sx(posterior.mean)} cy={PLOT.bottom + 4} r={2.4} fill={tone} />
          </g>
        )}
        {ticks.map((tick) => (
          <text
            key={tick}
            x={sx(tick)}
            y={CHART_HEIGHT - 2}
            textAnchor="middle"
            fontSize={8.5}
            fill="var(--muted-foreground)"
          >
            {format(tick)}
          </text>
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
        {curve.prior.x.length > 0 && (
          <span className="inline-flex items-center gap-1">
            <LegendSwatch dashed={curve.posteriors.length > 0} color={PRIOR_COLOR} />
            {curve.kind === "fitted" ? "conditioned prior" : "authored prior"}
          </span>
        )}
        {curve.posteriors.length > 0 && (
          <span className="inline-flex items-center gap-1">
            <LegendSwatch color={tone} />
            posterior histogram{curve.stale ? " · earlier panel" : ""}
          </span>
        )}
        {posterior && (
          <span className="inline-flex items-center gap-1">
            <svg width={14} height={6} aria-hidden="true">
              <line x1={1} x2={13} y1={3} y2={3} stroke={tone} strokeWidth={2} />
              <circle cx={7} cy={3} r={2.4} fill={tone} />
            </svg>
            mean · {formatPosteriorIntervalLabel(posterior)}
          </span>
        )}
        <span>density scale · 0–{formatSignificant(densityPeak(curve))}</span>
      </div>
    </figure>
  );
}
