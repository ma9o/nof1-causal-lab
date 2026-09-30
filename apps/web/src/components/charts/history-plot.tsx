"use client";

import { scaleLinear } from "d3-scale";
import { extent as valueExtent } from "d3-array";
import { curveLinear, curveStepAfter, line } from "d3-shape";
import { useId, useState } from "react";
import { formatModelDate, formatSignificant } from "@/lib/utils/format";
import { PlotNumberInput } from "./plot-number-input";

export interface HistoryLine {
  id: string;
  label: string;
  values: (number | null)[];
  color?: string;
  dashed?: boolean;
  emphasized?: boolean;
}

export const PATH_COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#e11d48"];

/** Display coordinates only: no smoothing, time thinning, imputation or statistical reduction. */
export function HistoryPlot({
  times,
  series,
  label,
  timeOrigin,
  xLabel,
  yLabel,
  pointsOnly = false,
  levels,
  support,
  markers = [],
  step = false,
  description,
}: {
  times: number[];
  series: HistoryLine[];
  label: string;
  timeOrigin?: string | null;
  xLabel?: string;
  yLabel?: string;
  pointsOnly?: boolean;
  levels?: string[] | null;
  support?: { start: (number | null)[]; end: (number | null)[] };
  markers?: number[];
  step?: boolean;
  description?: string;
}) {
  const id = useId();
  const [window, setWindow] = useState<[number, number] | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [highlight, setHighlight] = useState<string | null>(null);
  const bounds = valueExtent(
    [...times, ...(support?.start ?? []), ...(support?.end ?? [])].filter(
      (value): value is number => value != null,
    ),
  );
  const extent: [number, number] = [bounds[0] ?? 0, bounds[1] ?? 1];
  const domain = window ?? extent;
  const values = series.flatMap((row) =>
    row.values.filter(
      (value, index): value is number =>
        value !== null && times[index] >= domain[0] && times[index] <= domain[1],
    ),
  );
  const [low, high] = valueExtent(values);
  const lo = low ?? 0;
  const hi = high ?? 1;
  const width = 760,
    height = 290;
  const plot = { left: 66, right: 744, top: 18, bottom: 246 };
  const sx = scaleLinear()
    .domain(domain[0] === domain[1] ? [domain[0] - 0.5, domain[1] + 0.5] : domain)
    .range([plot.left, plot.right]);
  const sy = scaleLinear()
    .domain(
      levels?.length ? [-0.5, levels.length - 0.5] : lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi],
    )
    .range([plot.bottom, plot.top]);
  if (!levels?.length) sy.nice();
  const yTicks = levels?.length
    ? levels.map((_, index) => index).filter((index) => index % Math.ceil(levels.length / 8) === 0)
    : sy.ticks(5);
  const title = (time: number) =>
    timeOrigin
      ? `${new Date(Date.parse(timeOrigin) + time * 86400000).toISOString()} · day ${time}`
      : String(time);
  const axisLabel = xLabel ?? "Model day";
  const path = line<number | null>()
    .defined((v) => v !== null)
    .x((_, index) => sx(times[index]))
    .y((v) => sy(v!))
    .curve(step ? curveStepAfter : curveLinear);

  const canvas = (suffix: string) => (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={label}>
      <defs>
        <clipPath id={`${id}-${suffix}`}>
          <rect
            x={plot.left - 3}
            y={plot.top - 3}
            width={plot.right - plot.left + 6}
            height={plot.bottom - plot.top + 6}
          />
        </clipPath>
      </defs>
      {yTicks.map((tick) => (
        <g key={tick}>
          <line x1={plot.left} x2={plot.right} y1={sy(tick)} y2={sy(tick)} stroke="var(--border)" />
          <text
            x={plot.left - 7}
            y={sy(tick)}
            dominantBaseline="middle"
            textAnchor="end"
            fontSize={11}
            fill="currentColor"
          >
            {levels?.[tick] ?? formatSignificant(tick)}
          </text>
        </g>
      ))}
      {sx.ticks(5).map((tick) => (
        <text
          key={tick}
          x={sx(tick)}
          y={plot.bottom + 18}
          textAnchor="middle"
          fontSize={11}
          fill="currentColor"
        >
          {formatSignificant(tick)}
        </text>
      ))}
      <text
        x={(plot.left + plot.right) / 2}
        y={height - 7}
        textAnchor="middle"
        fontSize={11}
        fill="currentColor"
      >
        {axisLabel.replaceAll("_", " ")}
      </text>
      {yLabel && (
        <text x={plot.left} y={10} fontSize={10} fill="currentColor">
          {yLabel.replaceAll("_", " ")}
        </text>
      )}
      <g clipPath={`url(#${id}-${suffix})`}>
        {markers.map((time) => (
          <line
            key={time}
            x1={sx(time)}
            x2={sx(time)}
            y1={plot.top}
            y2={plot.bottom}
            stroke="var(--foreground)"
            strokeDasharray="3 3"
          >
            <title>Intervention at {title(time)}</title>
          </line>
        ))}
        {series.map((row, rowIndex) => {
          const color = row.color ?? PATH_COLORS[rowIndex % PATH_COLORS.length];
          const active = highlight === row.id;
          const opacity = highlight
            ? active
              ? 1
              : 0.12
            : row.emphasized
              ? 1
              : series.length > 1
                ? 0.38
                : 0.9;
          return (
            <g
              key={row.id}
              stroke={color}
              fill={color}
              opacity={opacity}
              onMouseEnter={() => setHighlight(row.id)}
              onMouseLeave={() => setHighlight(null)}
            >
              {!pointsOnly && (
                <path
                  d={path(row.values) ?? ""}
                  fill="none"
                  strokeWidth={row.emphasized || active ? 2 : 1}
                  strokeDasharray={row.dashed ? "5 3" : undefined}
                >
                  <title>{row.label}</title>
                </path>
              )}
              {row.values.map((value, index) =>
                value !== null &&
                (pointsOnly || (row.values[index - 1] == null && row.values[index + 1] == null)) ? (
                  <g key={index}>
                    {support?.start[index] != null && support.end[index] != null && (
                      <line
                        x1={sx(support.start[index]!)}
                        x2={sx(support.end[index]!)}
                        y1={sy(value)}
                        y2={sy(value)}
                        strokeWidth={2}
                        strokeOpacity={0.35}
                      />
                    )}
                    <circle
                      cx={sx(times[index])}
                      cy={sy(value)}
                      r={row.emphasized || active ? 2.8 : 1.7}
                      fill={row.dashed ? "var(--card)" : color}
                      strokeWidth={row.dashed ? 1 : 0}
                    >
                      <title>{`${row.label} · ${title(times[index])}: ${levels?.[value] ?? value}${support?.start[index] != null ? ` · support ${title(support.start[index]!)}–${title(support.end[index]!)}` : ""}`}</title>
                    </circle>
                  </g>
                ) : null,
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
  return (
    <figure className="m-0 space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        <label>
          From{" "}
          <PlotNumberInput
            aria-label={`${label}: from`}
            className="w-16 rounded border bg-background px-1"
            value={domain[0]}
            step="any"
            onValue={(v) => {
              if (v < domain[1]) setWindow([v, domain[1]]);
            }}
          />
        </label>
        <label>
          To{" "}
          <PlotNumberInput
            aria-label={`${label}: to`}
            className="w-16 rounded border bg-background px-1"
            value={domain[1]}
            step="any"
            onValue={(v) => {
              if (v > domain[0]) setWindow([domain[0], v]);
            }}
          />
        </label>
        <button type="button" className="underline" onClick={() => setWindow(null)}>
          Reset range
        </button>
        <button type="button" className="ml-auto underline" onClick={() => setExpanded(true)}>
          Expand plot
        </button>
      </div>
      {canvas("inline")}
      {timeOrigin && (
        <figcaption className="text-[10px] text-muted-foreground">
          {formatModelDate(domain[0], timeOrigin)} — {formatModelDate(domain[1], timeOrigin)} (UTC)
        </figcaption>
      )}
      {highlight && <p className="text-[10px]">{series.find((s) => s.id === highlight)?.label}</p>}
      {description && (
        <figcaption className="text-[11px] leading-relaxed text-muted-foreground">
          {description}
        </figcaption>
      )}
      {expanded && (
        <dialog
          ref={(dialog) => {
            if (dialog && !dialog.open) dialog.showModal();
          }}
          aria-label={label}
          className="fixed inset-4 z-50 m-auto w-[min(1100px,94vw)] max-w-none rounded-xl border bg-card p-5 shadow-2xl backdrop:bg-black/40"
          onCancel={() => setExpanded(false)}
        >
          <div className="flex justify-between gap-3 text-sm">
            <span>{label.replaceAll("_", " ")}</span>
            <button type="button" autoFocus onClick={() => setExpanded(false)}>
              Close
            </button>
          </div>
          {canvas("expanded")}
          {timeOrigin && (
            <p className="text-xs text-muted-foreground">
              {formatModelDate(domain[0], timeOrigin)} — {formatModelDate(domain[1], timeOrigin)}{" "}
              (UTC)
            </p>
          )}
          {description && (
            <p className="text-xs leading-relaxed text-muted-foreground">{description}</p>
          )}
        </dialog>
      )}
    </figure>
  );
}
