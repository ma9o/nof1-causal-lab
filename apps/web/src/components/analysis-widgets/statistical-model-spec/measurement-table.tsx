"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { LikelihoodSpec, LikelihoodDiagnostics } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { scaleLinear } from "d3-scale";
import { curveMonotoneX, line } from "d3-shape";
import katex from "katex";
import { type MouseEvent, memo, useMemo, useState } from "react";
import { SourceBadges } from "../source-badges";
import { SparklineTooltip } from "./sparkline-tooltip";

// ── Row type ──────────────────────────────────────────────

interface MeasurementRow {
  likelihood: LikelihoodSpec;
  label: string;
  diagnostics?: LikelihoodDiagnostics;
}

interface DisplayBin {
  binCenter: number;
  count: number;
  binStart: number;
  binEnd: number;
}

const MEASUREMENT_CHART_WIDTH = 192;
const MEASUREMENT_CHART_HEIGHT = 80;
const MEASUREMENT_CHART_MARGIN = { top: 4, right: 6, bottom: 15, left: 4 };
interface MeasurementChartPoint extends DisplayBin {
  prior?: number;
}

function measurementXDomain(data: MeasurementChartPoint[]): [number, number] {
  const min = Math.min(...data.map((bin) => Math.min(bin.binStart, bin.binCenter)));
  const max = Math.max(...data.map((bin) => Math.max(bin.binEnd, bin.binCenter)));

  if (min === max) {
    return [min - 0.5, max + 0.5];
  }

  return [min, max];
}

// ── Inline chart ──────────────────────────────────────────

const MeasurementSparkline = memo(
  function MeasurementSparkline({ row }: { row: MeasurementRow }) {
    const nObs = row.diagnostics?.profile?.n_obs ?? 0;
    const bins = (row.diagnostics?.histogram ?? []).map((bin) => ({
      binCenter: bin.bin_center,
      binStart: bin.bin_start,
      binEnd: bin.bin_end,
      count: bin.count,
    }));
    const hasHistogram = bins.length > 0 && nObs > 0;
    const hasPrior = row.diagnostics?.prior_counts != null;
    const priorOutsideFraction = row.diagnostics?.prior_outside_fraction ?? 0;
    const chartData: MeasurementChartPoint[] = bins.map((bin, index) => ({
      ...bin,
      ...(hasPrior ? { prior: row.diagnostics!.prior_counts![index] } : {}),
    }));
    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    if (!hasHistogram) {
      return <span className="text-xs text-muted-foreground">--</span>;
    }

    const plotLeft = MEASUREMENT_CHART_MARGIN.left;
    const plotRight = MEASUREMENT_CHART_WIDTH - MEASUREMENT_CHART_MARGIN.right;
    const plotTop = MEASUREMENT_CHART_MARGIN.top;
    const plotBottom = MEASUREMENT_CHART_HEIGHT - MEASUREMENT_CHART_MARGIN.bottom;
    const [xMin, xMax] = measurementXDomain(chartData);
    const xScale = scaleLinear().domain([xMin, xMax]).range([plotLeft, plotRight]);
    const maxY = Math.max(
      1,
      ...chartData.map((bin) => bin.count),
      ...chartData.map((bin) => bin.prior ?? 0),
    );
    const yScale = scaleLinear().domain([0, maxY]).nice().range([plotBottom, plotTop]);
    const gridTicks = yScale.ticks(3).filter((tick) => tick > 0);
    const defaultBarWidth = Math.max(2, (plotRight - plotLeft) / chartData.length - 1);
    const priorPath = hasPrior
      ? line<MeasurementChartPoint>()
          .x((point) => xScale(point.binCenter))
          .y((point) => yScale(point.prior ?? 0))
          .curve(curveMonotoneX)(chartData)
      : null;
    const xLabels = xMin === xMax ? [xMin] : [xMin, xMax];

    const hovered =
      hoverIndex != null && hoverIndex < chartData.length ? chartData[hoverIndex] : null;

    const handleMove = (event: MouseEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      if (rect.width === 0) return;
      const pointerX = ((event.clientX - rect.left) / rect.width) * MEASUREMENT_CHART_WIDTH;
      let nearest = 0;
      let nearestDist = Number.POSITIVE_INFINITY;
      for (let index = 0; index < chartData.length; index++) {
        const dist = Math.abs(xScale(chartData[index].binCenter) - pointerX);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = index;
        }
      }
      setHoverIndex(nearest);
    };

    return (
      <div
        className="h-20 w-48 cursor-crosshair"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <svg
          className="h-full w-full"
          viewBox={`0 0 ${MEASUREMENT_CHART_WIDTH} ${MEASUREMENT_CHART_HEIGHT}`}
          role="img"
          aria-label="Empirical data histogram overlaid with prior predictive line"
        >
          {gridTicks.map((tick) => (
            <line
              key={tick}
              x1={plotLeft}
              x2={plotRight}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="var(--muted)"
              strokeDasharray="3 3"
            />
          ))}
          {priorOutsideFraction > 0 && (
            <text
              x={plotRight}
              y={plotTop + 7}
              textAnchor="end"
              fill="var(--destructive)"
              fontSize={8}
            >
              PP outside {formatNumber(priorOutsideFraction * 100, 0)}%
            </text>
          )}
          {chartData.map((bin, index) => {
            const hasRange = bin.binStart !== bin.binEnd;
            const rangeWidth = hasRange
              ? xScale(bin.binEnd) - xScale(bin.binStart)
              : defaultBarWidth;
            const barWidth = Math.max(2, rangeWidth - 1);
            const x = hasRange ? xScale(bin.binStart) : xScale(bin.binCenter) - barWidth / 2;
            const y = yScale(bin.count);
            return (
              <rect
                key={`${bin.binStart}:${bin.binEnd}:${bin.binCenter}`}
                x={x}
                y={y}
                width={barWidth}
                height={plotBottom - y}
                fill="var(--muted-foreground)"
                opacity={hoverIndex === index ? 0.55 : 0.3}
              />
            );
          })}
          {priorPath && (
            <path d={priorPath} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
          )}
          <line
            x1={plotLeft}
            x2={plotRight}
            y1={plotBottom}
            y2={plotBottom}
            stroke="var(--border)"
          />
          {xLabels.map((value, index) => (
            <text
              key={value}
              x={index === 0 ? plotLeft : plotRight}
              y={MEASUREMENT_CHART_HEIGHT - 2}
              textAnchor={index === 0 ? "start" : "end"}
              fill="var(--muted-foreground)"
              fontSize={9}
            >
              {formatNumber(value, 1)}
            </text>
          ))}
          {hovered && (
            <g pointerEvents="none">
              <line
                x1={xScale(hovered.binCenter)}
                x2={xScale(hovered.binCenter)}
                y1={plotTop}
                y2={plotBottom}
                stroke="var(--muted-foreground)"
                strokeWidth={1}
                opacity={0.5}
              />
              <circle
                cx={xScale(hovered.binCenter)}
                cy={yScale(hovered.count)}
                r={2.5}
                fill="var(--muted-foreground)"
              />
              {hasPrior && (
                <circle
                  cx={xScale(hovered.binCenter)}
                  cy={yScale(hovered.prior ?? 0)}
                  r={2.5}
                  fill="var(--primary)"
                />
              )}
              <SparklineTooltip
                anchorX={xScale(hovered.binCenter)}
                anchorY={yScale(hovered.count)}
                width={MEASUREMENT_CHART_WIDTH}
                height={MEASUREMENT_CHART_HEIGHT}
                lines={[
                  `x = ${formatNumber(hovered.binCenter, 2)}`,
                  `count = ${formatNumber(hovered.count, 1)}`,
                  ...(hasPrior ? [`prior ≈ ${formatNumber(hovered.prior ?? 0, 1)}`] : []),
                ]}
              />
            </g>
          )}
        </svg>
      </div>
    );
  },
  (previous, next) =>
    previous.row.diagnostics === next.row.diagnostics &&
    previous.row.likelihood.law.distribution === next.row.likelihood.law.distribution,
);

// ── Table columns ─────────────────────────────────────────

const col = createColumnHelper<MeasurementRow>();

const baseColumns: ColumnDef<MeasurementRow, unknown>[] = [
  col.display({
    id: "variable",
    header: "Variable",
    cell: ({ row }) => <span className="font-medium font-mono text-xs">{row.original.label}</span>,
  }),
  col.display({
    id: "distribution",
    header: "Distribution",
    cell: ({ row }) => <Badge variant="outline">{row.original.likelihood.law.distribution}</Badge>,
  }),
  col.display({
    id: "chart",
    header: () => (
      <span className="inline-flex items-center gap-1">
        Data vs Prior
        <StatTooltip explanation="Empirical data histogram (grey bars) overlaid with marginal prior predictive samples (line). Compare to check whether priors imply a plausible data scale." />
      </span>
    ),
    cell: ({ row }) => <MeasurementSparkline row={row.original} />,
  }),
  col.display({
    id: "stats",
    header: "Stats",
    cell: ({ row }) => {
      const profile = row.original.diagnostics?.profile;
      if (!profile || profile.n_obs === 0 || profile.mean == null) {
        return <span className="text-xs text-muted-foreground">--</span>;
      }
      const latex = `n=${profile.n_obs} \\\\[2pt] \\hat{\\mu}=${formatNumber(profile.mean, 2)}`;
      return (
        <span
          className="text-xs text-muted-foreground"
          // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
          dangerouslySetInnerHTML={{
            __html: katex.renderToString(latex, {
              displayMode: false,
              throwOnError: false,
              strict: false,
            }),
          }}
        />
      );
    },
  }),
  col.display({
    id: "reasoning",
    header: "Reasoning",
    cell: ({ row }) => (
      <span className="max-w-xs whitespace-normal text-xs text-muted-foreground">
        {row.original.likelihood.reasoning}
      </span>
    ),
  }),
  col.display({
    id: "sources",
    header: () => (
      <HeaderWithTooltip
        label="Sources"
        tooltip="Literature sources supporting this likelihood distribution choice. Click to open."
      />
    ),
    cell: ({ row }) => <SourceBadges sources={row.original.likelihood.sources} />,
    meta: { align: "center" },
  }),
];

// ── Exported component ────────────────────────────────────

export function MeasurementTable({
  indicators,
  diagnostics,
}: {
  indicators: import("@nof1-causal-lab/api-types").Indicator[];
  diagnostics: Record<string, LikelihoodDiagnostics | undefined>;
}) {
  const rows: MeasurementRow[] = useMemo(
    () =>
      indicators.flatMap((indicator) =>
        indicator.likelihood
          ? [
              {
                likelihood: indicator.likelihood,
                label: indicator.name,
                diagnostics: diagnostics[indicator.id],
              },
            ]
          : [],
      ),
    [indicators, diagnostics],
  );

  const columns = baseColumns;

  return <InfoTable columns={columns} data={rows} estimateRowHeight={88} />;
}
