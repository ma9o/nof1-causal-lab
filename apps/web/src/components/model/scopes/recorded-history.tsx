import type {
  IndicatorId,
  ModelSnapshot,
  PathSeries,
  SimulationSeriesSummary,
  EmpiricalPoint,
  DataVariableDiff,
} from "@nof1-causal-lab/api-types";
import { useState } from "react";
import { HistoryPlot, PATH_COLORS, type HistoryLine } from "@/components/charts/history-plot";
import { PlotNumberInput } from "@/components/charts/plot-number-input";
import {
  useObservationHistory,
  usePredictiveHistory,
  useSimulationPaths,
} from "@/lib/hooks/use-visuals";
import { Hint } from "../scope-primitives";
import { COMPARISON_COLORS } from "@/lib/dag/palette";

/** Align saved timestamps for plotting; no resampling, imputation or pooled statistics. */
export function dataComparisonHistory(variable: DataVariableDiff) {
  const histories = [...variable.left, ...variable.right];
  const definition = histories.find((history) => history.variable !== null)!.variable!;
  const anchors = [
    ...new Set(histories.flatMap((history) => history.points.map((point) => point.anchor_time))),
  ].sort();
  const origin = anchors[0];
  const series: HistoryLine[] = (["left", "right"] as const).flatMap((side) =>
    variable[side].map((history, index) => {
      const points = new Map(history.points.map((point) => [point.anchor_time, point.value]));
      const reference = variable.reference_side === side;
      return {
        id: `${side}-${index}`,
        label: `${reference ? "Observed" : side} · history ${index + 1}`,
        values: anchors.map((anchor) => points.get(anchor) ?? null),
        color: reference ? "var(--foreground)" : side === "left" ? "#64748b" : "#0ea5e9",
        emphasized: reference,
        dashed: side === "left" && variable.reference_side === null,
      };
    }),
  );
  // Reference observations sit above all replicas.
  series.sort((a, b) => Number(a.emphasized) - Number(b.emphasized));
  for (const change of ["added", "removed", "revised"] as const) {
    const points = variable.changes.filter((point) => point.change === change);
    if (points.length === 0) continue;
    for (const side of ["left", "right"] as const) {
      const values = new Map(
        points.map((point) => [point.anchor_time, point[side]?.value ?? null]),
      );
      series.push({
        id: `${change}-${side}`,
        label: `${change} · ${side}`,
        values: anchors.map((anchor) => values.get(anchor) ?? null),
        color: COMPARISON_COLORS[change],
        emphasized: true,
        dashed: side === "left",
      });
    }
  }
  return {
    label: `${definition.name}: data comparison`,
    xLabel: "Days from first anchor",
    times: anchors.map((anchor) => (Date.parse(anchor) - Date.parse(origin)) / 86400000),
    timeOrigin: histories.every(
      (history) => history.variable === null || history.time_origin !== null,
    )
      ? origin
      : null,
    series,
    pointsOnly: variable.reference_side === null,
    levels: definition.ordinal_levels ?? definition.categorical_levels,
  };
}

export function DrawPager({
  start,
  count,
  total,
  onStart,
  onCount,
  source = "saved",
}: {
  start: number;
  count: number;
  total: number;
  onStart: (value: number) => void;
  onCount: (value: number) => void;
  source?: "saved" | "current-law";
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-[10px]">
      <button
        type="button"
        disabled={start === 0}
        className="disabled:opacity-30"
        onClick={() => onStart(Math.max(0, start - count))}
      >
        ←
      </button>
      <label>
        First draw{" "}
        <PlotNumberInput
          aria-label="First draw"
          min={1}
          max={total}
          value={start + 1}
          className="w-16 rounded border bg-background px-1"
          onValue={(value) => onStart(value - 1)}
        />
      </label>
      <label>
        Show{" "}
        <select
          aria-label="Draws per page"
          className="rounded border bg-background"
          value={count}
          onChange={(e) => onCount(Number(e.target.value))}
        >
          {[1, 24, 64, 128].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        disabled={start + count >= total}
        className="disabled:opacity-30"
        onClick={() => onStart(start + count)}
      >
        →
      </button>
      <span>
        {start + 1}–{Math.min(start + count, total)} of {total.toLocaleString()} {source} draws
      </span>
    </div>
  );
}

export function pathLines(series: PathSeries): HistoryLine[] {
  return [
    ...series.reference.map((path) => ({
      id: `reference-${path.draw}`,
      label: `Reference draw ${path.draw + 1}`,
      values: path.values,
      color: PATH_COLORS[path.draw % PATH_COLORS.length],
      dashed: true,
    })),
    ...series.action.map((path) => ({
      id: `action-${path.draw}`,
      label: `${series.reference.length ? "Intervened" : "Simulation"} draw ${path.draw + 1}`,
      values: path.values,
      color: PATH_COLORS[path.draw % PATH_COLORS.length],
    })),
  ];
}

export function SimulationHistory({
  model,
  id,
  kind,
  summary,
}: {
  model: ModelSnapshot;
  id: string;
  kind: "states" | "indicators" | "effect";
  summary?: SimulationSeriesSummary;
}) {
  const [start, setStart] = useState(0);
  const [count, setCount] = useState(24);
  const [showMean, setShowMean] = useState(false);
  const paths = useSimulationPaths(model, start, count);
  if (paths.error) return <Hint issue>{paths.error.message}</Hint>;
  if (!paths.data)
    return (
      <Hint>
        {paths.isLoading ? "Loading recorded paths…" : "No saved paths at this revision."}
      </Hint>
    );
  const series = kind === "effect" ? paths.data.effect : paths.data[kind][id];
  if (!series) return <Hint>No recorded series for this entity.</Hint>;
  const lines = pathLines(series);
  const description = [
    kind === "indicators"
      ? "Points are sampled observations at their recorded anchors."
      : "Each line is one saved draw, joined only between consecutive recorded points.",
    series.reference.length > 0
      ? kind === "indicators"
        ? "Reference observations are hollow; intervened observations are filled. Matching draw numbers are paired."
        : "Reference paths are dashed; intervened paths are solid. Matching draw numbers are paired."
      : "",
    "Missing and nonfinite values remain gaps.",
  ].join(" ");
  if (showMean && summary?.action.kind === "numeric") {
    lines.push({
      id: "mean",
      label: "Pointwise mean",
      values: summary.action.mean,
      color: "var(--foreground)",
      emphasized: true,
    });
    if (summary.reference?.kind === "numeric")
      lines.push({
        id: "reference-mean",
        label: "Reference pointwise mean",
        values: summary.reference.mean,
        color: "var(--foreground)",
        emphasized: true,
        dashed: true,
      });
  }
  return (
    <>
      <DrawPager
        start={start}
        count={count}
        total={paths.data.total_draws}
        onStart={setStart}
        onCount={setCount}
      />
      <HistoryPlot
        times={paths.data.times}
        series={lines}
        label={`${series.label}: recorded ${kind === "effect" ? "paired effects" : "draws"}`}
        timeOrigin={paths.data.time_origin}
        pointsOnly={kind === "indicators"}
        levels={series.levels}
        markers={model.findings.simulation?.value.design.interventions.map((event) => ({
          time: event.time,
          label: `Day ${event.time}: set to ${event.value}`,
        }))}
        description={description}
      />
      {summary?.action.kind === "numeric" && (
        <label className="text-[10px]">
          <input
            type="checkbox"
            checked={showMean}
            onChange={(e) => setShowMean(e.target.checked)}
          />{" "}
          Overlay pointwise mean
        </label>
      )}
    </>
  );
}

export function ObservationPlots({ model, id }: { model: ModelSnapshot; id: IndicatorId }) {
  const history = useObservationHistory(model, id);
  if (history.error) return <Hint issue>{history.error.message}</Hint>;
  if (!history.data)
    return (
      <Hint>
        {history.isLoading ? "Loading observations…" : "No prepared observations at this revision."}
      </Hint>
    );
  const data = history.data;
  return (
    <>
      <HistoryPlot
        times={data.times}
        series={[
          {
            id: "observed",
            label: data.label,
            values: data.values,
            emphasized: true,
            color: "var(--foreground)",
          },
        ]}
        label={`${data.label}: prepared observations`}
        timeOrigin={data.time_origin}
        pointsOnly
        levels={data.levels}
        support={{ start: data.support_start, end: data.support_end }}
        description="Every prepared observation at its actual time. Horizontal marks show measurement windows, not persistence between observations."
      />
      {data.empirical.length > 0 && (
        <EmpiricalPlot points={data.empirical} label={data.label} xLabel="Observed value" />
      )}
    </>
  );
}

export function EmpiricalPlot({
  points,
  label,
  xLabel,
}: {
  points: EmpiricalPoint[];
  label: string;
  xLabel: string;
}) {
  return (
    <>
      <HistoryPlot
        times={[points[0].value, ...points.map((point) => point.value)]}
        series={[
          {
            id: "empirical",
            label: "Empirical cumulative probability",
            values: [0, ...points.map((point) => point.probability)],
            emphasized: true,
          },
        ]}
        label={`${label}: empirical distribution`}
        xLabel={xLabel}
        yLabel="Cumulative probability"
        step
        description="Exact empirical distribution: each jump retains the count at that value. No binning or smoothing."
      />
    </>
  );
}

export function PredictiveHistoryPlot({ model, id }: { model: ModelSnapshot; id: IndicatorId }) {
  const history = usePredictiveHistory(model, id);
  if (history.error) return <Hint issue>{history.error.message}</Hint>;
  if (!history.data)
    return (
      <Hint>
        {history.isLoading ? "Loading predictive history…" : "No saved predictive history."}
      </Hint>
    );
  const { times, time_origin, overlay, standardized } = history.data;
  return (
    <>
      <HistoryPlot
        times={times}
        timeOrigin={time_origin}
        label="Observed and replicated observations"
        pointsOnly
        series={[
          ...overlay.spaghetti_draws.map((values, index) => ({
            id: `replicate-${index}`,
            label: `Replicate ${index + 1}`,
            values,
            color: PATH_COLORS[index % PATH_COLORS.length],
          })),
          {
            id: "observed",
            label: "Observed",
            values: overlay.observed,
            color: "var(--foreground)",
            emphasized: true,
          },
        ]}
      />
      <Hint>
        Observed points (dark) and all {overlay.spaghetti_draws.length} retained replicated series
        on the actual time axis. Gaps are not joined.{" "}
        {standardized
          ? "Values are on the model’s standardized observation scale."
          : "Values are on the observation scale."}
      </Hint>
    </>
  );
}
