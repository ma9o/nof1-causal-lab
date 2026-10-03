"use client";

import type { ParameterDrawColumn } from "@nof1-causal-lab/api-types";
import { HistoryPlot, type HistoryLine } from "./history-plot";

export function PosteriorPairsChart({
  x,
  y,
  divergent,
}: {
  x: Pick<ParameterDrawColumn, "label" | "values">;
  y: Pick<ParameterDrawColumn, "label" | "values">;
  divergent: readonly boolean[] | null;
}) {
  const flags = divergent ?? [];
  const series: HistoryLine[] = [
    {
      id: "retained",
      label: "Retained draw",
      color: "var(--primary)",
      values: y.values.map((value, index) => (flags[index] ? null : value)),
    },
  ];
  if (flags.some(Boolean))
    series.push({
      id: "divergent",
      label: "Divergent draw",
      color: "var(--destructive)",
      values: y.values.map((value, index) => (flags[index] ? value : null)),
      emphasized: true,
    });
  return (
    <HistoryPlot
      times={x.values}
      series={series}
      label={`${x.label} vs ${y.label}: joint draws`}
      xLabel={x.label}
      yLabel={y.label}
      pointsOnly
      description="Each point pairs coordinates from the same retained draw. No jitter, thinning or smoothing."
    />
  );
}
