"use client";

import type { PosteriorPair } from "@nof1-causal-lab/api-types";
import { HistoryPlot, type HistoryLine } from "./history-plot";

export function PosteriorPairsChart({ pair }: { pair: PosteriorPair }) {
  const flags = pair.divergent ?? [];
  const series: HistoryLine[] = [
    {
      id: "retained",
      label: "Retained draw",
      color: "var(--primary)",
      values: pair.y_values.map((value, index) => (flags[index] ? null : value)),
    },
  ];
  if (flags.some(Boolean))
    series.push({
      id: "divergent",
      label: "Divergent draw",
      color: "var(--destructive)",
      values: pair.y_values.map((value, index) => (flags[index] ? value : null)),
      emphasized: true,
    });
  return (
    <HistoryPlot
      times={pair.x_values}
      series={series}
      label={`${pair.param_x} vs ${pair.param_y}: joint draws`}
      xLabel={pair.param_x}
      yLabel={pair.param_y}
      pointsOnly
      description="Each point pairs coordinates from the same retained draw. No jitter, thinning or smoothing."
    />
  );
}
