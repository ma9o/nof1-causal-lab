"use client";

import { formatNumber } from "@/lib/utils/format";
import type { LOODiagnostics, ParetoKPoint } from "@nof1-causal-lab/api-types";
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

interface ParetoKChartProps {
  loo: LOODiagnostics;
  points: readonly ParetoKPoint[];
}

export function ParetoKChart({ loo, points }: ParetoKChartProps) {
  if (points.length === 0) return null;
  const PARETO_K_FAIL = loo.pareto_failure_limit;
  const PARETO_K_WARN = loo.pareto_warning_limit;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-mono text-muted-foreground">Pareto k (sorted)</span>
        <span className="text-[10px] font-mono text-muted-foreground">
          {loo.n_bad_k ?? "Unavailable"} &gt; {PARETO_K_FAIL} · {loo.n_warn_k ?? "Unavailable"} &gt;{" "}
          {PARETO_K_WARN} · n = {points.length}
        </span>
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 10, right: 40, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="rank"
              type="number"
              domain={[1, points.length]}
              tick={{ fontSize: 10 }}
              label={{ value: "Rank", position: "insideBottom", offset: -2, fontSize: 10 }}
            />
            <YAxis
              dataKey={(point: ParetoKPoint) => (typeof point.k === "number" ? point.k : null)}
              tick={{ fontSize: 10 }}
              label={{
                value: "Pareto k",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                fontSize: 10,
              }}
            />
            <RechartsTooltip
              formatter={(value) => [formatNumber(Number(value), 3), "Pareto k"]}
              labelFormatter={(label: unknown) => {
                const point = points.find((point) => point.rank === Number(label));
                return `rank ${String(label)}${point ? ` (timestep ${point.timestep})` : ""}`;
              }}
            />
            <ReferenceLine
              y={PARETO_K_FAIL}
              stroke="var(--destructive)"
              strokeDasharray="4 4"
              label={{
                value: `k = ${PARETO_K_FAIL}`,
                position: "right",
                fontSize: 9,
                fill: "var(--destructive)",
              }}
            />
            <ReferenceLine
              y={PARETO_K_WARN}
              stroke="var(--warning)"
              strokeDasharray="4 4"
              label={{
                value: `k = ${PARETO_K_WARN}`,
                position: "right",
                fontSize: 9,
                fill: "var(--warning)",
              }}
            />
            <Line
              dataKey={(point: ParetoKPoint) => (typeof point.k === "number" ? point.k : null)}
              type="linear"
              stroke="var(--primary)"
              strokeWidth={1.25}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        Pareto k diagnostic sorted from largest to smallest. Timesteps with k &gt; {PARETO_K_FAIL}{" "}
        (left edge) are highly influential and the LOO estimate may be unreliable. Hover any rank to
        recover the original timestep.
      </p>
    </div>
  );
}
