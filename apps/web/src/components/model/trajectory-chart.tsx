import type { SimulationTrajectoryBands } from "@nof1-causal-lab/api-types";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { humanize } from "@/lib/model-asset/selection";
import { formatModelDate } from "@/lib/utils/format";

/** Plot server-provided means and intervals; missing measurement anchors remain gaps. */
export function TrajectoryChart({
  times,
  series,
  timeOrigin,
}: {
  times: number[];
  series: SimulationTrajectoryBands;
  timeOrigin?: string | null;
}) {
  const data = times.map((time, index) => ({
    time,
    action: series.action.mean[index],
    actionBand:
      series.action.lower[index] == null
        ? null
        : [series.action.lower[index], series.action.upper[index]],
    reference: series.reference?.mean[index],
    referenceBand:
      series.reference?.lower[index] == null
        ? null
        : [series.reference.lower[index], series.reference.upper[index]],
  }));
  return (
    <div
      className="h-52 w-full"
      role="img"
      aria-label={`${humanize(series.label)}: simulated means and 95% pointwise intervals`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="time"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 10 }}
            tickFormatter={(day: number) =>
              timeOrigin ? formatModelDate(day, timeOrigin) : String(day)
            }
            minTickGap={20}
            label={{
              value: timeOrigin ? "Date (UTC)" : "Model day",
              position: "insideBottom",
              offset: -6,
              fontSize: 10,
            }}
          />
          <YAxis
            domain={["auto", "auto"]}
            width={45}
            tick={{ fontSize: 10 }}
            tickFormatter={(value: number) =>
              value.toLocaleString(undefined, { maximumSignificantDigits: 3 })
            }
          />
          <Tooltip
            labelFormatter={(value) =>
              timeOrigin
                ? `${formatModelDate(Number(value), timeOrigin)} · model day ${value}`
                : `Model day ${value}`
            }
            contentStyle={{ fontSize: 11 }}
          />
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
          {series.reference && (
            <Area
              dataKey="referenceBand"
              name="Reference 95% interval"
              stroke="none"
              fill="#64748b"
              fillOpacity={0.18}
              legendType="none"
              isAnimationActive={false}
            />
          )}
          <Area
            dataKey="actionBand"
            name="Simulated 95% interval"
            stroke="none"
            fill="#2563eb"
            fillOpacity={0.15}
            legendType="none"
            isAnimationActive={false}
          />
          {series.reference && (
            <Line
              dataKey="reference"
              name="Reference mean"
              stroke="#64748b"
              strokeDasharray="4 3"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          )}
          <Line
            dataKey="action"
            name={series.reference ? "Intervened mean" : "Simulated mean"}
            stroke="#2563eb"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
