import { DAG_COLORS, signColor } from "@/components/dag/core/palette";
import type { AnalysisSimulationResult } from "@/components/dag/intervention-dag-types";
import { formatClampValue } from "@/components/dag/intervention-dag-semantics";
import { formatSigned } from "./model-selection";

/** The effect on the outcome over the horizon, with the 95% interval at the end. */
export function EffectChart({
  simulation,
  width = 340,
}: {
  simulation: AnalysisSimulationResult;
  width?: number;
}) {
  const trajectory = simulation.effect_trajectory ?? [];
  if (trajectory.length < 2) {
    return null;
  }
  const height = 150;
  const x0 = 34;
  const x1 = width - 118;
  const y0 = 18;
  const y1 = height - 30;
  const horizon = trajectory[trajectory.length - 1].day;
  const values = [
    0,
    ...trajectory.map((point) => point.effect),
    simulation.summary.lower_95,
    simulation.summary.upper_95,
  ];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const sx = (day: number) => x0 + (day / horizon) * (x1 - x0);
  const sy = (value: number) => y1 - ((value - lo) / (hi - lo || 1)) * (y1 - y0);
  const path = trajectory
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${sx(point.day).toFixed(1)},${sy(point.effect).toFixed(1)}`,
    )
    .join("");
  const color = signColor(simulation.summary.mean);
  const clamp = simulation.request.clamps[0];
  const clampEnd = clamp.to_day ?? horizon;
  const axisDays = [0, 0.25, 0.5, 0.75, 1].map((fraction) => Math.round(horizon * fraction));
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ display: "block", maxWidth: width }}
      role="img"
      aria-label="Effect on the outcome over the horizon"
    >
      <line x1={x0} x2={x1} y1={sy(0)} y2={sy(0)} stroke={DAG_COLORS.line2} />
      <rect
        x={sx(clamp.from_day)}
        y={y1 + 6}
        width={Math.max(2, sx(clampEnd) - sx(clamp.from_day))}
        height={5}
        rx={2}
        fill={DAG_COLORS.intervention}
      />
      <text
        x={sx(clamp.from_day)}
        y={y1 + 22}
        fontSize={8.5}
        fill={DAG_COLORS.intervention}
        fontWeight={600}
      >
        do · {simulation.labels[clamp.target]} {formatClampValue(clamp)} · d{clamp.from_day}–
        {clampEnd}
      </text>
      {axisDays.map((day) => (
        <text
          key={day}
          x={sx(day)}
          y={y1 + 32}
          fontSize={8}
          fill={DAG_COLORS.muted}
          textAnchor="middle"
        >
          d{day}
        </text>
      ))}
      <text
        x={x0 - 6}
        y={sy(0)}
        fontSize={8}
        fill={DAG_COLORS.muted}
        textAnchor="end"
        dominantBaseline="middle"
      >
        0
      </text>
      <path d={path} fill="none" stroke={color} strokeWidth={2} strokeOpacity={0.95} />
      <line
        x1={x1 + 3}
        x2={x1 + 3}
        y1={sy(simulation.summary.lower_95)}
        y2={sy(simulation.summary.upper_95)}
        stroke={color}
        strokeWidth={2}
      />
      <text
        x={x1 + 12}
        y={sy(simulation.summary.mean)}
        fontSize={8.5}
        fontWeight={650}
        fontFamily="ui-monospace, monospace"
        fill={color}
        dominantBaseline="middle"
      >
        posterior {formatSigned(simulation.summary.mean, 3)}
      </text>
    </svg>
  );
}
