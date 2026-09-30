import type { IndicatorEmpiricalProfile } from "@nof1-causal-lab/api-types";
import { scaleLinear } from "d3-scale";
import { formatSignificant } from "@/lib/utils/format";

const WIDTH = 256;
const HEIGHT = 34;
const AXIS = 11;
/** Room a median label needs to clear the range labels at both ends. */
const LABEL_CLEARANCE = 34;

/** The profile's recorded range, quartiles, median and mean, drawn as they were measured. */
export function QuantileStrip({ profile }: { profile: IndicatorEmpiricalProfile }) {
  const { min, q25, q50, q75, max, mean } = profile;
  if (min == null || q25 == null || q50 == null || q75 == null || max == null) return null;
  const sx = scaleLinear()
    .domain(min === max ? [min - 0.5, max + 0.5] : [min, max])
    .range([4, WIDTH - 4]);
  const median = sx(q50);
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="w-full max-w-[256px]"
      role="img"
      aria-label={`Prepared values from ${formatSignificant(min)} to ${formatSignificant(max)}, quartiles ${formatSignificant(q25)} to ${formatSignificant(q75)}, median ${formatSignificant(q50)}`}
    >
      <title>
        {`min ${formatSignificant(min)} · q25 ${formatSignificant(q25)} · median ${formatSignificant(q50)} · q75 ${formatSignificant(q75)} · max ${formatSignificant(max)}${mean != null ? ` · mean ${formatSignificant(mean)}` : ""}`}
      </title>
      <line
        x1={sx(min)}
        x2={sx(max)}
        y1={AXIS}
        y2={AXIS}
        stroke="var(--muted-foreground)"
        strokeWidth={1}
      />
      <rect
        x={sx(q25)}
        y={AXIS - 6}
        width={Math.max(1.5, sx(q75) - sx(q25))}
        height={12}
        rx={2}
        fill="var(--chart-3)"
        fillOpacity={0.18}
        stroke="var(--chart-3)"
      />
      <line
        x1={median}
        x2={median}
        y1={AXIS - 6}
        y2={AXIS + 6}
        stroke="var(--chart-3)"
        strokeWidth={2}
      />
      {mean != null && <circle cx={sx(mean)} cy={AXIS} r={2.2} fill="var(--foreground)" />}
      <text x={4} y={HEIGHT - 2} fontSize={8.5} fill="var(--muted-foreground)">
        {formatSignificant(min)}
      </text>
      {median - 4 > LABEL_CLEARANCE && WIDTH - 4 - median > LABEL_CLEARANCE && (
        <text x={median} y={HEIGHT - 2} textAnchor="middle" fontSize={8.5} fill="var(--foreground)">
          {formatSignificant(q50)}
        </text>
      )}
      <text
        x={WIDTH - 4}
        y={HEIGHT - 2}
        textAnchor="end"
        fontSize={8.5}
        fill="var(--muted-foreground)"
      >
        {formatSignificant(max)}
      </text>
    </svg>
  );
}
