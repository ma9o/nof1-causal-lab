"use client";

import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/utils/format";

type ManifestEffects = Record<string, number | undefined> | null | undefined;

/** Indicator-level projection of an outcome effect through the measurement loadings. */
export function ManifestProjection({
  manifestEffects,
  className,
}: {
  manifestEffects: ManifestEffects;
  className?: string;
}) {
  const entries = Object.entries(manifestEffects ?? {})
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]));
  if (entries.length === 0) return null;

  return (
    <div className={cn("flex items-start gap-4", className)}>
      <span className="shrink-0 pt-px text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        Indicators
      </span>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {entries.map(([indicator, value]) => (
          <span key={indicator} className="inline-flex items-baseline gap-1.5">
            <span className="text-muted-foreground">{indicator}</span>
            <span className="font-mono text-xs tabular-nums">
              {value >= 0 ? "+" : ""}
              {formatNumber(value)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
