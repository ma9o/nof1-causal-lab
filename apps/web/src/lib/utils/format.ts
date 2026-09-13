import type { PosteriorEstimate } from "@nof1-causal-lab/api-types";

export function formatNumber(n: number, decimals = 3): string {
  if (Number.isNaN(n)) return "NaN";
  if (!Number.isFinite(n)) return n > 0 ? "+Inf" : "-Inf";
  return n.toFixed(decimals);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

const compactFormatter = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export function formatCompact(n: number): string {
  return compactFormatter.format(n);
}

const posteriorMassFormatter = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 2,
});

export function formatPosteriorIntervalLabel(
  estimate: Pick<PosteriorEstimate, "interval_kind" | "interval_mass">,
): string {
  const kind = { hdi: "HDI", equal_tail: "equal-tail interval" }[estimate.interval_kind];
  return `${posteriorMassFormatter.format(estimate.interval_mass)} ${kind}`;
}
