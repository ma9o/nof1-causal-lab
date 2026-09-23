import type { ParameterSpec } from "@nof1-causal-lab/api-types";
import { distributionText } from "@/lib/utils/distribution-format";
import type {
  ModelSpecAdmissionCheckResult,
  ModelSpecAdmissionConstructState,
  ModelSpecAdmissionConstructStatus,
  ModelSpecAdmissionReport,
  ModelSpecAdmissionReplayState,
  ModelSpecAdmissionTiming,
} from "@/lib/model-spec-admission-runtime";

export type AdmissionTimelineEntry = {
  key: string;
  status: ModelSpecAdmissionConstructStatus;
  results: ModelSpecAdmissionCheckResult[];
  timings: ModelSpecAdmissionTiming[];
} & (
  | { kind: "attempt"; attempt: number; durationMs?: number }
  | { kind: "recheck"; originator: string; closingEdges: string[] }
);

export function titleize(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function constructLabel(
  construct: ModelSpecAdmissionConstructState | null | undefined,
): string {
  if (!construct) return "Construct admission";
  return construct.label ?? titleize(construct.name);
}

export function formatCheckDuration(ms: number): string {
  if (ms > 0 && ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  return `${seconds >= 10 ? Math.round(seconds) : seconds.toFixed(1)}s`;
}

export function formatPriorSummary(
  param: ParameterSpec,
  distributions: import("@nof1-causal-lab/api-types").ModelSpec["distributions"] | undefined,
): string {
  if (param.value != null) return `Fixed: ${param.value}`;
  return param.distribution && distributions
    ? distributionText(distributions[param.distribution])
    : "Not authored";
}

export function reportKeyFor(
  constructName: string,
  report: ModelSpecAdmissionReport,
  index: number,
): string {
  return `${constructName}:${report.attempt}:${index}`;
}

export function progressCounts(state: ModelSpecAdmissionReplayState) {
  const admitted = state.constructs.filter(
    (construct) =>
      construct.status === "admitted" || construct.status === "admitted_with_consequences",
  ).length;
  const revising = state.constructs.filter((construct) => construct.status === "revising").length;
  const blocked = state.constructs.filter((construct) => construct.status === "blocked").length;
  return { admitted, revising, blocked, total: state.constructs.length };
}

export function checkpointLabel(ref: string): string {
  return ref.split("/").at(-1) ?? ref;
}

export function getFeaturedConstruct(state: ModelSpecAdmissionReplayState) {
  return (
    state.constructs.find((construct) => construct.name === state.activeConstructs[0]) ??
    (state.latestReport
      ? state.constructs.find((construct) => construct.name === state.latestReport?.name)
      : undefined) ??
    [...state.constructs].reverse().find((construct) => construct.status !== "pending") ??
    state.constructs[0] ??
    null
  );
}

export function truncateDagLabel(label: string): string {
  return label.length > 20 ? `${label.slice(0, 17)}...` : label;
}

export function statusColorVar(
  status: ModelSpecAdmissionConstructStatus,
  isSelected: boolean,
): string {
  if (isSelected) return "var(--primary)";
  switch (status) {
    case "active":
    case "checking":
      return "var(--primary)";
    case "revising":
    case "admitted_with_consequences":
      return "var(--warning)";
    case "admitted":
      return "var(--success)";
    case "blocked":
      return "var(--destructive)";
    default:
      return "var(--muted-foreground)";
  }
}

export function resultsStatus(
  results: readonly ModelSpecAdmissionCheckResult[],
): ModelSpecAdmissionConstructStatus {
  if (results.some((result) => !result.passed && result.mode === "hard")) return "blocked";
  if (results.some((result) => !result.passed)) return "revising";
  return "admitted";
}

export function reportStatus(report: ModelSpecAdmissionReport): ModelSpecAdmissionConstructStatus {
  if (!report.admitted) return resultsStatus(report.results);
  return report.annotations.length > 0 ? "admitted_with_consequences" : "admitted";
}

export function constructColorStatus(
  construct: ModelSpecAdmissionConstructState,
): ModelSpecAdmissionConstructStatus {
  if (
    construct.status === "pending" ||
    construct.status === "active" ||
    construct.status === "checking"
  ) {
    return construct.status;
  }
  const last = construct.reports[construct.reports.length - 1];
  return last ? reportStatus(last) : construct.status;
}

export function buildTimeline(
  state: ModelSpecAdmissionReplayState,
  construct: ModelSpecAdmissionConstructState,
): AdmissionTimelineEntry[] {
  const attempts: AdmissionTimelineEntry[] = construct.reports.map((report, index) => ({
    kind: "attempt",
    key: reportKeyFor(construct.name, report, index),
    status: reportStatus(report),
    results: report.results,
    timings: report.timings,
    attempt: report.attempt,
    durationMs: report.durationMs,
  }));
  const rechecks: AdmissionTimelineEntry[] = [];
  for (const other of state.constructs) {
    if (other.name === construct.name) continue;
    other.reports.forEach((report, index) => {
      const recheck = report.coupled_recheck;
      if (recheck?.constructs.includes(construct.name)) {
        rechecks.push({
          kind: "recheck",
          key: `recheck:${other.name}:${report.attempt}:${index}`,
          status: resultsStatus(recheck.results),
          results: recheck.results,
          timings: recheck.timings,
          originator: other.name,
          closingEdges: recheck.closing_edges ?? [],
        });
      }
    });
  }
  return [...attempts, ...rechecks];
}

export function statusTintClasses(status: ModelSpecAdmissionConstructStatus): string {
  switch (status) {
    case "active":
    case "checking":
      return "border-primary/30 bg-primary/5";
    case "revising":
    case "admitted_with_consequences":
      return "border-warning/40 bg-warning/10";
    case "admitted":
      return "border-success/30 bg-success/5";
    case "blocked":
      return "border-destructive/30 bg-destructive/5";
    default:
      return "border-border bg-muted/20";
  }
}

export const DAG_NODE_W = 152;
export const DAG_NODE_H = 40;
export const DAG_PAD = 10;
