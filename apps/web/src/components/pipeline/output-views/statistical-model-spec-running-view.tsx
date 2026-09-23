"use client";

import { useAdmissionView, useAdmissionGraph } from "@/lib/admission/use-admission-view";
import {
  titleize,
  constructLabel,
  formatCheckDuration,
  formatPriorSummary,
  checkpointLabel,
  truncateDagLabel,
  statusColorVar,
  constructColorStatus,
  statusTintClasses,
  DAG_PAD,
  type AdmissionTimelineEntry,
} from "@/lib/admission/presentation";

import { DagEdge } from "@/components/dag/core/dag-edge";
import { DagNodeShell } from "@/components/dag/core/dag-node";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  type ModelSpecAdmissionCheckResult,
  type ModelSpecAdmissionConstructState,
  type ModelSpecAdmissionConstructStatus,
  type ModelSpecAdmissionReplayState,
  type ModelSpecAdmissionTiming,
  useModelSpecAdmission,
} from "@/lib/hooks/use-model-spec-admission";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  CheckCircle2,
  Circle,
  Clock,
  FlaskConical,
  Loader2,
  RotateCcw,
  XCircle,
} from "lucide-react";
import type { KeyboardEvent } from "react";

function StatusIcon({ status }: { status: ModelSpecAdmissionConstructStatus }) {
  if (status === "active") return <Loader2 className="h-3.5 w-3.5 animate-spin" />;
  if (status === "checking") return <FlaskConical className="h-3.5 w-3.5" />;
  if (status === "revising") return <RotateCcw className="h-3.5 w-3.5" />;
  if (status === "blocked") return <XCircle className="h-3.5 w-3.5" />;
  if (status === "admitted_with_consequences") {
    return <AlertTriangle className="h-3.5 w-3.5" />;
  }
  if (status === "admitted") return <CheckCircle2 className="h-3.5 w-3.5" />;
  return <Circle className="h-3.5 w-3.5" />;
}

function ResumeSummary({ state }: { state: ModelSpecAdmissionReplayState }) {
  const resume = state.resume;
  if (!resume) return null;
  const retainedCount = resume.retainedConstructs.length;

  return (
    <div className="flex items-start gap-3 rounded-lg border border-primary/25 bg-primary/5 px-3 py-2.5">
      <RotateCcw className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
      <div className="min-w-0 space-y-1">
        <div className="text-sm font-medium">
          {resume.pinsChanged ? "Rebased after upstream changes" : "Resumed from checkpoint"}
        </div>
        <p className="text-xs text-muted-foreground">
          {resume.pinsChanged
            ? `${retainedCount} accepted construct${retainedCount === 1 ? "" : "s"} revalidated and retained.`
            : `${retainedCount} accepted construct${retainedCount === 1 ? "" : "s"} restored without rerunning checks.`}
          {resume.reopenedConstruct
            ? ` Reopened ${titleize(resume.reopenedConstruct)}.`
            : " No construct needed reopening."}
        </p>
        {resume.reason && <p className="text-xs text-muted-foreground">{resume.reason}</p>}
        <div className="truncate font-mono text-[11px] text-muted-foreground/70">
          {checkpointLabel(resume.sourceCheckpointRef)} → {checkpointLabel(resume.checkpointRef)}
        </div>
      </div>
    </div>
  );
}

/** The single status signal used everywhere: an icon (specific state) in the status color. */
function StatusIndicator({ status }: { status: ModelSpecAdmissionConstructStatus }) {
  return (
    <span className="shrink-0" style={{ color: statusColorVar(status, false) }}>
      <StatusIcon status={status} />
    </span>
  );
}

function MiniConstructDag({
  state,
  selectedName,
  onSelectConstruct,
}: {
  state: ModelSpecAdmissionReplayState;
  selectedName: string | null | undefined;
  onSelectConstruct: (constructName: string) => void;
}) {
  const { edges, width, height, isLayouting, geoByName, constructByName, orderByName } =
    useAdmissionGraph(state);

  return (
    <div className="max-h-[320px] overflow-auto rounded-lg border bg-muted/20">
      {isLayouting ? (
        <div className="flex h-[180px] items-center justify-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Laying out DAG...
        </div>
      ) : (
        <svg
          width={width + DAG_PAD * 2}
          height={height + DAG_PAD * 2}
          viewBox={`0 0 ${width + DAG_PAD * 2} ${height + DAG_PAD * 2}`}
          role="img"
          aria-label="Retained measurement-structure construct DAG"
          className="block"
        >
          <g transform={`translate(${DAG_PAD} ${DAG_PAD})`}>
            <g>
              {edges.map((edge) => {
                const source = constructByName.get(edge.source);
                const target = constructByName.get(edge.target);
                const isFeedback =
                  (orderByName.get(edge.source) ?? 0) > (orderByName.get(edge.target) ?? 0);
                const isActive =
                  source?.status !== "pending" &&
                  source?.status !== "blocked" &&
                  target?.status !== "pending";
                const color = isFeedback
                  ? "var(--warning)"
                  : isActive
                    ? "var(--primary)"
                    : "var(--muted-foreground)";
                return (
                  <DagEdge
                    key={edge.id}
                    points={edge.points}
                    color={color}
                    width={1.6}
                    dashed={isFeedback}
                    opacity={isFeedback ? 0.6 : isActive ? 0.55 : 0.28}
                  />
                );
              })}
            </g>
            <g>
              {state.constructs.map((construct) => {
                const geo = geoByName.get(construct.name);
                if (!geo) return null;
                const isSelected = construct.name === selectedName;
                const label = constructLabel(construct);
                const colorStatus = constructColorStatus(construct);
                const isPending = colorStatus === "pending";
                const handleKeyDown = (event: KeyboardEvent<SVGGElement>) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onSelectConstruct(construct.name);
                };
                return (
                  <g
                    key={construct.name}
                    transform={`translate(${geo.x} ${geo.y})`}
                    role="button"
                    tabIndex={0}
                    aria-label={`Show ${label}`}
                    aria-pressed={isSelected}
                    className="cursor-pointer outline-none"
                    onClick={() => onSelectConstruct(construct.name)}
                    onKeyDown={handleKeyDown}
                  >
                    <DagNodeShell
                      width={geo.width}
                      height={geo.height}
                      accent={
                        isSelected || !isPending
                          ? statusColorVar(colorStatus, isSelected)
                          : undefined
                      }
                      highlighted={isSelected}
                    >
                      <title>{label}</title>
                      <circle
                        cx={15}
                        cy={geo.height / 2}
                        r={isSelected ? 4.5 : 3.5}
                        fill={statusColorVar(colorStatus, isSelected)}
                        fillOpacity={isPending && !isSelected ? 0.4 : 1}
                      />
                      <text
                        x={28}
                        y={geo.height / 2}
                        fontSize={11}
                        fontWeight={500}
                        fill="var(--foreground)"
                        dominantBaseline="middle"
                      >
                        {truncateDagLabel(label)}
                      </text>
                    </DagNodeShell>
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      )}
    </div>
  );
}

function ConstructQueueItem({
  construct,
  index,
  isSelected,
  onSelect,
}: {
  construct: ModelSpecAdmissionConstructState;
  index: number;
  isSelected: boolean;
  onSelect: (constructName: string) => void;
}) {
  return (
    <li>
      <button
        type="button"
        aria-pressed={isSelected}
        className={cn(
          "grid w-full grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-2 rounded-lg border px-3 py-2.5 text-left transition-colors",
          statusTintClasses(constructColorStatus(construct)),
          isSelected && "ring-1 ring-primary",
        )}
        onClick={() => onSelect(construct.name)}
      >
        <div className="flex h-6 w-6 items-center justify-center rounded-md border border-border/60 bg-background/60 text-xs tabular-nums text-muted-foreground">
          {index + 1}
        </div>
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate text-sm font-medium" title={constructLabel(construct)}>
              {constructLabel(construct)}
            </span>
          </div>
        </div>
        {construct.reports.length > 0 ? (
          <span
            className="rounded-md border border-border/60 bg-background/60 px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground"
            aria-label={`${construct.reports.length} ${construct.reports.length === 1 ? "attempt" : "attempts"}`}
            title={`${construct.reports.length} ${construct.reports.length === 1 ? "attempt" : "attempts"}`}
          >
            {construct.reports.length}×
          </span>
        ) : (
          <span className="text-xs tabular-nums text-muted-foreground/50" aria-hidden>
            —
          </span>
        )}
      </button>
    </li>
  );
}

function ConstructDetail({ construct }: { construct: ModelSpecAdmissionConstructState | null }) {
  if (!construct) {
    return <p className="text-sm text-muted-foreground">Waiting for the construct order.</p>;
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="truncate text-sm font-semibold" title={constructLabel(construct)}>
          {constructLabel(construct)}
        </h3>
      </div>

      <div className="space-y-3 text-sm">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Authored parameters</div>
          {construct.parameters && construct.parameters.length > 0 ? (
            <div className="overflow-hidden rounded-md border">
              <table className="w-full table-fixed text-xs">
                <thead>
                  <tr className="border-b bg-muted/30 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <th className="w-[55%] px-2 py-1 text-left font-medium">Parameter</th>
                    <th className="px-2 py-1 text-right font-medium">Prior</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {construct.parameters.map((param) => (
                    <tr key={param.name}>
                      <td className="px-2 py-1.5 align-top">
                        <span className="block truncate font-mono" title={param.name}>
                          {param.name}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-right align-top font-mono tabular-nums text-muted-foreground">
                        <span
                          className="block truncate"
                          title={formatPriorSummary(param, construct.distributions)}
                        >
                          {formatPriorSummary(param, construct.distributions)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-md border bg-muted/25 px-3 py-2 text-xs text-muted-foreground">
              none
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AttemptHistory({
  entries,
  selectedKey,
  onSelect,
}: {
  entries: AdmissionTimelineEntry[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  const attemptCount = entries.filter((entry) => entry.kind === "attempt").length;

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold">Attempts</h4>
        <span className="text-xs tabular-nums text-muted-foreground">
          {attemptCount} {attemptCount === 1 ? "attempt" : "attempts"}
        </span>
      </div>
      {entries.length === 0 ? (
        <p className="text-xs text-muted-foreground">Not checked yet.</p>
      ) : (
        <ol className="space-y-2">
          {entries.map((entry) => {
            const isSelected = entry.key === selectedKey;
            return (
              <li key={entry.key}>
                <button
                  type="button"
                  aria-pressed={isSelected}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-xs transition-colors",
                    statusTintClasses(entry.status),
                    isSelected && "ring-1 ring-primary",
                  )}
                  onClick={() => onSelect(entry.key)}
                >
                  <StatusIndicator status={entry.status} />
                  {entry.kind === "attempt" ? (
                    <span className="flex min-w-0 flex-1 items-center justify-between gap-2">
                      <span className="font-medium tabular-nums">Attempt {entry.attempt}</span>
                      {entry.durationMs !== undefined && (
                        <span
                          className="flex shrink-0 items-center gap-1 tabular-nums text-muted-foreground"
                          title="End-to-end check runtime"
                        >
                          <Clock className="h-3 w-3" aria-hidden />
                          {formatCheckDuration(entry.durationMs)}
                        </span>
                      )}
                    </span>
                  ) : (
                    <span className="min-w-0 truncate">
                      <span className="font-medium">Coupled recheck</span>
                      <span
                        className="ml-1.5 text-muted-foreground"
                        title={`Triggered when ${entry.originator} closed the feedback loop`}
                      >
                        from {entry.originator}
                      </span>
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function CheckRow({ result }: { result: ModelSpecAdmissionCheckResult }) {
  return (
    <li
      className={cn(
        "rounded-lg border p-3",
        result.passed
          ? "border-success/25 bg-success/5"
          : result.mode === "hard"
            ? "border-destructive/30 bg-destructive/5"
            : "border-warning/30 bg-warning/10",
      )}
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">{result.check}</span>
        </div>
        <div className="mt-1 text-xs text-muted-foreground">
          {result.target ? `${result.target}: ` : ""}
          {result.value}
        </div>
        {result.band && (
          <div className="mt-1 text-xs text-muted-foreground">Target: {result.band}</div>
        )}
      </div>
      {!result.passed && (
        <div className="mt-2 space-y-1 text-xs leading-relaxed text-muted-foreground">
          {result.note && <p>{result.note}</p>}
          {result.diagnosis?.map((diagnosis) => (
            <p key={diagnosis}>{diagnosis}</p>
          ))}
        </div>
      )}
    </li>
  );
}

function TimingBreakdown({ timings }: { timings: ModelSpecAdmissionTiming[] }) {
  if (timings.length === 0) return null;
  const totalMs = timings.reduce((total, timing) => total + timing.duration_ms, 0);

  return (
    <div className="overflow-hidden rounded-lg border bg-muted/10">
      <div className="flex items-center justify-between gap-3 border-b bg-muted/20 px-3 py-2">
        <h4 className="text-xs font-semibold">Backend timing breakdown</h4>
        <span className="text-xs tabular-nums text-muted-foreground">
          {formatCheckDuration(totalMs)} measured
        </span>
      </div>
      <dl className="max-h-[260px] divide-y divide-border overflow-auto">
        {timings.map((timing, index) => (
          <div
            key={`${timing.phase}-${index}`}
            className="flex items-start justify-between gap-3 px-3 py-2 text-xs"
          >
            <dt className="min-w-0">
              <span className="block font-medium">{timing.label}</span>
              {timing.checks.length > 0 && (
                <span className="mt-0.5 block truncate text-muted-foreground">
                  {timing.checks.join(", ")}
                </span>
              )}
            </dt>
            <dd className="shrink-0 font-mono tabular-nums text-muted-foreground">
              {formatCheckDuration(timing.duration_ms)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function ReachabilityPanel({ entry }: { entry: AdmissionTimelineEntry | null }) {
  if (!entry) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <div>
          <h3 className="text-sm font-semibold">Awaiting report</h3>
        </div>
        <div className="flex items-center gap-2 rounded-md border bg-muted/20 px-3 py-3 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Waiting for the first prior-predictive check.
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">
            {entry.kind === "recheck"
              ? `Loop closed by ${entry.originator}`
              : `Attempt ${entry.attempt}`}
          </h3>
          {entry.kind === "recheck" && entry.closingEdges.length > 0 && (
            <div className="mt-0.5 text-xs text-muted-foreground">
              Closing edge{entry.closingEdges.length > 1 ? "s" : ""}:{" "}
              {entry.closingEdges.join(", ")}
            </div>
          )}
        </div>
        {entry.kind === "attempt" && entry.durationMs !== undefined && (
          <div
            className="flex shrink-0 items-center gap-1 rounded-md border bg-muted/20 px-2 py-1 text-xs tabular-nums text-muted-foreground"
            title="End-to-end check runtime"
          >
            <Clock className="h-3 w-3" aria-hidden />
            {formatCheckDuration(entry.durationMs)}
          </div>
        )}
      </div>
      <TimingBreakdown timings={entry.timings} />
      <ol className="max-h-[520px] min-h-0 space-y-2 overflow-auto pr-1 xl:max-h-none xl:flex-1">
        {entry.results.map((result, index) => (
          <CheckRow key={`${result.check}-${result.target}-${index}`} result={result} />
        ))}
      </ol>
    </div>
  );
}

export function ModelSpecAdmissionRunningView({
  state,
  showError = true,
}: {
  state: ModelSpecAdmissionReplayState | null;
  showError?: boolean;
}) {
  const {
    counts,
    featuredConstruct,
    timeline,
    selectedEntry,
    progress,
    handleSelectConstruct,
    handleSelectEntry,
  } = useAdmissionView(state);

  if (!state?.plan || state.constructs.length === 0) {
    return (
      <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Preparing construct admission order...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card size="sm">
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="text-sm font-semibold">Construct admission</div>
              <div className="mt-1 text-sm text-muted-foreground">
                {counts.admitted} of {counts.total} constructs admitted
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {counts.revising > 0 && <Badge variant="warning">{counts.revising} revising</Badge>}
              {counts.blocked > 0 && <Badge variant="destructive">{counts.blocked} blocked</Badge>}
              {state.done && <Badge variant="success">done</Badge>}
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-md bg-muted">
            <div
              className="h-full rounded-md bg-primary transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <ResumeSummary state={state} />
          <MiniConstructDag
            state={state}
            selectedName={featuredConstruct?.name}
            onSelectConstruct={handleSelectConstruct}
          />
          {showError && state.error && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              {state.error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Master → detail → report reads left to right across one connected surface. */}
      <div className="grid divide-y divide-border overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10 xl:grid-cols-[minmax(280px,0.82fr)_minmax(360px,1fr)_minmax(360px,1.1fr)] xl:divide-x xl:divide-y-0">
        <section className="space-y-3 p-4">
          <h3 className="text-sm font-semibold">Construct queue</h3>
          <ol className="max-h-[720px] space-y-2 overflow-auto pr-1">
            {state.constructs.map((construct, index) => (
              <ConstructQueueItem
                key={construct.name}
                construct={construct}
                index={index}
                isSelected={construct.name === featuredConstruct?.name}
                onSelect={handleSelectConstruct}
              />
            ))}
          </ol>
        </section>

        <section className="space-y-4 p-4">
          <ConstructDetail construct={featuredConstruct} />
          {featuredConstruct && (
            <AttemptHistory
              entries={timeline}
              selectedKey={selectedEntry?.key ?? null}
              onSelect={handleSelectEntry}
            />
          )}
        </section>

        <section className="flex min-h-0 flex-col p-4">
          <ReachabilityPanel entry={selectedEntry} />
        </section>
      </div>
    </div>
  );
}

export default function StatisticalModelSpecRunningOutputView({
  workspaceId,
  showError = true,
}: {
  workspaceId: string;
  showError?: boolean;
}) {
  const state = useModelSpecAdmission(workspaceId);
  return <ModelSpecAdmissionRunningView state={state} showError={showError} />;
}
