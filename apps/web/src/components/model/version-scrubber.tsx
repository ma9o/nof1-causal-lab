import { primaryArtifact } from "@/lib/model-asset/journal";
import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import { moveLabel } from "./model-selection";
import type { ArtifactId, PipelineSectionId } from "@nof1-causal-lab/api-types";
import type { TransitionTiming } from "@/lib/hooks/pipeline-progress";
import type { JournalTick } from "@/lib/model-asset/journal";
import { cn } from "@/lib/utils";
import { ARTIFACT_LABEL } from "./model-selection";

export interface ScrubberSegment {
  label: string;
  kind: "llm" | "tool" | "compute" | "io" | "wait" | "fail";
  seconds: number;
  note?: string;
}

const SEGMENT_COLOR: Record<ScrubberSegment["kind"], string> = {
  llm: "#16191d",
  tool: "#2f6bf0",
  compute: "#0f9b8e",
  io: "#9aa0a8",
  wait: "#cfd4db",
  fail: "var(--destructive)",
};

export function formatDuration(seconds: number): string {
  if (seconds >= 3600)
    return `${Math.floor(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
  if (seconds >= 60)
    return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  return `${Math.round(seconds)}s`;
}

/**
 * Where a move's time went, from what the journal and telemetry record: the wait since
 * the previous checkpoint and the run itself. Finer splits arrive with richer telemetry.
 */
export function segmentsForTick(
  tick: JournalTick,
  previous: JournalTick | undefined,
  timing: TransitionTiming | undefined,
): ScrubberSegment[] {
  if (tick.move.kind === "write") return [];
  const finished = Date.parse(tick.ts);
  const started = previous ? Date.parse(previous.ts) : null;
  const total = started != null && finished > started ? (finished - started) / 1000 : null;
  if (total == null) return [];
  const runKind = tick.status === "raised" ? "fail" : tick.traceIds.length > 0 ? "llm" : "compute";
  const runStart = timing?.startedAt;
  if (runStart != null && started != null && runStart > started && runStart < finished) {
    return [
      { label: "waiting", kind: "wait", seconds: (runStart - started) / 1000 },
      {
        label: tick.status === "raised" ? "raised" : "run",
        kind: runKind,
        seconds: (finished - runStart) / 1000,
      },
    ];
  }
  return [{ label: tick.status === "raised" ? "raised" : "run", kind: runKind, seconds: total }];
}

function Bars({ segments }: { segments: ScrubberSegment[] }) {
  const total = segments.reduce((sum, segment) => sum + segment.seconds, 0);
  return (
    <div className="flex h-full gap-px overflow-hidden rounded-sm">
      {segments.map((segment, index) => (
        <span
          key={`${segment.label}-${index}`}
          title={`${segment.label} · ${formatDuration(segment.seconds)}`}
          style={{
            width: `${(segment.seconds / total) * 100}%`,
            background: SEGMENT_COLOR[segment.kind],
          }}
          className="h-full min-w-px flex-none"
        />
      ))}
    </div>
  );
}

export function VersionScrubber({
  ticks,
  playhead,
  latest,
  timings,
  running,
  staleArtifacts,
  nextRun,
  selectedSeq,
  onSelectTick,
  onPlayhead,
}: {
  ticks: JournalTick[];
  playhead: number;
  latest: number;
  timings: Partial<Record<PipelineSectionId, TransitionTiming>>;
  running: PipelineSectionId[];
  staleArtifacts: ReadonlySet<ArtifactId>;
  nextRun: import("@nof1-causal-lab/api-types").OperationId | null;
  selectedSeq: number | null;
  onSelectTick: (seq: number) => void;
  onPlayhead: (seq: number) => void;
}) {
  const isNow = playhead >= latest && running.length === 0;
  const extra = running.length > 0 || (isNow && nextRun) ? 1 : 0;
  const n = ticks.length + extra;
  const headIndex =
    running.length > 0 ? ticks.length : ticks.findIndex((tick) => tick.seq === playhead);
  const headPct = ((headIndex + 0.5) / n) * 100;
  const selected = selectedSeq != null ? ticks.find((tick) => tick.seq === selectedSeq) : undefined;
  const selectedIndex = selected ? ticks.indexOf(selected) : -1;
  const selectedSegments = selected
    ? segmentsForTick(selected, ticks[selectedIndex - 1], timingFor(selected, timings))
    : [];
  const selectedTotal = selectedSegments.reduce((sum, segment) => sum + segment.seconds, 0);

  return (
    <section className="flex-none border-b bg-card px-6 pt-2.5 pb-1.5">
      <div className="flex items-baseline gap-2.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        Versions
        <span className="text-[11px] font-normal normal-case tracking-normal">
          one tick per applied move · the bars before a tick are where that move&apos;s time went ·
          click a tick for its version, double-click to view the asset as it was
        </span>
      </div>
      <div className="relative mt-6 h-16">
        <div className="absolute inset-x-0 top-[9px] h-0.5 bg-border" />
        <div
          className="absolute left-0 top-[9px] h-0.5 bg-foreground"
          style={{ width: `${headPct}%` }}
        />
        {ticks.map((tick, index) => {
          const segments = segmentsForTick(tick, ticks[index - 1], timingFor(tick, timings));
          if (segments.length === 0) return null;
          const width = 0.82 / n;
          let left = (index + 0.5) / n - 0.02 / n - width;
          let span = width;
          if (left < 0.01 / n) {
            span = width + left - 0.01 / n;
            left = 0.01 / n;
          }
          const total = segments.reduce((sum, segment) => sum + segment.seconds, 0);
          return (
            <div
              key={`span-${tick.seq}`}
              className={cn(
                "absolute top-1.5 h-2",
                tick.seq > playhead && "opacity-30",
                selectedSeq === tick.seq && "outline-2 outline-offset-2 outline-foreground",
              )}
              style={{ left: `${left * 100}%`, width: `${span * 100}%` }}
            >
              <Bars segments={segments} />
              <span className="absolute -top-[13px] left-0 whitespace-nowrap font-mono text-[8px] text-muted-foreground">
                {formatDuration(total)}
              </span>
            </div>
          );
        })}
        {ticks.map((tick, index) => {
          const past = tick.seq <= playhead;
          const head = tick.seq === playhead && running.length === 0;
          const raised = tick.status === "raised";
          const stale =
            tick.status === "applied" &&
            isNow &&
            [primaryArtifact(tick.move), ...tick.derived].some((artifactId) =>
              staleArtifacts.has(artifactId),
            );
          const sub =
            tick.retracted.length > 0
              ? `${tick.retracted.length} derived retracted`
              : tick.move.kind === "write"
                ? "edited by hand"
                : tick.derived.length > 0
                  ? `+${tick.derived.length} derived`
                  : "";
          return (
            <div
              key={tick.seq}
              className={cn(
                "absolute top-0 w-[118px] -translate-x-1/2 text-center",
                !past && "opacity-30",
              )}
              style={{ left: `${((index + 0.5) / n) * 100}%` }}
            >
              <button
                type="button"
                onClick={() => onSelectTick(tick.seq)}
                onDoubleClick={() => onPlayhead(tick.seq)}
                aria-label={`${moveLabel(tick.move)} v${tick.version ?? "·"} at move ${tick.seq}`}
                className={cn(
                  "mt-1 inline-block h-3 w-3 cursor-pointer rounded-full bg-foreground shadow-[0_0_0_3px_var(--card)]",
                  tick.move.kind === "write" && "rotate-45 scale-[.85] rounded-sm",
                  raised &&
                    "bg-transparent text-[13px] font-bold leading-3 text-destructive shadow-none",
                  head && "outline-2 outline-offset-2 outline-foreground",
                  selectedSeq === tick.seq && "ring-2 ring-primary/40",
                )}
              >
                {raised ? "✕" : ""}
              </button>
              <div
                className={cn(
                  "mt-0.5 truncate font-mono text-[9px]",
                  raised && "text-destructive",
                  stale && "text-warning-foreground",
                )}
              >
                {moveLabel(tick.move)}
                {raised ? " · failed" : ""}{" "}
                {tick.version != null ? (
                  <span className="text-muted-foreground">
                    v{tick.version}
                    {stale ? " · stale" : ""}
                  </span>
                ) : null}
              </div>
              {sub ? <div className="truncate text-[9px] text-muted-foreground">{sub}</div> : null}
              {head ? (
                <div className="absolute left-1/2 -top-[20px] -translate-x-1/2 whitespace-nowrap rounded-md bg-foreground px-1.5 text-[9px] font-semibold text-primary-foreground">
                  v{playhead} · {isNow ? "now" : "viewing"}
                </div>
              ) : null}
            </div>
          );
        })}
        {running.length > 0 ? (
          <div
            className="absolute top-0 w-[118px] -translate-x-1/2 text-center"
            style={{ left: `${((ticks.length + 0.5) / n) * 100}%` }}
          >
            <span className="mt-1 inline-block h-2.5 w-2.5 animate-pulse rounded-full border-2 border-[#2f6bf0] bg-card shadow-[0_0_0_3px_var(--card)]" />
            <div className="mt-0.5 truncate font-mono text-[9px]">
              {running.map((id) => TRANSITION_META[id].label).join(", ")}
            </div>
            <div className="truncate text-[9px] text-muted-foreground">materializing</div>
            <div className="absolute left-1/2 -top-[20px] -translate-x-1/2 whitespace-nowrap rounded-md bg-foreground px-1.5 text-[9px] font-semibold text-primary-foreground">
              now
            </div>
          </div>
        ) : isNow && nextRun ? (
          <div
            className="absolute top-0 w-[118px] -translate-x-1/2 text-center"
            style={{ left: `${((ticks.length + 0.5) / n) * 100}%` }}
          >
            <span className="mt-1 inline-block h-[11px] w-[11px] rounded-full border-[1.5px] border-dashed border-[#2f6bf0] bg-card text-center text-[7px] leading-[11px] text-[#2f6bf0] shadow-[0_0_0_3px_var(--card)]">
              ▶
            </span>
            <div className="mt-0.5 truncate font-mono text-[9px] font-semibold text-[#2f6bf0]">
              run {TRANSITION_META[nextRun].label}
            </div>
            <div className="truncate text-[9px] text-[#2f6bf0]">
              {staleArtifacts.size > 0 ? "recompute · inputs changed" : "next legal move"}
            </div>
          </div>
        ) : null}
      </div>
      {selected ? (
        <div className="mt-1 flex flex-col gap-1.5 border-t border-dashed pt-2">
          <div className="flex items-baseline gap-2.5 text-[11px]">
            <b className="font-semibold">
              v{selected.seq} · {ARTIFACT_LABEL[primaryArtifact(selected.move)]}
              {selected.version != null ? ` v${selected.version}` : ""}
            </b>
            <span className="text-[10.5px] text-muted-foreground">
              {selectedSegments.length > 0
                ? `${formatDuration(selectedTotal)} since the previous checkpoint · `
                : ""}
              finished {new Date(selected.ts).toLocaleTimeString()}
            </span>
          </div>
          {selectedSegments.length > 0 ? (
            <>
              <div className="flex h-[26px] gap-0.5">
                {selectedSegments.map((segment, index) => (
                  <div
                    key={`${segment.label}-${index}`}
                    title={`${segment.label} · ${formatDuration(segment.seconds)}`}
                    style={{
                      flex: `${segment.seconds} 0 0`,
                      background: SEGMENT_COLOR[segment.kind],
                      color: segment.kind === "wait" ? "#3a3f47" : "#fff",
                    }}
                    className="flex h-full min-w-0.5 items-center gap-1.5 overflow-hidden rounded px-1.5 text-[9.5px] whitespace-nowrap"
                  >
                    <span className="truncate">{segment.label}</span>
                    <span className="font-mono text-[8.5px] opacity-85">
                      {formatDuration(segment.seconds)}
                    </span>
                  </div>
                ))}
              </div>
              <div className="flex justify-between font-mono text-[8.5px] text-muted-foreground">
                {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
                  <span key={fraction}>{formatDuration(selectedTotal * fraction)}</span>
                ))}
              </div>
            </>
          ) : (
            <div className="text-[10.5px] text-muted-foreground">
              {selected.move.kind === "write"
                ? "A write lands at once; no span to break down."
                : "No previous checkpoint to measure this move against."}
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}

function timingFor(
  tick: JournalTick,
  timings: Partial<Record<PipelineSectionId, TransitionTiming>>,
): TransitionTiming | undefined {
  return (timings as Partial<Record<string, TransitionTiming>>)[primaryArtifact(tick.move)];
}
