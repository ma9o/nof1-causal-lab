"use client";

import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { useEffect, useMemo, useRef } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { JournalTick } from "@/lib/model-asset/journal";
import { cn } from "@/lib/utils";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { ARTIFACT_LABEL } from "./model-selection";

export type MoveTraceState =
  | { status: "loading" }
  | { status: "ready"; trace: LLMTrace }
  | { status: "absent" };

/** A hook the pane calls per open turn; the app backs it with the trace endpoint, stories with fixtures. */
export type UseMoveTrace = (seq: number, enabled: boolean) => MoveTraceState;

function TurnBody({
  tick,
  question,
  useMoveTrace,
}: {
  tick: JournalTick;
  question: string | undefined;
  useMoveTrace: UseMoveTrace;
}) {
  const traceState = useMoveTrace(tick.seq, tick.traceIds.length > 0);
  const messages = useMemo(
    () => (traceState.status === "ready" ? traceToUIMessages(traceState.trace) : []),
    [traceState],
  );
  if (tick.status === "raised") {
    return (
      <div className="flex items-baseline gap-1.5 pl-4 text-[10.5px] text-destructive">
        <span className="font-bold">✕</span>
        <span>
          {tick.error ?? "raised"} · the run raised before writing anything · the asset is unchanged
        </span>
      </div>
    );
  }
  if (tick.move.kind === "write") {
    const byHand = tick.move.provenance === "human";
    return (
      <div className="flex flex-col gap-1 pl-4 text-[11.5px]">
        <div className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
          {byHand ? "you" : tick.move.provenance}
        </div>
        <div className="rounded-xl border bg-secondary px-2.5 py-1.5 text-pretty">
          {tick.move.artifact_id === "question" && question
            ? question
            : `${ARTIFACT_LABEL[tick.move.artifact_id]} written${byHand ? " by hand" : ""}`}
        </div>
        {tick.derived.length > 0 || tick.retracted.length > 0 ? (
          <MachineLine>
            {tick.derived.length > 0 ? `re-derived ${tick.derived.join(", ")}` : null}
            {tick.derived.length > 0 && tick.retracted.length > 0 ? " · " : null}
            {tick.retracted.length > 0 ? `retracted ${tick.retracted.join(", ")}` : null}
          </MachineLine>
        ) : null}
      </div>
    );
  }
  if (tick.traceIds.length === 0) {
    return (
      <div className="pl-4">
        <MachineLine>
          computed · installed {[tick.move.artifact_id, ...tick.derived].join(", ")}
        </MachineLine>
      </div>
    );
  }
  if (traceState.status === "loading") {
    return <div className="pl-4 text-[10.5px] text-muted-foreground">loading the trace…</div>;
  }
  if (traceState.status === "absent") {
    return (
      <div className="pl-4">
        <MachineLine>no trace was promoted for this move</MachineLine>
      </div>
    );
  }
  return (
    <div className="pl-4 [&_.prose]:text-[11.5px] [&_pre]:text-[10px]">
      <ChatMessages messages={messages} />
      {tick.derived.length > 0 ? (
        <div className="mt-1.5">
          <MachineLine>derived {tick.derived.join(", ")}</MachineLine>
        </div>
      ) : null}
    </div>
  );
}

function MachineLine({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-1.5 text-[10.5px] text-muted-foreground text-pretty">
      <span className="relative -top-px inline-block h-[7px] w-[7px] flex-none rounded-full bg-muted-foreground" />
      <span>{children}</span>
    </div>
  );
}

function traceMeta(tick: JournalTick, traceState: MoveTraceState | null): string {
  if (tick.status === "raised") return `raised · ${tick.error ?? "error"}`;
  if (tick.move.kind === "write")
    return tick.move.provenance === "human" ? "by hand" : tick.move.provenance;
  if (tick.traceIds.length === 0) return "computed";
  if (traceState?.status === "ready") {
    const { trace } = traceState;
    return `${trace.messages.length} messages · ${formatCompact(trace.usage.input_tokens)} in / ${formatCompact(trace.usage.output_tokens)} out`;
  }
  return `${tick.traceIds.length} trace${tick.traceIds.length === 1 ? "" : "s"}`;
}

function Turn({
  tick,
  open,
  head,
  isNow,
  future,
  focused,
  question,
  useMoveTrace,
  onSelect,
}: {
  tick: JournalTick;
  open: boolean;
  head: boolean;
  isNow: boolean;
  future: boolean;
  focused: boolean;
  question: string | undefined;
  useMoveTrace: UseMoveTrace;
  onSelect: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [focused]);
  const raised = tick.status === "raised";
  return (
    <div
      ref={ref}
      className={cn(
        "flex flex-col gap-1.5 border-b py-1.5 last:border-b-0",
        future && "opacity-30",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className="flex min-w-0 cursor-pointer items-center gap-1.5 overflow-hidden whitespace-nowrap text-left text-[11px] text-muted-foreground"
      >
        <span
          className={cn(
            "inline-block h-[9px] w-[9px] flex-none rounded-full bg-foreground",
            tick.move.kind === "write" && "rotate-45 rounded-[1px]",
            raised && "bg-transparent text-[11px] font-bold leading-[9px] text-destructive",
          )}
        >
          {raised ? "✕" : ""}
        </span>
        <b
          className={cn(
            "font-semibold text-foreground",
            focused && "underline underline-offset-[3px]",
          )}
        >
          {ARTIFACT_LABEL[tick.move.artifact_id]}
        </b>
        {tick.version != null ? (
          <span className="font-mono text-[10px]">v{tick.version}</span>
        ) : null}
        <span className="min-w-0 truncate">{traceMeta(tick, null)}</span>
        {head ? (
          <span className="ml-auto flex-none rounded-md bg-foreground px-1.5 text-[9px] font-semibold text-primary-foreground">
            {isNow ? "now" : "viewing"}
          </span>
        ) : null}
      </button>
      {open ? <TurnBody tick={tick} question={question} useMoveTrace={useMoveTrace} /> : null}
    </div>
  );
}

/** The journal read as a thread: every move is a turn, folded unless it is where you are. */
export function ConversationPane({
  ticks,
  playhead,
  latest,
  running,
  question,
  focusSeq,
  useMoveTrace,
  onSelectTick,
}: {
  ticks: JournalTick[];
  playhead: number;
  latest: number;
  running: string[];
  question: string | undefined;
  focusSeq: number | null;
  useMoveTrace: UseMoveTrace;
  onSelectTick: (seq: number) => void;
}) {
  const isNow = playhead >= latest && running.length === 0;
  const headIndex = ticks.findIndex((tick) => tick.seq === playhead);
  const open = new Set<number>(
    focusSeq == null
      ? ticks.slice(Math.max(0, headIndex - 1), headIndex + 1).map((tick) => tick.seq)
      : [focusSeq, playhead],
  );
  return (
    <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border bg-card">
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2.5">
        <span className="text-xs font-semibold">Conversation</span>
        <span className="text-[11px] text-muted-foreground">
          {ticks.length} moves · one thread per workspace
        </span>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-1.5">
        {ticks.map((tick) => (
          <Turn
            key={tick.seq}
            tick={tick}
            open={open.has(tick.seq) && tick.seq <= playhead}
            head={tick.seq === playhead && running.length === 0}
            isNow={isNow}
            future={tick.seq > playhead}
            focused={focusSeq === tick.seq}
            question={question}
            useMoveTrace={useMoveTrace}
            onSelect={() => onSelectTick(tick.seq)}
          />
        ))}
        {running.length > 0 ? (
          <div className="flex flex-col gap-1.5 py-1.5">
            <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span className="inline-block h-2 w-2 flex-none animate-pulse rounded-full border-2 border-[#2f6bf0] bg-card" />
              <b className="font-semibold text-foreground">
                {running
                  .map((id) => ARTIFACT_LABEL[id as keyof typeof ARTIFACT_LABEL] ?? id)
                  .join(", ")}
              </b>
              <span>running</span>
              <span className="ml-auto rounded-md bg-foreground px-1.5 text-[9px] font-semibold text-primary-foreground">
                now
              </span>
            </div>
            <div className="pl-4 text-[10.5px] text-[#2f6bf0]">
              the new layer will appear here, the rest of the asset stays put
            </div>
          </div>
        ) : null}
      </div>
      <div className="flex flex-none flex-col gap-1.5 border-t px-3 py-2.5">
        <div className="flex min-h-[38px] items-center gap-2 rounded-xl border border-dashed bg-muted px-2.5 py-2">
          <span className="flex-1 text-xs text-muted-foreground">
            {isNow
              ? "Ask about the model, or say what to change…"
              : `viewing v${playhead} of v${latest}`}
          </span>
        </div>
        <p className="m-0 text-[10px] text-muted-foreground text-pretty">
          {isNow
            ? "Asking is not wired to the machine yet: moves come from the harness or the facade for now."
            : "Return to now to continue the conversation."}
        </p>
      </div>
    </aside>
  );
}
