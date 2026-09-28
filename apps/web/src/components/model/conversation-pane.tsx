"use client";

import type { ActionMessage, ActionMessageEvent, LLMTrace } from "@nof1-causal-lab/api-types";
import { LoaderCircle, X } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { JournalTick } from "@/lib/model-asset/journal";
import { revisionBranch, revisionTimeline } from "@/lib/model-asset/revision-timeline";
import { ARTIFACT_LABEL, actionLabel } from "@/lib/model-asset/selection";
import { cn } from "@/lib/utils";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";

export type ActionTraceState =
  | { status: "loading" }
  | { status: "ready"; trace: LLMTrace }
  | { status: "absent" };
export type UseActionTrace = (seq: number, enabled: boolean) => ActionTraceState;

function ActionLabels({ messages }: { messages: ActionMessage[] }) {
  return (
    <ul className="space-y-1 px-4 py-1 text-[10px] font-mono" aria-label="Action messages">
      {messages.map((message, index) => (
        <li
          key={`${message.timestamp}:${index}`}
          className={cn(
            "flex flex-wrap gap-x-2 text-muted-foreground",
            message.level === "warn" && "text-amber-700",
            message.level === "error" && "text-destructive",
          )}
        >
          <time dateTime={message.timestamp} title={message.timestamp}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </time>
          <span>{message.level}</span>
          <span className="break-all">{message.label}</span>
        </li>
      ))}
    </ul>
  );
}

function TurnBody({ tick, useActionTrace }: { tick: JournalTick; useActionTrace: UseActionTrace }) {
  const traceState = useActionTrace(tick.seq, tick.traceIds.length > 0);
  const messages = useMemo(
    () => (traceState.status === "ready" ? traceToUIMessages(traceState.trace) : []),
    [traceState],
  );
  if (tick.status !== "applied" || tick.traceIds.length === 0) return null;
  if (traceState.status === "loading")
    return (
      <p role="status" className="pl-4 text-xs text-muted-foreground">
        Loading conversation…
      </p>
    );
  if (traceState.status === "absent")
    return <p className="pl-4 text-xs text-muted-foreground">Conversation unavailable.</p>;
  return (
    <div className="pl-4 [&_.prose]:text-[11.5px] [&_pre]:text-[10px]">
      <ChatMessages messages={messages} />
    </div>
  );
}

function Turn({
  tick,
  open,
  future,
  focused,
  useActionTrace,
  onSelect,
}: {
  tick: JournalTick;
  open: boolean;
  future: boolean;
  focused: boolean;
  useActionTrace: UseActionTrace;
  onSelect: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focused]);
  const failed = tick.status !== "applied";
  return (
    <div
      ref={ref}
      className={cn(
        "flex flex-col gap-1.5 border-b px-2 py-2.5 last:border-b-0",
        focused && "rounded-lg bg-muted ring-1 ring-inset ring-border",
        future && !focused && "opacity-40",
      )}
    >
      <button
        type="button"
        aria-label={
          failed
            ? `Inspect failed attempt ${tick.seq}: ${actionLabel(tick.action)}`
            : `View version ${tick.seq}: ${actionLabel(tick.action)}`
        }
        aria-current={focused ? "step" : undefined}
        onClick={onSelect}
        className="flex min-w-0 cursor-pointer items-center gap-2 text-left text-[11px] text-muted-foreground"
      >
        {failed ? (
          <X className="size-3.5 flex-none text-destructive" aria-hidden="true" />
        ) : (
          <span
            aria-hidden="true"
            className={cn(
              "inline-block size-2 flex-none rounded-full bg-foreground",
              tick.action === "edit_model" && "rotate-45 rounded-[1px]",
            )}
          />
        )}
        <span className="min-w-0 flex-1 font-medium text-foreground">
          {actionLabel(tick.action)}
        </span>
        <span className="flex-none font-mono text-[10px]">
          {failed ? `attempt ${tick.seq}` : `v${tick.seq}`}
        </span>
      </button>
      {open && (
        <>
          <ActionLabels messages={tick.messages} />
          <TurnBody tick={tick} useActionTrace={useActionTrace} />
        </>
      )}
    </div>
  );
}

/** The log owns recorded conversation; results are inspected through the selected action. */
export function ConversationPane({
  ticks,
  branches,
  branch,
  playhead,
  latest,
  running,
  focusSeq,
  useActionTrace,
  onSelectTick,
  actionMessages,
}: {
  ticks: JournalTick[];
  branches: Record<string, string>;
  branch: string;
  playhead: number;
  latest: number;
  running: string[];
  focusSeq: number;
  useActionTrace: UseActionTrace;
  onSelectTick: (seq: number) => void;
  actionMessages: ActionMessageEvent[];
}) {
  const turns = useMemo(() => {
    const lineage = revisionBranch(revisionTimeline(ticks, branches), branches[branch]);
    const commits = new Set(lineage.nodes.map((node) => node.tick.commitId));
    return ticks.filter((tick) =>
      tick.status === "applied"
        ? commits.has(tick.commitId)
        : tick.parentIds.some((id) => commits.has(id)),
    );
  }, [ticks, branches, branch]);
  const open = new Set([focusSeq, playhead]);
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex-none border-b px-3 py-2.5">
        <h2 className="text-xs font-semibold">Action log</h2>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-1 py-1.5">
        {turns.map((tick) => (
          <Turn
            key={tick.seq}
            tick={tick}
            open={open.has(tick.seq)}
            future={playhead < latest && tick.seq > playhead}
            focused={focusSeq === tick.seq}
            useActionTrace={useActionTrace}
            onSelect={() => onSelectTick(tick.seq)}
          />
        ))}
        {actionMessages.length > 0 && (
          <div role="log" aria-live="polite">
            <ActionLabels messages={actionMessages.map((event) => event.message)} />
          </div>
        )}
        {running.length > 0 && (
          <div role="status" className="flex items-center gap-2 px-2 py-3 text-xs">
            <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
            <span>
              {running
                .map((id) => ARTIFACT_LABEL[id as keyof typeof ARTIFACT_LABEL] ?? id)
                .join(", ")}
            </span>
            <span className="sr-only">running</span>
          </div>
        )}
      </div>
    </div>
  );
}
