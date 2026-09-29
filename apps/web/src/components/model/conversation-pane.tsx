"use client";

import type { ActionMessage, LLMTrace, RunningAction } from "@nof1-causal-lab/api-types";
import { LoaderCircle, X } from "lucide-react";
import { useMemo } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { JournalTick } from "@/lib/model-asset/journal";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
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

/** The model calls recorded inside the action, failed attempts included. */
function ActionTrace({
  tick,
  useActionTrace,
}: {
  tick: JournalTick;
  useActionTrace: UseActionTrace;
}) {
  const traceState = useActionTrace(tick.seq, tick.traceIds.length > 0);
  const messages = useMemo(
    () => (traceState.status === "ready" ? traceToUIMessages(traceState.trace) : []),
    [traceState],
  );
  if (tick.traceIds.length === 0) return null;
  if (traceState.status === "loading")
    return (
      <p role="status" className="px-4 text-xs text-muted-foreground">
        Loading conversation…
      </p>
    );
  if (traceState.status === "absent")
    return <p className="px-4 text-xs text-muted-foreground">Conversation unavailable.</p>;
  return (
    <div className="px-4 [&_.prose]:text-[11.5px] [&_pre]:text-[10px]">
      <ChatMessages messages={messages} />
    </div>
  );
}

/**
 * The selected action's own log: its messages and recorded model calls. An action the episode
 * workflow is executing has no timeline node yet, so it streams in under the head it started from.
 */
export function ConversationPane({
  tick,
  running,
  useActionTrace,
}: {
  tick: JournalTick | undefined;
  running: RunningAction | null;
  useActionTrace: UseActionTrace;
}) {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2.5">
        <h2 className="text-xs font-semibold">Action log</h2>
        {tick && (
          <span className="ml-auto flex min-w-0 items-center gap-1 font-mono text-[10px] text-muted-foreground">
            {tick.status !== "applied" && (
              <X className="size-3 flex-none text-destructive" aria-hidden="true" />
            )}
            <span className="truncate">{timelineTickLabel(tick)}</span>
            {tick.status !== "applied" && <span className="sr-only"> · failed</span>}
          </span>
        )}
      </div>
      <div key={tick?.seq} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto py-2">
        {tick ? (
          <>
            {tick.messages.length > 0 ? (
              <ActionLabels messages={tick.messages} />
            ) : (
              tick.traceIds.length === 0 && (
                <p className="px-4 text-xs text-muted-foreground">No log was recorded.</p>
              )
            )}
            <ActionTrace tick={tick} useActionTrace={useActionTrace} />
          </>
        ) : (
          !running && <p className="px-4 text-xs text-muted-foreground">No actions yet.</p>
        )}
        {running && (
          <section
            role="log"
            aria-live="polite"
            aria-label={`${running.action} · running`}
            className="border-t pt-2 first:border-t-0 first:pt-0"
          >
            <h3 className="flex items-center gap-2 px-4 pb-1 text-[11px] font-medium">
              <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
              {running.action} · running
            </h3>
            <ActionLabels messages={running.messages} />
          </section>
        )}
      </div>
    </div>
  );
}
