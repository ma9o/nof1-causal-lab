"use client";

import type { ActionMessage, LLMTrace, RunningAction } from "@nof1-causal-lab/api-types";
import { LoaderCircle, X } from "lucide-react";
import { useMemo } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { ActionFindings } from "./action-findings";
import { Hint, Section } from "./scope-primitives";
import { DataDetails } from "./scopes/data-details";
import { EditDetails } from "./scopes/edit-details";
import { FitCalibration } from "./scopes/fit-calibration";
import { FitDetails } from "./scopes/fit-details";
import { SimulationEvidence } from "./simulation-evidence";

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
 * What the selected action did and what came of it, stated rather than shown: its verdicts, its
 * warnings with their reasons, and the results that matter. The state the action left belongs to
 * the graph and the details pane.
 * - edit_model: what changed, in model terms, and what its checks concluded (identification kept
 *   or lost, which predictive checks failed and why).
 * - prepare_data: which data arrived (source, window, variables, volume) and the problems found.
 * - fit: how it ran, in one line, and its verdict, such as parameter convergence failing for 7 of
 *   7 parameters with a worst R-hat of 2.3.
 * - simulate: the simulator's log and nothing more; everything the simulation produced is state.
 * An action the episode workflow is still executing has no timeline node yet, so it streams in
 * under the head it started from.
 */
export function ActionRecord({
  context,
  tick,
  running,
  useActionTrace,
}: {
  context: ScopeContext;
  tick: JournalTick | undefined;
  running: RunningAction | null;
  useActionTrace: UseActionTrace;
}) {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2.5">
        <h2 className="text-xs font-semibold">Action record</h2>
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
      <div key={tick?.seq} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-2">
        {tick ? (
          <>
            {tick.status !== "applied" ? (
              <Section
                title={`${humanize(tick.action).replace(/^./, (letter) => letter.toUpperCase())} failed`}
              >
                <Hint issue>{tick.error}</Hint>
              </Section>
            ) : (
              <>
                {tick.action === "edit_model" && <EditDetails context={context} tick={tick} />}
                {tick.action === "prepare_data" && <DataDetails context={context} tick={tick} />}
                {tick.action === "fit" && (
                  <>
                    <FitDetails context={context} />
                    <FitCalibration context={context} />
                  </>
                )}
                {tick.action === "simulate" && <SimulationEvidence context={context} />}
                <ActionFindings context={context} tick={tick} />
              </>
            )}
            {tick.traceIds.length > 0 && (
              <details className="px-3 text-xs">
                <summary className="cursor-pointer text-muted-foreground">
                  Agent conversation
                </summary>
                <ActionTrace tick={tick} useActionTrace={useActionTrace} />
              </details>
            )}
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
