import { JsonViewer } from "@/components/ui/json-viewer";
import type { JournalTick } from "@/lib/model-asset/journal";
import { ARTIFACT_LABEL, actionLabel, humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, Section, StatusIcon } from "../scope-primitives";
import { ModelFindings } from "../model-findings";
import { SimulationEvidence } from "../simulation-evidence";
import { DataDetails } from "./data-details";
import { EditDetails } from "./edit-details";
import { FitDetails } from "./fit-details";

export function VersionScope({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  if (tick.status !== "applied")
    return (
      <Section title={`${actionLabel(tick.action)} failed`} wide>
        <div className="flex items-start gap-2">
          <StatusIcon status="failed" />
          <span>{tick.errorType ?? tick.status}</span>
        </div>
        <Hint>No scientific version was saved. The graph shows the unchanged parent version.</Hint>
        <pre className="whitespace-pre-wrap break-words font-mono text-[11px] text-destructive">
          {tick.error}
        </pre>
      </Section>
    );
  const messages = tick.messages.filter(
    (message) => message.level === "warn" || message.level === "error",
  );
  return (
    <>
      {tick.action === "edit_model" && <EditDetails context={context} tick={tick} />}
      {tick.action === "prepare_data" && <DataDetails context={context} />}
      {tick.action === "fit" && <FitDetails context={context} tick={tick} />}
      {tick.action === "simulate" && <SimulationEvidence model={context.model} />}
      {messages.length > 0 && (
        <Section title="Action messages">
          {messages.map((message, index) => (
            <div key={index} className="flex items-start gap-2">
              <StatusIcon status={message.level === "error" ? "failed" : "warning"} />
              <Hint issue>{humanize(message.label.toLowerCase())}</Hint>
            </div>
          ))}
        </Section>
      )}
      {tick.action !== "simulate" && <ModelFindings context={context} />}
      {tick.retracted.length > 0 && (
        <Section title="Removed">
          {tick.retracted.map((id) => (
            <Hint key={id}>{ARTIFACT_LABEL[id]}</Hint>
          ))}
        </Section>
      )}
      {Object.keys(tick.diagnostics).length > 0 && (
        <details className="w-[400px] min-w-0 flex-none text-xs">
          <summary className="cursor-pointer text-muted-foreground">Recorded diagnostics</summary>
          <JsonViewer data={tick.diagnostics} />
        </details>
      )}
    </>
  );
}
