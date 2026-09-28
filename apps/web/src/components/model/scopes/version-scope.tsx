import { JsonViewer } from "@/components/ui/json-viewer";
import type { JournalTick } from "@/lib/model-asset/journal";
import { ARTIFACT_LABEL, actionLabel } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, Section, StatusIcon } from "../scope-primitives";
import { ModelFindings } from "../model-findings";

export function VersionScope({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  return (
    <>
      {tick.status !== "applied" && (
        <Section title={actionLabel(tick.action)}>
          <div className="flex items-start gap-2">
            <StatusIcon status="failed" />
            <Hint issue>{tick.error}</Hint>
          </div>
          <Hint>The model is unchanged.</Hint>
        </Section>
      )}
      {tick.retracted.length > 0 && (
        <Section title="Removed">
          {tick.retracted.map((id) => (
            <Hint key={id}>{ARTIFACT_LABEL[id]}</Hint>
          ))}
        </Section>
      )}
      {Object.keys(tick.diagnostics).length > 0 && (
        <details className="w-[400px] min-w-0 flex-none text-xs">
          <summary className="cursor-pointer text-muted-foreground">Action findings</summary>
          <JsonViewer data={tick.diagnostics} />
        </details>
      )}
      <ModelFindings context={context} />
    </>
  );
}
