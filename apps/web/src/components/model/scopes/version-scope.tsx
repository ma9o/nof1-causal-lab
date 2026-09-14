import { JsonViewer } from "@/components/ui/json-viewer";
import { primaryArtifact } from "@/lib/model-asset/journal";
import { moveLabel } from "../model-selection";
import { Button } from "@/components/ui/button";
import type { JournalTick } from "@/lib/model-asset/journal";

import { ArtifactChip, Hint, KeyValue, Section, Tag } from "../scope-primitives";
import type { ScopeContext } from "./scope-context";

export function VersionScope({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const record = context.ticks.find((candidate) => candidate.seq === tick.seq);
  if (!record) return null;
  const provenance = record.move.kind === "write" ? record.move.provenance : "computed";
  const viewingHere = context.snapshot.playhead === record.seq;
  return (
    <>
      <Section title={`${record.move.kind} ${primaryArtifact(record.move)}`}>
        <KeyValue
          rows={[
            ["status", record.status],
            ["provenance", provenance],
            ["time", new Date(record.ts).toLocaleString()],
            ...(record.error
              ? ([["error", record.error]] as Array<[string, React.ReactNode]>)
              : []),
          ]}
        />
      </Section>
      {record.status === "applied" ? (
        <Section title="Installed by this move">
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {record.version != null && (
              <li className="flex flex-wrap items-center gap-1.5">
                <ArtifactChip id={primaryArtifact(record.move)} version={record.version} />
              </li>
            )}
            {record.derived.map((artifactId) => (
              <li key={artifactId} className="flex flex-wrap items-center gap-1.5">
                <ArtifactChip
                  id={artifactId}
                  version={context.current.versions[artifactId] ?? null}
                />
                <Tag>derived</Tag>
              </li>
            ))}
            {record.retracted.map((artifactId) => (
              <li key={`retracted-${artifactId}`} className="flex flex-wrap items-center gap-1.5">
                <ArtifactChip id={artifactId} version={null} retracted />
              </li>
            ))}
          </ul>
        </Section>
      ) : (
        <Section title="Outcome">
          <Hint issue>
            {moveLabel(record.move)} raised before writing anything; the asset is unchanged.
          </Hint>
        </Section>
      )}
      {record.diagnostics.workers != null && (
        <Section title="Extraction workers">
          <JsonViewer data={record.diagnostics.workers} />
        </Section>
      )}
      <Section title="Conversation">
        <Hint>
          {record.traceIds.length > 0
            ? `${record.traceIds.length} trace${record.traceIds.length === 1 ? "" : "s"} promoted by this move.`
            : "No LLM trace: this move was a computation or a write."}
        </Hint>
        <div className="mt-1 flex flex-wrap gap-1.5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => context.focusConversation(record.seq)}
          >
            Show in conversation
          </Button>
          {viewingHere ? (
            <Button type="button" variant="outline" size="sm" onClick={() => context.viewAt(null)}>
              Return to now
            </Button>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => context.viewAt(record.seq)}
            >
              View asset at v{record.seq}
            </Button>
          )}
        </div>
      </Section>
    </>
  );
}
