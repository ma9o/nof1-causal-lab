import { useModelDiff } from "@/lib/hooks/use-model-diff";
import {
  editBase,
  initialModelSummary,
  modelChangeSummary,
} from "@/lib/model-asset/action-presentation";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { DefinitionView } from "../definition-view";
import { Hint, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const base = editBase(tick);
  const diff = useModelDiff(
    context.model.context.workspace_id,
    base ?? "",
    base ? tick.commitId : null,
  );
  const names = new Map<string, string>(
    [
      ...context.entities.constructs,
      ...context.entities.indicators,
      ...context.entities.parameters,
    ].map((entity) => [entity.id, humanize(entity.name)]),
  );
  for (const edge of context.entities.edges) {
    names.set(edge.id, `${names.get(edge.cause.id)} → ${names.get(edge.effect.id)}`);
  }
  return (
    <Section title="Model changes" wide>
      {!base ? (
        <>
          <p className="font-medium">{initialModelSummary(context.model)}</p>
          <Hint>{context.model.model?.value.question}</Hint>
        </>
      ) : diff.error ? (
        <p role="alert" className="text-destructive">
          Unable to read model changes: {diff.error.message}
        </p>
      ) : !diff.data ? (
        <p role="status">Reading model changes…</p>
      ) : (
        <>
          <p className="font-medium">{modelChangeSummary(diff.data)}</p>
          {diff.data.definition_changes.map((change) => (
            <details key={change.path} className="border-t pt-2 text-xs">
              <summary className="cursor-pointer break-words">
                <span className="font-medium capitalize">{change.change}</span>
                {" · "}
                {change.path
                  .split("/")
                  .slice(1)
                  .map((part) => names.get(part) ?? humanize(part))
                  .join(" / ")}
              </summary>
              <div className="mt-2 space-y-2">
                {change.change !== "added" && (
                  <>
                    <Hint>Before</Hint>
                    <DefinitionView value={change.before} />
                  </>
                )}
                {change.change !== "removed" && (
                  <>
                    <Hint>After</Hint>
                    <DefinitionView value={change.after} />
                  </>
                )}
              </div>
            </details>
          ))}
        </>
      )}
    </Section>
  );
}
