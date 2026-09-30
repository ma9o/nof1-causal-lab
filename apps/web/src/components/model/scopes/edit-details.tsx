import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { initialModelSummary, modelChangeSummary } from "@/lib/model-asset/action-presentation";
import { parameterOwner, resolveEntity } from "@/lib/model-asset/entities";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { DefinitionView } from "../definition-view";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const workspaceId = context.model.context.workspace_id;
  const base = tick.parentIds[0];
  const previous = useModelSnapshot(workspaceId, base, tick.branch, base !== undefined);
  const ready = base === undefined || (previous.data && !previous.isPlaceholderData);
  const hasModel = ready && previous.data?.model != null;
  const diff = useModelDiff(workspaceId, base ?? "", hasModel ? tick.commitId : null);
  const error = previous.error ?? diff.error;
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
      {error ? (
        <p role="alert" className="text-destructive">
          Unable to read model changes: {error.message}
        </p>
      ) : !ready || (hasModel && !diff.data) ? (
        <p role="status">Reading model changes…</p>
      ) : !hasModel ? (
        <>
          <p className="font-medium">{initialModelSummary(context.model)}</p>
          <Hint>{context.model.model?.value.question}</Hint>
        </>
      ) : diff.data ? (
        <>
          <p className="font-medium">{modelChangeSummary(diff.data)}</p>
          {diff.data.definition_changes.map((change) => {
            const path = change.path.split("/");
            const indicator = context.entities.indicators.find((item) => path.includes(item.id));
            const edge = context.entities.edges.find((item) => path.includes(item.id));
            const construct = context.entities.constructs.find((item) => path.includes(item.id));
            const parameter = context.entities.parameters.find(
              (item) =>
                path.includes(item.id) || (item.distribution && path.includes(item.distribution)),
            );
            const owner = parameter
              ? parameterOwner(context.entities, parameter.id)
              : indicator
                ? resolveEntity(context.entities, { kind: "indicator", id: indicator.id })
                : edge
                  ? resolveEntity(context.entities, { kind: "edge", id: edge.id })
                  : construct
                    ? resolveEntity(context.entities, { kind: "construct", id: construct.id })
                    : null;
            const label = path
              .slice(1)
              .map((part) => names.get(part) ?? humanize(part))
              .join(" / ");
            return (
              <details key={change.path} className="border-t pt-2 text-xs">
                <summary className="cursor-pointer break-words">
                  <span className="font-medium capitalize">{change.change}</span>
                  {" · "}
                  {owner ? (
                    <OwnerLink onClick={() => context.select(owner.selection)}>{label}</OwnerLink>
                  ) : (
                    label
                  )}
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
            );
          })}
        </>
      ) : null}
    </Section>
  );
}
