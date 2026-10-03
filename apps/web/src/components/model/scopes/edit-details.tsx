import { modelConstructs } from "@/lib/model-accessors";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { StudyRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: StudyRevision }) {
  const workspaceId = context.model.workspace_id;
  const base = tick.parent_ids.at(0);
  const previous = useModelSnapshot(workspaceId, base, tick.record.branch, base !== undefined);
  const ready = base === undefined || (previous.data && !previous.isPlaceholderData);
  const hasModel = ready && previous.data?.model != null;
  const diff = useModelDiff(workspaceId, base ?? "", hasModel ? tick.commit_id : null);
  const error = previous.error ?? diff.error;
  return (
    <Section title="Model changes" wide>
      {error ? (
        <p role="alert" className="text-destructive">
          Unable to read model changes: {error.message}
        </p>
      ) : !ready || (hasModel && !diff.data) ? (
        <p role="status">Reading model changes…</p>
      ) : !hasModel ? (
        <p className="font-medium">Model created</p>
      ) : diff.data ? (
        <>
          {diff.data.constructs
            .filter((change) => change.kind !== "unchanged")
            .map((change) => {
              const ref = change.kind === "removed" ? change.before : change.after;
              const model =
                change.kind === "removed" ? diff.data.beforeModel : diff.data.afterModel;
              const definition = modelConstructs(model).find((item) => item.id === ref.id);
              return (
                <p key={ref.id}>
                  {humanize(change.kind)} construct ·{" "}
                  <OwnerLink onClick={() => context.select({ kind: "construct", id: ref.id })}>
                    {humanize(definition?.name ?? ref.id)}
                  </OwnerLink>
                </p>
              );
            })}
          {diff.data.edges
            .filter((change) => change.kind !== "unchanged")
            .map((change) => {
              const ref = change.kind === "removed" ? change.before : change.after;
              const model =
                change.kind === "removed" ? diff.data.beforeModel : diff.data.afterModel;
              const edge = model.edges.find((item) => item.id === ref.id);
              const name = (id: string) =>
                humanize(modelConstructs(model).find((item) => item.id === id)?.name ?? id);
              return (
                <p key={ref.id}>
                  {humanize(change.kind)} edge ·{" "}
                  <OwnerLink onClick={() => context.select({ kind: "edge", id: ref.id })}>
                    {edge ? `${name(edge.cause.id)} → ${name(edge.effect.id)}` : ref.id}
                  </OwnerLink>
                </p>
              );
            })}
          {diff.data.parameters.map((item) => {
            const parameter = item.kind === "removed" ? item.before : item.after;
            const owner = parameterOwner(context.entities, parameter.id);
            return (
              <p key={parameter.id}>
                {humanize(item.kind)} law ·{" "}
                {owner ? (
                  <OwnerLink onClick={() => context.select(owner.selection)}>
                    {humanize(parameter.name)}
                  </OwnerLink>
                ) : (
                  humanize(parameter.name)
                )}
              </p>
            );
          })}
          {diff.data.changed_inputs.length > 0 && (
            <Hint>Updated {diff.data.changed_inputs.map(humanize).join(", ")}.</Hint>
          )}
        </>
      ) : null}
    </Section>
  );
}
