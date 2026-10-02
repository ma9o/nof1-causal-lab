import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { StudyRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: StudyRevision }) {
  const workspaceId = context.model.context.workspace_id;
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
          {diff.data.graph.constructs
            .filter((item) => item.change.kind !== "unchanged")
            .map((item) => (
              <p key={item.construct_id}>
                {humanize(item.change.kind)} construct ·{" "}
                <OwnerLink
                  onClick={() => context.select({ kind: "construct", id: item.construct_id })}
                >
                  {humanize(
                    item.change.kind === "removed"
                      ? item.change.before.name
                      : item.change.after.name,
                  )}
                </OwnerLink>
              </p>
            ))}
          {diff.data.graph.edges
            .filter((item) => item.change.kind !== "unchanged")
            .map((item) => {
              const edge = item.change.kind === "removed" ? item.change.before : item.change.after;
              const name = (id: string) => {
                const construct = diff.data.graph.constructs.find((row) => row.construct_id === id);
                if (!construct) return null;
                const definition =
                  construct.change.kind === "removed"
                    ? construct.change.before
                    : construct.change.after;
                return humanize(definition.name);
              };
              return (
                <p key={item.edge_id}>
                  {humanize(item.change.kind)} edge ·{" "}
                  <OwnerLink onClick={() => context.select({ kind: "edge", id: item.edge_id })}>
                    {name(edge.cause.id)} → {name(edge.effect.id)}
                  </OwnerLink>
                </p>
              );
            })}
          {diff.data.parameters.map((item) => {
            const parameter =
              item.change.kind === "removed" ? item.change.before : item.change.after;
            const owner = parameterOwner(context.entities, item.parameter_id);
            return (
              <p key={item.parameter_id}>
                {humanize(item.change.kind)} law ·{" "}
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
