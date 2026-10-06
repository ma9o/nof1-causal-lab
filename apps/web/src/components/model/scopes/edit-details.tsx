import { modelConstructs } from "@/lib/model-accessors";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { ModelDiffOutput, TimelineRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: TimelineRevision }) {
  const workspaceId = context.model.workspace_id;
  const outcome = tick.record.attempt.outcome;
  const model =
    outcome.status === "applied"
      ? outcome.effects.produced.find((artifact) => artifact.artifact_id === "model")
      : undefined;
  const base = model?.derived_from.model ?? null;
  const hasModel = base !== null;
  const diff = useModelDiff(workspaceId, base, hasModel ? tick.commit_id : null, context.ticks);
  const error = diff.error;
  return (
    <Section title="Model changes" wide>
      {error ? (
        <p role="alert" className="text-destructive">
          Unable to read model changes: {error.message}
        </p>
      ) : hasModel && diff.isLoading ? (
        <p role="status">Reading model changes…</p>
      ) : !hasModel ? (
        <p className="font-medium">Model created</p>
      ) : diff.data ? (
        <ModelChanges context={context} report={diff.data} />
      ) : (
        <Hint>No saved comparison for these model versions.</Hint>
      )}
    </Section>
  );
}

function ModelChanges({ context, report }: { context: ScopeContext; report: ModelDiffOutput }) {
  return (
    <>
      {report.constructs
        .filter((change) => change.kind !== "unchanged")
        .map((change) => {
          const ref = change.kind === "removed" ? change.before : change.after;
          const model = change.kind === "removed" ? report.before_model : report.after_model;
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
      {report.edges
        .filter((change) => change.kind !== "unchanged")
        .map((change) => {
          const ref = change.kind === "removed" ? change.before : change.after;
          const model = change.kind === "removed" ? report.before_model : report.after_model;
          const edge = model?.edges.find((item) => item.id === ref.id);
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
      {report.parameters.map((item) => {
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
      {report.changed_inputs.length > 0 && (
        <Hint>Updated {report.changed_inputs.map(humanize).join(", ")}.</Hint>
      )}
    </>
  );
}

export function ModelComparisonDetails({ context }: { context: ScopeContext }) {
  const report = context.result?.action === "model_diff" ? context.result.body : null;
  return (
    <Section title="Model comparison" wide>
      {report ? (
        <ModelChanges context={context} report={report} />
      ) : (
        <Hint>Reading the saved comparison…</Hint>
      )}
    </Section>
  );
}
