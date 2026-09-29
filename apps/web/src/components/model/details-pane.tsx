import { DefinitionView } from "./definition-view";
import { DefinitionContext } from "./definition-context";
import { Button } from "@/components/ui/button";
import { entityOptions, resolveEntity } from "@/lib/model-asset/entities";
import { type ModelSelection, humanize } from "@/lib/model-asset/selection";
import { Hint, PosteriorTable, Section } from "./scope-primitives";
import { distributionText } from "@/lib/utils/distribution-format";
import { posteriorRows } from "@/lib/model-asset/inspector";
import { ConstructScope } from "./scopes/construct-scope";
import { EdgeScope } from "./scopes/edge-scope";
import { IndicatorScope } from "./scopes/indicator-scope";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { VersionScope } from "./scopes/version-scope";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";

function ScopeBody({ selection, context }: { selection: ModelSelection; context: ScopeContext }) {
  switch (selection.kind) {
    case "revision": {
      const tick = context.ticks.find((candidate) => candidate.seq === selection.seq);
      return tick ? <VersionScope context={context} tick={tick} /> : null;
    }
    case "construct":
      return <ConstructScope context={context} id={selection.id} />;
    case "edge":
      return <EdgeScope context={context} id={selection.id} />;
    case "indicator":
      return <IndicatorScope context={context} id={selection.id} />;
    case "parameter": {
      const model = context.model.model?.value;
      const parameter = context.entities.parameterById.get(selection.id);
      const fitted = posteriorRows(
        parameter ? [parameter] : [],
        context.model.findings.fit?.value.report,
      );
      return parameter ? (
        <>
          <Section title="Parameter">
            <Hint>{humanize(parameter.description)}</Hint>
            {parameter.value != null && <p className="font-mono">Fixed at {parameter.value}</p>}
          </Section>
          {parameter.distribution && (
            <Section title="Probability law">
              <p className="break-words font-mono">
                {distributionText(model!.distributions[parameter.distribution])}
              </p>
              {parameter.distribution_transform === "dt_persistence_to_ct_decay" && (
                <Hint>The law describes persistence; fitted values are decay rates.</Hint>
              )}
              {parameter.distribution_transform === "dt_effect_to_ct_rate" && (
                <Hint>The law describes interval effects; fitted values are rates.</Hint>
              )}
            </Section>
          )}
          {fitted.length > 0 && (
            <Section title="Fit" source={context.model.findings.fit?.source}>
              <PosteriorTable rows={fitted} />
            </Section>
          )}
        </>
      ) : (
        <Hint>This parameter is absent from this revision.</Hint>
      );
    }
  }
}

/** The inspector follows a persistent model entity, without creating query identities. */
export function DetailsPane({
  selection,
  context,
  loading,
  onClose,
}: {
  selection: ModelSelection;
  context: ScopeContext;
  loading: boolean;
  onClose: () => void;
}) {
  const entities = entityOptions(context.entities);
  const selectedKey = JSON.stringify(selection);
  const entity = resolveEntity(context.entities, selection);
  const missingEntity = selection.kind !== "revision" && !entity;
  const selectedTick =
    selection.kind === "revision"
      ? context.ticks.find((tick) => tick.seq === selection.seq)
      : undefined;
  return (
    <section
      aria-label="Model details"
      className="flex h-[460px] min-h-0 min-w-0 flex-none flex-col gap-3 overflow-hidden rounded-2xl border bg-card px-3 py-3 md:max-h-[52%]"
    >
      <div className="flex flex-none flex-wrap items-center gap-2 text-xs">
        <h2 className="font-semibold">Details</h2>
        <select
          aria-label="Details scope"
          className="min-w-0 flex-1 truncate rounded-md border bg-background px-2 py-1 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring"
          value={selectedKey}
          onChange={(event) => context.select(JSON.parse(event.target.value) as ModelSelection)}
        >
          {selectedTick && <option value={selectedKey}>{timelineTickLabel(selectedTick)}</option>}
          {missingEntity && (
            <option value={selectedKey}>
              Selected {selection.kind} · absent from this revision
            </option>
          )}
          {entities.map((entity) => (
            <option key={JSON.stringify(entity.selection)} value={JSON.stringify(entity.selection)}>
              {entity.label}
            </option>
          ))}
        </select>
        <Button
          type="button"
          size="icon-sm"
          variant="ghost"
          aria-label="Close details"
          onClick={onClose}
        >
          ×
        </Button>
      </div>
      {entity && <DefinitionContext entity={entity} onSelect={context.select} />}
      <div
        key={selectedKey}
        className="flex min-h-0 flex-1 flex-col flex-wrap content-start items-start gap-x-[18px] gap-y-3 overflow-x-auto pb-2"
      >
        {loading ? (
          <p role="status" className="text-xs text-muted-foreground">
            Loading version…
          </p>
        ) : missingEntity ? (
          <Hint>This {selection.kind} is absent from this revision.</Hint>
        ) : (
          <>
            <ScopeBody selection={selection} context={context} />
            {entity && (
              <details className="max-h-full w-[400px] min-w-0 flex-none overflow-auto border-t pt-2 text-xs">
                <summary className="cursor-pointer text-muted-foreground">Definition</summary>
                <div className="mt-2">
                  <DefinitionView value={entity.definition} />
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </section>
  );
}
