import type { ReactNode } from "react";
import { humanize, type ModelSelection, SCOPE_KIND_LABEL } from "./model-selection";
import { QueryCollection } from "./query-collection";
import { AssetScope } from "./scopes/asset-scope";
import { ConstructScope } from "./scopes/construct-scope";
import { EdgeScope } from "./scopes/edge-scope";
import { IndicatorScope } from "./scopes/indicator-scope";
import { QueryScope } from "./scopes/query-scope";
import type { ScopeContext } from "./scopes/scope-context";
import { VersionScope } from "./scopes/version-scope";

function Crumb({ parts, onAsset }: { parts: ReactNode[]; onAsset: () => void }) {
  return (
    <div className="flex min-w-0 items-center gap-1.5 truncate text-[13px] text-muted-foreground">
      <button
        type="button"
        onClick={onAsset}
        className="cursor-pointer underline underline-offset-[3px]"
      >
        Asset
      </button>
      {parts.map((part, index) => (
        <span key={index} className="flex min-w-0 items-center gap-1.5">
          <span>›</span>
          <span
            className={
              index === parts.length - 1 ? "truncate font-semibold text-foreground" : "truncate"
            }
          >
            {part}
          </span>
        </span>
      ))}
    </div>
  );
}

function crumbParts(selection: ModelSelection, context: ScopeContext): ReactNode[] {
  switch (selection.kind) {
    case "asset":
      return [];
    case "version":
      return [`v${selection.seq}`];
    case "construct":
      return [humanize(context.entities.constructById.get(selection.id)?.name ?? selection.id)];
    case "edge": {
      const edge = context.entities.edgeById.get(selection.id);
      return [
        edge
          ? `${humanize(context.entities.constructById.get(edge.cause.id)!.name)} → ${humanize(context.entities.constructById.get(edge.effect.id)!.name)}`
          : selection.id,
      ];
    }
    case "indicator": {
      const indicator = context.entities.indicatorById.get(selection.id);
      return indicator
        ? [humanize(context.entities.indicatorOwnerById.get(indicator.id)!.name), indicator.name]
        : [selection.id];
    }
    case "query":
      return [
        "query",
        context.queries.find((query) => query.key === selection.key)?.title ?? selection.key,
      ];
  }
}

function ScopeBody({ selection, context }: { selection: ModelSelection; context: ScopeContext }) {
  switch (selection.kind) {
    case "asset":
      return <AssetScope context={context} />;
    case "version": {
      const tick = context.ticks.find((candidate) => candidate.seq === selection.seq);
      return tick ? <VersionScope context={context} tick={tick} /> : null;
    }
    case "construct":
      return <ConstructScope context={context} id={selection.id} />;
    case "edge":
      return <EdgeScope context={context} id={selection.id} />;
    case "indicator":
      return <IndicatorScope context={context} id={selection.id} />;
    case "query": {
      const query = context.queries.find((candidate) => candidate.key === selection.key);
      return query ? (
        <QueryScope
          key={`${context.model.context.workspace_id}:${query.key}:${JSON.stringify(query.request)}`}
          context={context}
          query={query}
        />
      ) : null;
    }
  }
}

/**
 * The bottom pane: one owner per view. Its sections flow into columns across the width;
 * the query collection stays at its left once the model has results.
 */
export function DetailsPane({
  selection,
  context,
  posteriorStale,
}: {
  selection: ModelSelection;
  context: ScopeContext;
  posteriorStale: boolean;
}) {
  const selectedQueryKey = selection.kind === "query" ? selection.key : null;
  return (
    <section className="flex h-[460px] flex-none flex-col gap-2 rounded-2xl border bg-card px-3 py-2.5">
      <div className="flex h-[22px] flex-none items-center gap-2.5 overflow-hidden">
        <Crumb
          parts={crumbParts(selection, context)}
          onAsset={() => context.select({ kind: "asset" })}
        />
        <span className="rounded border px-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          {SCOPE_KIND_LABEL[selection.kind]}
        </span>
      </div>
      <div className="flex min-h-0 flex-1 gap-3.5">
        {context.queries.length > 0 ? (
          <QueryCollection
            queries={context.queries}
            selectedKey={selectedQueryKey}
            outcome={context.outcome}
            modelVersion={context.snapshot.versions.model ?? null}
            posteriorStale={posteriorStale}
            onSelect={(key) => context.select({ kind: "query", key })}
          />
        ) : null}
        <div className="flex h-full min-w-0 flex-1 flex-col flex-wrap content-start gap-x-[18px] gap-y-2 overflow-x-auto">
          <ScopeBody selection={selection} context={context} />
        </div>
      </div>
    </section>
  );
}
