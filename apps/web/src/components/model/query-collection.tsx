import { signColor } from "@/components/dag/core/palette";
import { cn } from "@/lib/utils";
import { formatSigned } from "./model-selection";
import type { ModelQuery } from "./queries";
import { ArtifactChip, Tag } from "./scope-primitives";

/** The model's query collection: available beside every scope after inference. */
export function QueryCollection({
  queries,
  selectedKey,
  outcome,
  modelVersion,
  posteriorStale,
  onSelect,
}: {
  queries: ModelQuery[];
  selectedKey: string | null;
  outcome: string | null;
  modelVersion: number | null;
  posteriorStale: boolean;
  onSelect: (key: string) => void;
}) {
  return (
    <aside className="flex w-[190px] flex-none flex-col gap-1.5 overflow-hidden border-r border-border pr-2.5">
      <div className="flex items-baseline gap-1.5 text-xs font-semibold">
        Queries{" "}
        <span className="text-[10px] font-normal text-muted-foreground">{queries.length}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        <ArtifactChip id="model" version={modelVersion} stale={posteriorStale} />
      </div>
      {posteriorStale ? (
        <div className="rounded-md border border-warning/40 bg-warning/14 px-1.5 py-1 text-[9.5px] leading-tight text-warning-foreground text-pretty">
          the selected posterior is stale · session results keep their original fit
        </div>
      ) : null}
      <div className="flex min-h-0 flex-col gap-1.5 overflow-y-auto">
        {queries.map((query) => (
          <button
            key={query.key}
            type="button"
            onClick={() => onSelect(query.key)}
            className={cn(
              "flex min-w-0 cursor-pointer flex-col gap-0.5 rounded-lg border bg-card px-1.5 py-1 text-left",
              selectedKey === query.key && "border-primary bg-primary/5 ring-2 ring-primary/30",
            )}
          >
            <span className="truncate font-mono text-[9.5px]" title={query.title}>
              {query.title}
            </span>
            <span className="truncate text-[9px] text-muted-foreground">
              {query.startKind === "abducted" ? "from observed history" : "steady state"}
              {query.horizonDays != null ? ` · ${query.horizonDays} d` : ""}
            </span>
            {query.posterior ? (
              <span className="flex items-baseline gap-1.5 font-mono text-[10px]">
                <span
                  className={cn("font-semibold", posteriorStale && "opacity-50")}
                  style={{ color: signColor(query.posterior.mean) }}
                >
                  {formatSigned(query.posterior.mean, 3)}
                </span>
                {posteriorStale ? <Tag tone="warning">stale</Tag> : null}
              </span>
            ) : null}
          </button>
        ))}
      </div>
      {outcome ? (
        <div className="mt-auto pt-1 text-[9.5px] text-muted-foreground text-pretty">
          outcome {outcome.replaceAll("_", " ")}
        </div>
      ) : null}
    </aside>
  );
}
