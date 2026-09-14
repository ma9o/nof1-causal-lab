import { useState } from "react";
import { createSimulateDispatch } from "@/components/dag/interactive/dispatch-simulate";
import { Button } from "@/components/ui/button";
import { signColor } from "@/components/dag/core/palette";
import { formatClampValue } from "@/components/dag/intervention-dag-semantics";
import { EffectChart } from "../effect-chart";
import { formatPlain, formatSigned, humanize } from "../model-selection";
import type { ModelQuery } from "../queries";
import { ArtifactChip, Hint, KeyValue, Section, Tag } from "../scope-primitives";
import { type ScopeContext } from "./scope-context";

export function QueryScope({ context, query }: { context: ScopeContext; query: ModelQuery }) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const simulation = query.simulation;
  const posterior = simulation?.summary ?? query.posterior;
  const modelVersion = simulation?.model.version ?? query.modelVersion;
  const modelId = simulation?.model.workspace_id ?? context.model.context.workspace_id;
  const outcome = simulation ? simulation.labels[simulation.request.outcome] : query.outcome;
  async function run() {
    setRunning(true);
    setError(null);
    try {
      const result = await createSimulateDispatch(context.model.context.workspace_id)(
        query.request,
      );
      context.setSimulation(query.key, result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Simulation failed");
    } finally {
      setRunning(false);
    }
  }
  const stale =
    modelId === context.model.context.workspace_id &&
    (modelVersion !== context.snapshot.versions.model ||
      context.model.findings.fit?.source.validity === "stale");
  const clamps = simulation?.request.clamps ?? query.request?.clamps ?? [];
  const clamp = clamps[0];
  return (
    <>
      <Section title="Query">
        <KeyValue
          rows={[
            [
              "intervene on",
              <span key="v" className="font-mono">
                {clamps.length
                  ? clamps
                      .map(
                        (item) =>
                          `${simulation?.labels[item.target] ?? context.entities.constructs.find((c) => c.id === item.target)?.name ?? item.target}: ${formatClampValue(item)}`,
                      )
                      .join("; ")
                  : query.title}
              </span>,
            ],
            ...(clamp
              ? ([
                  ["mode", formatClampValue(clamp)],
                  [
                    "window",
                    `d${clamp.from_day} to ${clamp.to_day != null ? `d${clamp.to_day}` : "horizon"}`,
                  ],
                ] as Array<[string, React.ReactNode]>)
              : []),
            [
              "start",
              query.startKind === "abducted" ? "observed history (abducted)" : "steady state",
            ],
            ...(query.horizonDays != null
              ? ([["horizon", `${query.horizonDays} d`]] as Array<[string, React.ReactNode]>)
              : []),
            ...(query.outcome
              ? ([["outcome", humanize(query.outcome)]] as Array<[string, React.ReactNode]>)
              : []),
          ]}
        />
      </Section>
      <Section title="Simulate">
        <Button size="sm" onClick={run} disabled={running || !context.canSimulate}>
          {running ? "Simulating…" : "Run on current fit"}
        </Button>
        <Hint>Runs the nonlinear drift across posterior draws. Results stay in this view.</Hint>
        {error ? <Hint issue>{error}</Hint> : null}
      </Section>
      {posterior ? (
        <Section
          title="Result"
          chips={<ArtifactChip id="model" version={modelVersion} stale={stale} />}
        >
          <div
            className={`flex flex-wrap items-baseline gap-1.5 font-mono text-[10.5px] ${stale ? "opacity-55" : ""}`}
          >
            <span className="min-w-[52px] font-sans text-[9.5px] font-semibold uppercase tracking-wide text-muted-foreground">
              posterior
            </span>
            <span
              className="text-[13px] font-semibold"
              style={{ color: signColor(posterior.mean) }}
            >
              {formatSigned(posterior.mean, 3)}
            </span>
            <span className="text-muted-foreground">
              [{formatPlain(posterior.lower_95)}, {formatPlain(posterior.upper_95)}]
            </span>
            <span className="text-[9.5px] text-muted-foreground">
              P&gt;0 {Math.round(posterior.prob_positive * 100)}%
            </span>
            {stale ? <Tag tone="warning">stale</Tag> : null}
          </div>
          <Hint>
            {posterior.mean > 0 ? "raises" : posterior.mean < 0 ? "lowers" : "leaves"}{" "}
            {outcome ? humanize(outcome) : "the outcome"}
            {query.horizonDays != null ? ` at day ${query.horizonDays}` : " at the steady state"} ·{" "}
            {modelId} · pinned to model v{modelVersion ?? "?"}
          </Hint>
          {simulation?.warnings.map((warning) => (
            <Hint key={warning} issue>
              {warning}
            </Hint>
          ))}
        </Section>
      ) : (
        <Section title="Result">
          <Hint>Run this request to see its result on the current fit.</Hint>
        </Section>
      )}
      {simulation?.effect_trajectory && simulation.effect_trajectory.length > 1 ? (
        <Section title="Over the horizon" wide>
          <EffectChart simulation={simulation} />
        </Section>
      ) : null}
    </>
  );
}
