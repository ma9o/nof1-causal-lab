import { useState } from "react";
import { signColor } from "@/components/dag/core/palette";
import { formatClampValue } from "@/components/dag/intervention-dag-semantics";
import { EffectChart } from "../effect-chart";
import { formatPlain, formatSigned, humanize } from "../model-selection";
import type { ModelQuery } from "../queries";
import { ArtifactChip, Hint, KeyValue, Section, Tag } from "../scope-primitives";
import { type ScopeContext } from "./scope-context";

export function QueryScope({ context, query }: { context: ScopeContext; query: ModelQuery }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const simulation =
    query.evaluations.find((item) => item.evaluation.id === selectedId) ?? query.simulation;
  const posterior = simulation?.result.summary ?? query.posterior;
  const posteriorVersion = simulation?.evaluation.posterior.version ?? query.posteriorVersion;
  const modelId = simulation?.evaluation.model.id ?? context.model.model.id;
  const outcome = simulation?.result.outcome_label ?? query.outcome;
  const posteriorStatus = context.artifacts.find(
    (artifact) => artifact.artifact_id === "posterior",
  );
  const stale =
    modelId === context.model.model.id &&
    posteriorVersion === context.snapshot.versions.posterior &&
    posteriorStatus?.stale === true;
  const clamps = simulation?.query.clamps ?? query.savedQuery?.clamps ?? [];
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
                  ? clamps.map((item) => `${item.variable}: ${formatClampValue(item)}`).join("; ")
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
              : query.origin === "ranking"
                ? ([["mode", "+1 latent unit, held"]] as Array<[string, React.ReactNode]>)
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
        {query.prompt ? <Hint>asked: “{query.prompt}”</Hint> : null}
      </Section>
      {query.evaluations.length > 1 ? (
        <Section title="Evaluations" wide>
          <Hint>
            Compare the same question across fitted models. Select an evaluation to inspect its
            result.
          </Hint>
          <table className="w-full text-left text-[10.5px]">
            <thead className="text-muted-foreground">
              <tr>
                <th className="py-1">Model / posterior</th>
                <th>Mean effect</th>
                <th>95% interval</th>
              </tr>
            </thead>
            <tbody>
              {query.evaluations.map((item) => (
                <tr
                  key={item.evaluation.id}
                  className={item.evaluation.id === simulation?.evaluation.id ? "bg-muted/60" : ""}
                >
                  <td className="py-1.5">
                    <button
                      type="button"
                      className="text-left underline underline-offset-2"
                      aria-pressed={item.evaluation.id === simulation?.evaluation.id}
                      onClick={() => setSelectedId(item.evaluation.id)}
                    >
                      {item.evaluation.model.id} · posterior v{item.evaluation.posterior.version}
                    </button>
                  </td>
                  <td className="font-mono">{formatSigned(item.result.summary.mean, 3)}</td>
                  <td className="font-mono">
                    [{formatPlain(item.result.summary.lower_95)},{" "}
                    {formatPlain(item.result.summary.upper_95)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      ) : null}
      {posterior ? (
        <Section
          title="Result"
          chips={<ArtifactChip id="posterior" version={posteriorVersion} stale={stale} />}
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
            {modelId} · pinned to posterior v{posteriorVersion ?? "?"}
          </Hint>
          {simulation?.result.warnings.map((warning) => (
            <Hint key={warning} issue>
              {warning}
            </Hint>
          ))}
        </Section>
      ) : (
        <Section title="Result">
          <Hint>A kept query without a result of its own; asking it again produces one.</Hint>
        </Section>
      )}
      {simulation?.result.effect_trajectory && simulation.result.effect_trajectory.length > 1 ? (
        <Section title="Over the horizon" wide>
          {query.blurb ? <Hint>{query.blurb}</Hint> : null}
          <EffectChart simulation={simulation} />
        </Section>
      ) : null}
    </>
  );
}
