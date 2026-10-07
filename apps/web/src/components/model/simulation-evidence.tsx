import type { SimulationReport } from "@nof1-causal-lab/api-types";
import { useState } from "react";
import type { ArmChoice } from "@/components/charts/series-adapters";
import { resolveEntity, type ModelEntities } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatPlain, humanize } from "@/lib/model-asset/selection";
import { hasCausalEffects } from "@/lib/simulation-report";
import { ALL_DRAWS, type DrawSelection, useSimulationPaths } from "@/lib/hooks/use-visuals";
import { formatModelDate } from "@/lib/utils/format";
import { SimulationHistory } from "./scopes/recorded-history";
import { Hint, KeyValue, Section, StatusIcon } from "./scope-primitives";

type DrawMode = "all" | 24 | 1;

interface SimulationView {
  readonly mode: DrawMode;
  readonly start: number;
  readonly arms: ArmChoice;
}

const selectionOf = (view: SimulationView): DrawSelection =>
  view.mode === "all" ? ALL_DRAWS : { start: view.start, count: view.mode };

function Choice({
  pressed,
  onClick,
  children,
}: {
  pressed: boolean;
  onClick: () => void;
  children: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className="rounded border px-2 py-1 text-[10.5px] aria-pressed:border-foreground aria-pressed:bg-foreground aria-pressed:text-background"
    >
      {children}
    </button>
  );
}

/** One control for every chart of the simulation: which draws, and which arms. */
function SimulationControls({
  view,
  onChange,
  total,
  paired,
}: {
  view: SimulationView;
  onChange: (view: SimulationView) => void;
  total: number;
  paired: boolean;
}) {
  const page = view.mode === "all" ? total : view.mode;
  return (
    <div
      className="flex flex-wrap items-center gap-x-4 gap-y-2"
      role="group"
      aria-label="Shown draws"
    >
      <div className="flex flex-wrap items-center gap-1">
        <span className="mr-1 text-[10px] uppercase tracking-wide text-muted-foreground">
          Draws
        </span>
        {(["all", 24, 1] as const).map((mode) => (
          <Choice
            key={mode}
            pressed={view.mode === mode}
            onClick={() => onChange({ ...view, mode, start: 0 })}
          >
            {mode === "all" ? `All ${total}` : mode === 1 ? "One" : `${mode}`}
          </Choice>
        ))}
        {view.mode !== "all" && (
          <span className="flex items-center gap-1 text-[10.5px]">
            <button
              type="button"
              aria-label="Previous draws"
              disabled={view.start === 0}
              className="px-1 disabled:opacity-30"
              onClick={() => onChange({ ...view, start: Math.max(0, view.start - page) })}
            >
              ←
            </button>
            {view.start + 1}
            {page > 1 ? `–${Math.min(view.start + page, total)}` : ""} of {total}
            <button
              type="button"
              aria-label="Next draws"
              disabled={view.start + page >= total}
              className="px-1 disabled:opacity-30"
              onClick={() => onChange({ ...view, start: view.start + page })}
            >
              →
            </button>
          </span>
        )}
      </div>
      {paired && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="mr-1 text-[10px] uppercase tracking-wide text-muted-foreground">
            Arms
          </span>
          {(
            [
              ["both", "Both"],
              ["reference", "Reference"],
              ["intervened", "Intervened"],
            ] as const
          ).map(([arms, label]) => (
            <Choice
              key={arms}
              pressed={view.arms === arms}
              onClick={() => onChange({ ...view, arms })}
            >
              {label}
            </Choice>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * The selected simulation: the question's outcome first, then every state and simulated
 * observation in the same chart, all following one draw and arm control.
 */
export function SimulationEvidence({ context }: { context: ScopeContext }) {
  const { model, entities } = context;
  const simulation = model.simulation;
  const [view, setView] = useState<SimulationView>({ mode: "all", start: 0, arms: "both" });
  const selection = selectionOf(view);
  const paths = useSimulationPaths(model, selection);
  if (!simulation)
    return (
      <Section title="Simulation">
        <Hint>No simulation recorded at this version.</Hint>
      </Section>
    );
  const report = simulation;
  const names = new Map(entities.constructs.map((item) => [item.id, humanize(item.name)]));
  const outcome = hasCausalEffects(report)
    ? report.causal.value.outcome
    : (model.question?.outcome ?? null);
  const simulatedOutcome =
    outcome !== null && report.evidence.state_ids.some((id) => id === outcome) ? outcome : null;
  const timeLabel = (day: number) =>
    `${formatModelDate(day, report.evidence.time_origin)} (day ${day})`;
  const chart = { model, selection, arms: view.arms };
  return (
    <>
      <Section title="The question's outcome" wide>
        <SimulationControls
          view={view}
          onChange={setView}
          total={report.evidence.draws}
          paired={report.evidence.reference_latent_paths !== null}
        />
        {simulatedOutcome ? (
          <SimulationHistory
            {...chart}
            id={simulatedOutcome}
            kind="states"
            title={names.get(simulatedOutcome) ?? simulatedOutcome}
            height={200}
          />
        ) : (
          <Hint>The question names no simulated outcome.</Hint>
        )}
        {hasCausalEffects(report) ? (
          <>
            <Hint>
              Certified effect on{" "}
              {humanize(
                report.causal.value.labels[report.causal.value.outcome] ??
                  report.causal.value.outcome,
              )}{" "}
              at the end of the simulation.
            </Hint>
            {paths.data?.effect_summary && (
              <KeyValue
                rows={[
                  ["Mean", formatPlain(paths.data.effect_summary.mean)],
                  ["Median", formatPlain(paths.data.effect_summary.median)],
                  [
                    "95% interval",
                    `[${formatPlain(paths.data.effect_summary.lower_95)}, ${formatPlain(paths.data.effect_summary.upper_95)}]`,
                  ],
                  ["Probability positive", formatPlain(paths.data.effect_summary.prob_positive)],
                ]}
              />
            )}
            <SimulationHistory
              {...chart}
              id={report.causal.value.outcome}
              kind="effect"
              title="Paired effect"
              height={160}
            />
            {report.causal.value.warnings.map((warning) => (
              <Hint key={warning} issue>
                {warning}
              </Hint>
            ))}
          </>
        ) : (
          <Hint>{report.causal.kind !== "available" && report.causal.reason}</Hint>
        )}
      </Section>
      <Section title="Simulation design">
        <KeyValue
          rows={[
            ["Start", timeLabel(report.evidence.times[0])],
            ["Horizon", report.evidence.design.horizon],
            ["End", timeLabel(report.evidence.times.at(-1) ?? report.evidence.times[1])],
            ["Draws", report.evidence.draws.toLocaleString()],
            ["Fit reliability", humanize(report.fit_reliability)],
            ["Laws", humanize(report.law.interpretation)],
          ]}
        />
        {report.evidence.design.interventions.length === 0 ? (
          <Hint>No intervention requested.</Hint>
        ) : (
          report.evidence.design.interventions.map((event, index) => (
            <p key={`${event.target}-${event.after ?? "start"}`} className="m-0 border-t pt-2">
              {event.after ? `${event.after} after the start` : "At the start"} (
              {timeLabel(report.evidence.assignments[index]?.time ?? report.evidence.times[0])}):
              set {names.get(event.target) ?? event.target} to {event.value}.
            </p>
          ))
        )}
      </Section>
      <Section title="States" wide>
        <div className="grid grid-cols-2 gap-3">
          {report.evidence.state_ids
            .filter((id) => id !== simulatedOutcome)
            .map((id) => (
              <SimulationHistory
                key={id}
                {...chart}
                id={id}
                kind="states"
                title={names.get(id) ?? id}
                height={110}
              />
            ))}
        </div>
      </Section>
      <Section title="Simulated observations" wide>
        <div className="grid grid-cols-2 gap-3">
          {report.evidence.observation_layout.variables.map((variable) => (
            <SimulationHistory
              key={variable.id}
              {...chart}
              id={variable.id}
              kind="indicators"
              title={humanize(variable.name)}
              height={110}
            />
          ))}
        </div>
      </Section>
      {report.findings.length > 0 && (
        <Section title="Simulation checks" wide>
          <PredictiveFindings findings={report.findings} entities={entities} />
        </Section>
      )}
    </>
  );
}

export function PredictiveFindings({
  findings,
  entities,
}: {
  findings: SimulationReport["findings"];
  entities: ModelEntities;
}) {
  const names = new Map<string, string>([
    ...[...entities.constructs, ...entities.indicators].map((entity): [string, string] => [
      "observation" in entity ? entity.observation.id : entity.id,
      "observation" in entity ? entity.observation.name : entity.name,
    ]),
    ...entities.edges.flatMap((edge): [string, string][] => {
      const entity = resolveEntity(entities, { kind: "edge", id: edge.id });
      return entity ? [[edge.id, entity.label]] : [];
    }),
  ]);
  if (findings.length === 0) return null;
  return (
    <table className="w-full table-fixed text-left text-[11px] [overflow-wrap:anywhere]">
      <thead className="text-muted-foreground">
        <tr>
          <th className="w-1/3 pb-2 pr-2 font-medium">Check / target</th>
          <th className="w-1/5 pb-2 pr-2 font-medium">Measured</th>
          <th className="pb-2 font-medium">Criterion and finding</th>
        </tr>
      </thead>
      <tbody>
        {findings.map((finding) => {
          const target =
            typeof finding.subject.target === "string"
              ? finding.subject.target
              : finding.subject.target.id;
          return (
            <tr key={`${finding.subject.check}-${target}`} className="border-t align-top">
              <td className="py-2 pr-2">
                <div className="flex items-start gap-1">
                  <StatusIcon
                    status={finding.kind === "evaluated" ? finding.outcome : "not_evaluated"}
                  />
                  <span>{finding.subject.check.replaceAll("_", " ")}</span>
                </div>
                {target !== "whole_model" && (
                  <span className="mt-1 block text-[10px] text-muted-foreground">
                    {humanize(names.get(target) ?? target)}
                  </span>
                )}
              </td>
              <td className="py-2 pr-2 font-mono">
                {finding.kind === "evaluated"
                  ? finding.evidence
                      .map((item) => item.display_value || item.value.toLocaleString())
                      .join("; ")
                  : "Not evaluated"}
              </td>
              <td className="py-2">
                <span
                  className={
                    finding.kind === "evaluated" && finding.outcome === "failed"
                      ? "text-destructive"
                      : "text-muted-foreground"
                  }
                >
                  {finding.kind === "evaluated"
                    ? finding.evidence.map((item) => item.band_label).join("; ")
                    : humanize(finding.reason)}
                </span>
                <p className="mt-1 leading-relaxed text-muted-foreground">
                  {finding.kind === "evaluated"
                    ? finding.evidence.map((item) => item.note).join("; ")
                    : finding.detail}
                </p>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
