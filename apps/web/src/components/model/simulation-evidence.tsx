import type { ModelPredictiveReport } from "@nof1-causal-lab/api-types";
import { resolveEntity, type ModelEntities } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatPlain, humanize } from "@/lib/model-asset/selection";
import { hasCausalEffects } from "@/lib/simulation-report";
import { useSimulationPaths } from "@/lib/hooks/use-visuals";
import { formatModelDate } from "@/lib/utils/format";
import { SimulationHistory } from "./scopes/recorded-history";
import { Hint, KeyValue, Section, StatusIcon } from "./scope-primitives";

/** The design, saved histories and certified effect of the selected simulation. */
export function SimulationEvidence({ context }: { context: ScopeContext }) {
  const { model, entities } = context;
  const simulation = model.simulation;
  const paths = useSimulationPaths(model);
  if (!simulation)
    return (
      <Section title="Simulation">
        <Hint>No simulation recorded at this version.</Hint>
      </Section>
    );
  const report = simulation.value;
  const names = new Map(entities.constructs.map((item) => [item.id, humanize(item.name)]));

  const timeLabel = (day: number) => `${formatModelDate(day, report.evidence.time_origin)} (day ${day})`;
  return (
    <>
      <Section title="Causal effect" source={simulation.source} wide>
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
            <SimulationHistory model={model} id={report.causal.value.outcome} kind="effect" />

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
      <Section title="Simulation design" source={simulation.source}>
        <KeyValue
          rows={[
            ["Start", timeLabel(report.evidence.times[0])],
            ["Horizon", report.evidence.design.horizon],
            ["End", timeLabel(report.evidence.times.at(-1) ?? report.evidence.times[1])],
            ["Draws", report.evidence.draws.toLocaleString()],
            ["Fit reliability", humanize(report.fit_reliability)],
            ["Laws", humanize(report.law?.interpretation ?? "unknown")],
          ]}
        />
        {report.evidence.design.interventions.length === 0 ? (
          <Hint>No intervention requested.</Hint>
        ) : (
          report.evidence.design.interventions.map((event, index) => (
            <p key={`${event.target}-${event.after ?? "start"}`} className="m-0 border-t pt-2">
              {event.after ? `${event.after} after the start` : "At the start"} (
              {timeLabel(report.evidence.assignments[index]?.time ?? report.evidence.times[0])}): set{" "}
              {names.get(event.target) ?? event.target} to {event.value}.
            </p>
          ))
        )}
      </Section>
      {report.evidence.state_ids.map((id) => (
        <Section key={id} title={names.get(id) ?? id} source={simulation.source} wide>
          <SimulationHistory model={model} id={id} kind="states" />
        </Section>
      ))}
      {report.evidence.observation_layout.variables.map((variable) => (
        <Section key={variable.id} title={humanize(variable.name)} source={simulation.source} wide>
          <SimulationHistory model={model} id={variable.id} kind="indicators" />
        </Section>
      ))}
      {report.findings.length > 0 && (
        <Section title="Simulation checks" source={simulation.source} wide>
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
  findings: Extract<ModelPredictiveReport["evaluation"], { kind: "evaluated" }>["findings"];
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
