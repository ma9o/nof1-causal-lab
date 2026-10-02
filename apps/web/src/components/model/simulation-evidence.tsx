import { presentEntries } from "@/lib/model-accessors";
import type { ModelPredictiveReport } from "@nof1-causal-lab/api-types";
import { resolveEntity, type ModelEntities } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatPlain, humanize } from "@/lib/model-asset/selection";
import { hasCausalEffects } from "@/lib/simulation-report";
import { formatModelDate } from "@/lib/utils/format";
import { SimulationHistory } from "./scopes/recorded-history";
import { EffectChart } from "./effect-chart";
import { Hint, KeyValue, Section, StatusIcon } from "./scope-primitives";

/** The design, saved histories and certified effect of the selected simulation. */
export function SimulationEvidence({ context }: { context: ScopeContext }) {
  const { model, entities } = context;
  const simulation = model.findings.simulation;
  if (!simulation)
    return (
      <Section title="Simulation">
        <Hint>No simulation recorded at this version.</Hint>
      </Section>
    );
  const report = simulation.value;
  const names = new Map(entities.constructs.map((item) => [item.id, humanize(item.name)]));

  const timeLabel = (day: number) =>
    report.time_origin ? `${formatModelDate(day, report.time_origin)} (day ${day})` : `Day ${day}`;
  return (
    <>
      <Section title="Causal effect" source={simulation.source} wide>
        {hasCausalEffects(report) ? (
          <>
            <Hint>
              Certified effect on{" "}
              {humanize(
                report.causal_result.labels[report.causal_result.outcome] ??
                  report.causal_result.outcome,
              )}{" "}
              at the end of the simulation.
            </Hint>
            <KeyValue
              rows={[
                ["Mean", formatPlain(report.causal_result.summary.mean)],
                [
                  "95% interval",
                  `[${formatPlain(report.causal_result.summary.lower_95)}, ${formatPlain(report.causal_result.summary.upper_95)}]`,
                ],
              ]}
            />
            <EffectChart simulation={report} />
            <SimulationHistory model={model} id={report.causal_result.outcome} kind="effect" />

            {report.causal_result.warnings.map((warning) => (
              <Hint key={warning} issue>
                {warning}
              </Hint>
            ))}
          </>
        ) : (
          <Hint>
            {report.causal_unavailable_reason ??
              (report.design.interventions.length
                ? "No causal effect was recorded."
                : "No intervention was requested.")}
          </Hint>
        )}
      </Section>
      <Section title="Simulation design" source={simulation.source}>
        <KeyValue
          rows={[
            ["Start", timeLabel(report.times[0])],
            ["End", timeLabel(report.design.end)],
            ["Draws", report.draws.toLocaleString()],
            ["Fit reliability", humanize(report.predictive.fit_reliability)],
            ["Laws", humanize(report.law?.interpretation ?? "unknown")],
          ]}
        />
        {report.design.interventions.length === 0 ? (
          <Hint>No intervention requested.</Hint>
        ) : (
          report.design.interventions.map((event) => (
            <p key={`${event.target}-${event.time}`} className="m-0 border-t pt-2">
              {timeLabel(event.time)}: set {names.get(event.target) ?? event.target} to{" "}
              {event.value}.
            </p>
          ))
        )}
      </Section>
      {(["states", "indicators"] as const).flatMap((kind) =>
        presentEntries(report.predictive[kind]).map(([id, series]) => (
          <Section key={id} title={humanize(series.label)} source={simulation.source} wide>
            <SimulationHistory model={model} id={id} kind={kind} summary={series} />
          </Section>
        )),
      )}
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
  findings: ModelPredictiveReport["findings"];
  entities: ModelEntities;
}) {
  const names = new Map<string, string>([
    ...[...entities.constructs, ...entities.indicators].map((entity): [string, string] => [
      entity.id,
      entity.name,
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
