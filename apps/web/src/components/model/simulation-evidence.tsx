import type { ModelSnapshot, PredictiveCheckFinding } from "@nof1-causal-lab/api-types";
import { useSimulationTrajectories } from "@/lib/hooks/use-simulation-trajectories";
import { modelConstructs } from "@/lib/model-accessors";
import { formatPlain, humanize } from "@/lib/model-asset/selection";
import { hasCausalEffects } from "@/lib/simulation-report";
import { formatModelDate } from "@/lib/utils/format";
import { EffectChart } from "./effect-chart";
import { Hint, KeyValue, Section, StatusIcon } from "./scope-primitives";
import { TrajectoryChart } from "./trajectory-chart";

/** The design, saved histories and certified effect of the selected simulation. */
export function SimulationEvidence({ model }: { model: ModelSnapshot }) {
  const simulation = model.findings.simulation;
  const projection = useSimulationTrajectories(
    model.context.workspace_id,
    model.context.commit_id,
    !!simulation,
  );
  if (!simulation)
    return (
      <Section title="Simulation">
        <Hint>No simulation recorded at this version.</Hint>
      </Section>
    );
  const report = simulation.value;
  const names = new Map(
    modelConstructs(model.model?.value).map((item) => [item.id, humanize(item.name)]),
  );
  const trajectories = projection.data?.value;
  const timeLabel = (day: number) =>
    trajectories?.time_origin
      ? `${formatModelDate(day, trajectories.time_origin)} (day ${day})`
      : `Day ${day}`;
  return (
    <>
      <Section title="Simulation design" source={simulation.source}>
        <KeyValue
          rows={[
            ["Start", timeLabel(report.times[0])],
            ["End", timeLabel(report.design.end)],
            ["Draws", report.draws.toLocaleString()],
            ["Seed", String(report.seed)],
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
        <Hint>
          Shading shows pointwise 95% intervals across saved draws (2.5th–97.5th percentiles).
        </Hint>
      </Section>
      {projection.error ? (
        <Section title="Outcome trajectories" wide>
          <p role="alert" className="text-xs text-destructive">
            Unable to read saved trajectories: {projection.error.message}
          </p>
        </Section>
      ) : !projection.isSuccess ? (
        <Section title="Outcome trajectories" wide>
          <p role="status">Reading saved trajectories…</p>
        </Section>
      ) : trajectories?.outcome_state ? (
        <>
          <Section
            title={humanize(trajectories.outcome_state.label)}
            source={projection.data?.source}
            wide
          >
            <Hint>Latent outcome</Hint>
            <TrajectoryChart
              times={trajectories.times}
              timeOrigin={trajectories.time_origin}
              series={trajectories.outcome_state}
            />
          </Section>
          {Object.entries(trajectories.indicators).map(([id, series]) => (
            <Section key={id} title={humanize(series.label)} source={projection.data?.source} wide>
              <Hint>Simulated indicator · gaps mark unobserved anchors</Hint>
              <TrajectoryChart
                times={trajectories.times}
                timeOrigin={trajectories.time_origin}
                series={series}
              />
            </Section>
          ))}
        </>
      ) : (
        <Section title="Outcome trajectories">
          <Hint>No default outcome is recorded for this simulation.</Hint>
        </Section>
      )}
      <Section title="Causal effect" source={simulation.source} wide>
        {hasCausalEffects(report) ? (
          <>
            <Hint>
              Certified effect on{" "}
              {humanize(report.causal_result.labels[report.causal_result.outcome])} at the end of
              the simulation.
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
      {report.findings.length > 0 && (
        <Section title="Simulation checks" source={simulation.source} wide>
          <PredictiveFindings findings={report.findings} />
        </Section>
      )}
    </>
  );
}

export function PredictiveFindings({ findings }: { findings: PredictiveCheckFinding[] }) {
  return (
    <>
      {findings.length > 0 && (
        <div>
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="pb-2 pr-3 font-medium">Check / target</th>
                <th className="pb-2 pr-3 font-medium">Measured</th>
                <th className="pb-2 font-medium">Criterion and finding</th>
              </tr>
            </thead>
            <tbody>
              {findings.map((finding) => (
                <tr key={`${finding.check}-${finding.target}`} className="border-t align-top">
                  <td className="py-2 pr-3">
                    <div className="flex items-start gap-2">
                      {finding.passed !== null && (
                        <StatusIcon status={finding.passed ? "passed" : "failed"} />
                      )}
                      <span>{finding.check.replaceAll("_", " ")}</span>
                    </div>
                    {finding.target !== "model" && (
                      <span className="mt-1 block text-[10px] text-muted-foreground">
                        {finding.target}
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3 font-mono">{finding.value}</td>
                  <td className="py-2">
                    <span
                      className={
                        finding.passed === false ? "text-destructive" : "text-muted-foreground"
                      }
                    >
                      {finding.band}
                    </span>
                    <p className="mt-1 leading-relaxed text-muted-foreground">{finding.note}</p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
