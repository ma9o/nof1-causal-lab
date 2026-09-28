import type { ModelSnapshot, PredictiveCheckFinding } from "@nof1-causal-lab/api-types";
import { hasCausalEffects } from "@/lib/simulation-report";
import { DefinitionView } from "./definition-view";
import { EffectChart } from "./effect-chart";
import { Section, StatusIcon } from "./scope-primitives";

/** One recorded simulation and its server-produced measurements, pinned to its input model. */
export function SimulationEvidence({ model }: { model: ModelSnapshot }) {
  const simulation = model.findings.simulation;
  if (!simulation) return null;
  const report = simulation.value;
  const fresh = simulation.source.validity === "fresh";
  return (
    <Section title="Simulation" source={simulation.source} wide>
      {fresh && hasCausalEffects(report) && <EffectChart simulation={report} />}
      <PredictiveFindings findings={report.findings} />
      {report.causal_unavailable_reason && (
        <p className="text-xs text-muted-foreground">{report.causal_unavailable_reason}</p>
      )}
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">Simulation design</summary>
        <div className="mt-2">
          <DefinitionView value={report.design} />
        </div>
      </details>
    </Section>
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
