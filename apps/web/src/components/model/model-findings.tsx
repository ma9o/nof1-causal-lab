import { PPCWarningsTable } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, Section, StatusIcon } from "./scope-primitives";
import { PredictiveFindings } from "./simulation-evidence";
import { SpecificationFindings } from "./specification-findings";

/** Recorded checks for the selected version, without browser-side judgments. */
export function ModelFindings({ context }: { context: ScopeContext }) {
  const { specification, identification, validation_report, predictive } = context.model.findings;
  const comparison = predictive?.value.predictive_checks;
  const name = (id: string) =>
    humanize(context.entities.constructs.find((item) => item.id === id)?.name ?? id);
  return (
    <>
      {specification && (
        <Section title="Specification" source={specification.source}>
          <SpecificationFindings report={specification.value} />
        </Section>
      )}
      {identification && (
        <Section title="Identification" source={identification.source}>
          {identification.value.outcome && (
            <Hint>Outcome: {name(identification.value.outcome)}</Hint>
          )}
          {Object.keys(identification.value.treatments).length === 0 && (
            <Hint>No treatment identification findings recorded.</Hint>
          )}
          {Object.entries(identification.value.treatments).map(([id, finding]) => (
            <div key={id} className="space-y-1 border-b pb-2">
              <div className="flex items-start gap-2">
                <StatusIcon status={finding.status === "identified" ? "passed" : "failed"} />
                <span>
                  {name(id)} · {humanize(finding.status)}
                </span>
              </div>
              {finding.status === "identified" ? (
                <>
                  <Hint>{humanize(finding.method)}</Hint>
                  <details>
                    <summary className="cursor-pointer text-muted-foreground">Estimand</summary>
                    <p className="break-words font-mono">{finding.estimand}</p>
                  </details>
                </>
              ) : (
                <Hint issue>
                  {finding.notes}
                  {finding.confounders.length > 0 &&
                    ` Confounders: ${finding.confounders.map(name).join(", ")}.`}
                </Hint>
              )}
            </div>
          ))}
        </Section>
      )}
      {validation_report && (
        <Section title="Fit preflight" source={validation_report.source}>
          <SpecificationFindings report={validation_report.value.preflight} />
        </Section>
      )}
      {predictive && (
        <Section title="Predictive checks" source={predictive.source} wide>
          {predictive.value.status === "not_evaluated" && (
            <Hint>
              {predictive.value.detail ?? humanize(predictive.value.reason ?? "Not evaluated")}
            </Hint>
          )}
          <PredictiveFindings findings={predictive.value.findings} />
          {comparison && predictive.source.validity === "fresh" && (
            <PPCWarningsTable
              warnings={comparison.per_variable_warnings}
              testStats={comparison.test_stats}
              overlays={comparison.overlays}
              indicators={context.entities.indicators}
            />
          )}
        </Section>
      )}
    </>
  );
}
