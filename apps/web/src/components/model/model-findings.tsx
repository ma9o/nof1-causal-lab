import { JsonViewer } from "@/components/ui/json-viewer";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, KeyValue, Section, StatusIcon } from "./scope-primitives";
import { PredictiveFindings, SimulationEvidence } from "./simulation-evidence";
import { SpecificationFindings } from "./specification-findings";

/** Findings recorded for the version shared by the graph, details and chat. */
export function ModelFindings({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit?.value;
  const specification = context.model.findings.specification;
  const predictive = context.model.findings.predictive;
  const comparison = predictive?.value.predictive_checks;
  const loo = fit?.report.loo_diagnostics;
  return (
    <>
      {fit && (
        <Section title="Fit" source={context.model.findings.fit?.source}>
          {loo?.n_bad_k != null && loo.n_bad_k > 0 && (
            <div className="flex items-start gap-2">
              <StatusIcon status="warning" />
              <Hint issue>{loo.n_bad_k} observations have high Pareto k.</Hint>
            </div>
          )}
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground">
              Inference diagnostics
            </summary>
            <div className="mt-3 space-y-3">
              <KeyValue
                rows={[
                  ["Method", fit.report.inference_metadata.method.replaceAll("_", " ")],
                  ["Draws", fit.report.inference_metadata.n_samples.toLocaleString()],
                  ...(loo
                    ? ([["LOO", `elpd ${loo.elpd_loo.toFixed(0)} ± ${loo.se.toFixed(0)}`]] as Array<
                        [string, string]
                      >)
                    : []),
                  ...(loo?.n_bad_k != null
                    ? ([["High Pareto k", `${loo.n_bad_k} of ${loo.n_data_points}`]] as Array<
                        [string, string]
                      >)
                    : []),
                ]}
              />
              {Object.keys(fit.report.inference_diagnostics).length > 0 && (
                <JsonViewer data={fit.report.inference_diagnostics} />
              )}
            </div>
          </details>
        </Section>
      )}
      {specification && (
        <Section title="Checks" source={specification.source}>
          <SpecificationFindings report={specification.value} />
        </Section>
      )}
      {predictive && (
        <Section title="Predictive checks" source={predictive.source} wide>
          {predictive.value.status === "not_evaluated" && (
            <Hint>
              {predictive.value.detail ??
                predictive.value.reason?.replaceAll("_", " ").toLowerCase()}
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
      <SimulationEvidence model={context.model} />
    </>
  );
}

import { PPCWarningsTable } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
