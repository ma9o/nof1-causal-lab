import { PPCWarningsTable } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import { JsonViewer } from "@/components/ui/json-viewer";
import { ArtifactChip, FactChip, KeyValue, Prose, Section } from "../scope-primitives";
import { chipFor, type ScopeContext } from "./scope-context";

export function AssetScope({ context }: { context: ScopeContext }) {
  const { model } = context;
  const rawData = model.data.raw_data?.value;
  const fit = model.findings.fit?.value;
  const simulation = model.findings.simulation;
  const loo = fit?.report.loo_diagnostics;
  return (
    <>
      {context.question ? (
        <Section title="Question" chips={<ArtifactChip {...chipFor(context, "model")} />}>
          <Prose>{context.question}</Prose>
        </Section>
      ) : null}
      {rawData ? (
        <Section title="Raw data" chips={<ArtifactChip {...chipFor(context, "raw_data")} />}>
          <KeyValue
            rows={[
              ["records", rawData.n_records.toLocaleString()],
              ["columns", `${rawData.n_columns} · parquet`],
              ["span", `${rawData.date_range.start} → ${rawData.date_range.end}`],
            ]}
          />
        </Section>
      ) : null}
      {fit ? (
        <Section title="Fit" chips={<FactChip source={context.model.findings.fit?.source} />}>
          <KeyValue
            rows={[
              ["method", fit.report.inference_metadata.method.replaceAll("_", " ")],
              ["draws", fit.report.inference_metadata.n_samples.toLocaleString()],
              ...(loo
                ? ([
                    [
                      "LOO",
                      `elpd ${loo.elpd_loo.toFixed(0)} ± ${loo.se.toFixed(0)} · ${loo.n_bad_k ?? 0} bad k of ${loo.n_data_points}`,
                    ],
                  ] as Array<[string, React.ReactNode]>)
                : []),
            ]}
          />
          {Object.keys(fit.report.inference_diagnostics).length > 0 && (
            <details className="mt-2 text-xs">
              <summary className="cursor-pointer py-1 text-muted-foreground">
                Inference diagnostics
              </summary>
              <JsonViewer data={fit.report.inference_diagnostics} />
            </details>
          )}
        </Section>
      ) : null}
      {simulation ? (
        <Section title="Simulation" chips={<FactChip source={simulation.source} />}>
          <KeyValue
            rows={[
              ["draws", simulation.value.design.draws.toLocaleString()],
              ["model revision", String(simulation.value.model.version)],
            ]}
          />
          <div className="space-y-1 text-xs">
            {simulation.value.findings.map((finding, index) => (
              <p key={index}>
                {finding.check} · {finding.target}: {finding.value} ({finding.criterion}) —{" "}
                {finding.explanation}
              </p>
            ))}
          </div>
          {simulation.value.predictive_checks && simulation.source.validity === "fresh" ? (
            <PPCWarningsTable
              warnings={simulation.value.predictive_checks.per_variable_warnings}
              testStats={simulation.value.predictive_checks.test_stats}
              overlays={simulation.value.predictive_checks.overlays}
              indicators={context.entities.indicators}
            />
          ) : null}
        </Section>
      ) : null}
    </>
  );
}
