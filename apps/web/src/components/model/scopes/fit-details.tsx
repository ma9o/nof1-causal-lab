import { JsonViewer } from "@/components/ui/json-viewer";
import { recordValue } from "@/lib/model-asset/action-presentation";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, PosteriorTable, Section, StatusIcon } from "../scope-primitives";

const metric = (value: unknown) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumSignificantDigits: 5 })
    : "Unavailable";

export function FitDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const fit = context.model.findings.fit;
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  const report = fit.value.report;
  const warning = tick.messages.some((message) => message.label === "CONVERGENCE_CHECK_FAILED");
  const mcmc = recordValue(report.inference_diagnostics.mcmc);
  const parameters = Array.isArray(mcmc?.per_parameter)
    ? mcmc.per_parameter.map(recordValue).filter((entry) => entry !== null)
    : [];
  const edges = Object.entries(fit.value.edge_estimates).map(([id, estimate]) => {
    const edge = context.entities.edges.find((edge) => edge.id === id);
    const label = edge
      ? `${context.entities.constructById.get(edge.cause.id)!.name} → ${context.entities.constructById.get(edge.effect.id)!.name}`
      : id;
    return { ...estimate, parameter: label };
  });
  return (
    <>
      <Section title="Fit outcome" source={fit.source}>
        <div className="flex items-start gap-2">
          {warning && <StatusIcon status="warning" />}
          <Hint issue={warning}>
            {warning
              ? "Convergence warning recorded by the fit."
              : "No convergence warning recorded."}
          </Hint>
        </div>
        <KeyValue
          rows={[
            ["Method", humanize(report.inference_metadata.method)],
            ["Draws", report.inference_metadata.n_samples.toLocaleString()],
            ["Duration", `${metric(report.inference_metadata.duration_seconds)} s`],
            ...(typeof mcmc?.num_chains === "number"
              ? [["Chains", String(mcmc.num_chains)] as [string, string]]
              : []),
          ]}
        />
        {fit.source.validity === "stale" && (
          <Hint>
            These results use the fit’s pinned observation panel, which differs from the panel
            selected at this version.
          </Hint>
        )}
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">
            Full inference diagnostics
          </summary>
          <JsonViewer data={report.inference_diagnostics} />
        </details>
      </Section>
      <Section title="Parameter diagnostics" source={fit.source} wide>
        {parameters.length ? (
          <table className="w-full text-left text-[10px]">
            <thead className="text-muted-foreground">
              <tr>
                {["Parameter", "R-hat", "ESS bulk", "ESS tail", "MCSE mean"].map((label) => (
                  <th key={label} className="pb-2 pr-2 font-medium">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parameters.map((parameter, index) => (
                <tr key={index} className="border-t align-top">
                  <td className="break-all py-2 pr-2">
                    {typeof parameter.parameter === "string"
                      ? humanize(parameter.parameter)
                      : "Unnamed parameter"}
                  </td>
                  {["r_hat", "ess_bulk", "ess_tail", "mcse_mean"].map((name) => (
                    <td key={name} className="py-2 pr-2 font-mono">
                      {metric(parameter[name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Hint>No per-parameter diagnostics retained.</Hint>
        )}
      </Section>
      <Section title="Edge effect posteriors" source={fit.source} wide>
        {edges.length ? (
          <PosteriorTable rows={edges} />
        ) : (
          <Hint>
            {fit.source.validity === "stale"
              ? "Edge summaries are unavailable for the selected model and panel. The recorded fit diagnostics remain above."
              : "No edge effect summaries recorded."}
          </Hint>
        )}
      </Section>
    </>
  );
}
