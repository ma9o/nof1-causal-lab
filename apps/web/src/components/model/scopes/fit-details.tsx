import { JsonViewer } from "@/components/ui/json-viewer";
import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import { recordValue } from "@/lib/model-asset/action-presentation";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, PosteriorTable, Section, StatusIcon } from "../scope-primitives";
import {
  ChainLegend,
  coordinateKey,
  LatentStepChart,
  RankBars,
  readChainDiagnostics,
  readLatentSteps,
  TraceSparkline,
} from "./fit-charts";

const metric = (value: unknown) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumSignificantDigits: 5 })
    : "Unavailable";

const share = (value: unknown) =>
  typeof value === "number"
    ? `${(value * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`
    : null;

export function FitDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const fit = context.model.findings.fit;
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  return <FitFindings context={context} tick={tick} fit={fit} />;
}

function FitFindings({
  context,
  tick,
  fit,
}: {
  context: ScopeContext;
  tick: JournalTick;
  fit: NonNullable<ScopeContext["model"]["findings"]["fit"]>;
}) {
  const { workspace_id, commit_id, branch } = context.model.context;
  // The snapshot carries the summary; per-draw diagnostics load with the full report.
  const detail = useInferenceReport(workspace_id, commit_id, branch);
  const diagnostics = detail.data?.value.inference_diagnostics;
  const chains = diagnostics ? readChainDiagnostics(diagnostics) : null;
  const latentSteps = diagnostics ? readLatentSteps(diagnostics) : null;
  const report = fit.value.report;
  const warning = tick.messages.some((message) => message.label === "CONVERGENCE_CHECK_FAILED");
  const mcmc = recordValue(report.inference_diagnostics.mcmc);
  const gibbs = recordValue(report.inference_diagnostics.marginal_particle_gibbs);
  const chainCount = typeof mcmc?.num_chains === "number" ? mcmc.num_chains : null;
  const parameters = Array.isArray(mcmc?.per_parameter)
    ? mcmc.per_parameter.map(recordValue).filter((entry) => entry !== null)
    : [];
  const latentRows: Array<[string, string]> = [
    ["Latent moves accepted", share(gibbs?.latent_update_fraction)],
    ["Latent coordinates frozen", share(gibbs?.latent_frozen_fraction)],
    [
      "Mean latent move",
      typeof gibbs?.latent_move_rms_mean === "number" ? metric(gibbs.latent_move_rms_mean) : null,
    ],
  ].filter((row): row is [string, string] => row[1] !== null);
  const edges = Object.entries(fit.value.edge_estimates).map(([id, estimate]) => {
    const edge = context.entities.edges.find((edge) => edge.id === id);
    const label = edge
      ? `${context.entities.constructById.get(edge.cause.id)!.name} → ${context.entities.constructById.get(edge.effect.id)!.name}`
      : id;
    return { ...estimate, parameter: label };
  });
  const pending = <span className="text-muted-foreground">…</span>;
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
            ...(chainCount !== null ? [["Chains", String(chainCount)] as [string, string]] : []),
            ...latentRows,
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
          {diagnostics ? (
            <JsonViewer data={diagnostics} />
          ) : (
            <Hint>{detail.error ? detail.error.message : "Loading diagnostics…"}</Hint>
          )}
        </details>
      </Section>
      <Section title="Parameter diagnostics" source={fit.source} wide>
        {parameters.length ? (
          <>
            <table className="w-full text-left text-[10px]">
              <thead className="text-muted-foreground">
                <tr>
                  {[
                    "Parameter",
                    "R-hat",
                    "ESS bulk",
                    "ESS tail",
                    "MCSE mean",
                    "Draws",
                    "Ranks",
                  ].map((label) => (
                    <th key={label} className="pb-2 pr-2 font-medium">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {parameters.map((parameter) => {
                  const name = typeof parameter.parameter === "string" ? parameter.parameter : null;
                  const coordinate = coordinateKey(parameter.coordinate);
                  const trace = coordinate ? chains?.traces.get(coordinate) : undefined;
                  const ranks = coordinate ? chains?.ranks.get(coordinate) : undefined;
                  return (
                    <tr
                      key={name ?? JSON.stringify(parameter.coordinate)}
                      className="border-t align-top"
                    >
                      <td className="break-words py-2 pr-2">
                        {name ? humanize(name) : "Unnamed parameter"}
                      </td>
                      {["r_hat", "ess_bulk", "ess_tail", "mcse_mean"].map((key) => (
                        <td key={key} className="py-2 pr-2 font-mono">
                          {metric(parameter[key])}
                        </td>
                      ))}
                      <td className="py-2 pr-2">
                        {chains ? trace ? <TraceSparkline chains={trace} /> : "—" : pending}
                      </td>
                      <td className="py-2 pr-2">
                        {chains ? ranks ? <RankBars histogram={ranks} /> : "—" : pending}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {chainCount !== null && (
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                <ChainLegend chains={chainCount} />
                <span className="text-[10px] text-muted-foreground">
                  Ranks: each chain’s share of the pooled draws; mixed chains stay near the dashed
                  line.
                </span>
              </div>
            )}
          </>
        ) : (
          <Hint>No per-parameter diagnostics retained.</Hint>
        )}
      </Section>
      {latentSteps && (
        <Section title="Latent step size" source={fit.source} wide>
          <LatentStepChart steps={latentSteps} />
          <Hint>
            Latent proposal step size at each time point after warmup adaptation, log scale; the
            dashed line is the starting size. Very small steps mark where latent paths barely move.
          </Hint>
          {chainCount !== null && <ChainLegend chains={chainCount} />}
        </Section>
      )}
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
