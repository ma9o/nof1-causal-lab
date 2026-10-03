import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatSignificant } from "@/lib/utils/format";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { ChainLegend, LatentStepChart } from "./fit-charts";
import { PosteriorPairs } from "./law-sections";
import { FitCalibration } from "./fit-calibration";

export function FitOutcome({ context }: { context: ScopeContext }) {
  const fit = context.model.fit;
  if (!fit) return <Hint>No inference report recorded.</Hint>;
  const { report } = fit.value;
  const { convergence } = report;
  const mcmc = report.inference_diagnostics;
  return (
    <Section title="Parameter convergence" source={fit.source}>
      <Hint>
        {humanize(report.inference_metadata.method)} · {mcmc?.num_chains ?? "Unavailable"} chains ×{" "}
        {mcmc?.num_samples ?? "Unavailable"} draws ·{" "}
        {report.inference_metadata.duration_seconds.toLocaleString()} s
      </Hint>
      <div className="flex items-center gap-2">
        <StatusIcon status={convergence.status} />
        <span>Parameter convergence: {humanize(convergence.status)}.</span>
      </div>
      {convergence.messages.map((message) => (
        <p key={message}>{message}</p>
      ))}
    </Section>
  );
}

/** Model-wide fitted evidence; individual laws own their chains. */
export function FitDetails({ context }: { context: ScopeContext }) {
  const fit = context.model.fit;
  const detail = useInferenceReport(context.model);
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  const { report } = fit.value;
  const plot = detail.data?.value.detail;
  const mcmc = report.inference_diagnostics;
  const gibbs = report.sampler_diagnostics;
  return (
    <>
      <FitCalibration context={context} />
      {(gibbs || plot?.final_latent_delta) && (
        <Section title="Latent mixing" source={fit.source} wide>
          <Hint>Parameter convergence does not assess latent-path mixing.</Hint>
          <KeyValue
            rows={[
              ["Latent moves accepted", gibbs?.latent_update_fraction],
              ["Latent coordinates frozen", gibbs?.latent_frozen_fraction],
              ["Mean latent move", gibbs?.latent_move_rms_mean],
            ]
              .filter((row): row is [string, number] => typeof row[1] === "number")
              .map(([label, value]): [string, string] => [label, formatSignificant(value)])}
          />
          {plot && <LatentStepChart detail={plot} />}
          {typeof mcmc?.num_chains === "number" && <ChainLegend chains={mcmc.num_chains} />}
        </Section>
      )}
      <PosteriorPairs context={context} />
    </>
  );
}
