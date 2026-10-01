import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import { recordValue } from "@/lib/model-asset/action-presentation";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatSignificant } from "@/lib/utils/format";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { ChainLegend, LatentStepChart, readLatentSteps } from "./fit-charts";
import { PosteriorPairs } from "./law-sections";
import { FitCalibration } from "./fit-calibration";

export function FitOutcome({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit;
  if (!fit) return <Hint>No inference report recorded.</Hint>;
  const { report, convergence } = fit.value;
  const mcmc = recordValue(report.inference_diagnostics.mcmc)!;
  return (
    <Section title="Parameter convergence" source={fit.source}>
      <Hint>
        {humanize(report.inference_metadata.method)} · {String(mcmc.num_chains)} chains ×{" "}
        {String(mcmc.num_samples)} draws ·{" "}
        {report.inference_metadata.duration_seconds.toLocaleString()} s
      </Hint>
      <div className="flex items-center gap-2">
        <StatusIcon
          status={!convergence.checked ? "not_evaluated" : convergence.passed ? "passed" : "failed"}
        />
        <span>
          {!convergence.checked
            ? "Parameter convergence not evaluated."
            : convergence.passed
              ? "Parameter convergence passed."
              : "Parameter convergence failed."}
        </span>
      </div>
      {[...new Set(convergence.failures.map((failure) => failure.criterion))].map((criterion) => (
        <p key={criterion}>
          {criterion} not met for{" "}
          {convergence.failures.filter((failure) => failure.criterion === criterion).length} of{" "}
          {convergence.checked} parameters.
        </p>
      ))}
    </Section>
  );
}

/** Model-wide fitted evidence; individual laws own their chains. */
export function FitDetails({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit;
  const detail = useInferenceReport(context.model);
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  const { report } = fit.value;
  const latentSteps = detail.data ? readLatentSteps(detail.data.value.inference_diagnostics) : null;
  const mcmc = recordValue(report.inference_diagnostics.mcmc);
  const gibbs = recordValue(report.inference_diagnostics.marginal_particle_gibbs);
  return (
    <>
      <FitCalibration context={context} />
      {(gibbs || latentSteps) && (
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
          {latentSteps && <LatentStepChart steps={latentSteps} />}
          {typeof mcmc?.num_chains === "number" && <ChainLegend chains={mcmc.num_chains} />}
        </Section>
      )}
      <PosteriorPairs context={context} />
    </>
  );
}
