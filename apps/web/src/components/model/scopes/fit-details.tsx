import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import { recordValue } from "@/lib/model-asset/action-presentation";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatSignificant } from "@/lib/utils/format";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, OwnerLink, Section, StatusIcon } from "../scope-primitives";
import { ChainLegend, LatentStepChart, readLatentSteps } from "./fit-charts";
import { PosteriorPairs } from "./law-sections";

/** The fit owns its verdict and joint diagnostics; individual laws own their chains. */
export function FitDetails({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit;
  const detail = useInferenceReport(context.model);
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  const { report, convergence } = fit.value;
  const latentSteps = detail.data ? readLatentSteps(detail.data.value.inference_diagnostics) : null;
  const mcmc = recordValue(report.inference_diagnostics.mcmc);
  const gibbs = recordValue(report.inference_diagnostics.marginal_particle_gibbs);
  const failures = [
    ...new Map(
      convergence.failures.map((failure) => [failure.subject.parameter_id, failure]),
    ).values(),
  ];
  return (
    <>
      <Section title="Parameter convergence" source={fit.source}>
        <div className="flex items-center gap-2">
          <StatusIcon
            status={
              !convergence.checked ? "not_evaluated" : convergence.passed ? "passed" : "failed"
            }
          />
          <span>
            {!convergence.checked
              ? "No parameter convergence diagnostics recorded."
              : convergence.passed
                ? "Passed"
                : "Failed"}
          </span>
        </div>
        {failures.map((failure) => {
          const owner = parameterOwner(context.entities, failure.subject.parameter_id)!;
          return (
            <OwnerLink
              key={failure.subject.parameter_id}
              onClick={() => context.select(owner.selection)}
            >
              {humanize(context.entities.parameterById.get(failure.subject.parameter_id)!.name)}
            </OwnerLink>
          );
        })}
        <KeyValue
          rows={[
            ["Method", humanize(report.inference_metadata.method)],
            ["Draws", report.inference_metadata.n_samples.toLocaleString()],
            ["Chains", String(mcmc?.num_chains)],
            ["Duration", `${report.inference_metadata.duration_seconds.toLocaleString()} s`],
          ]}
        />
        {fit.source.validity === "stale" && (
          <Hint>
            These results use the fit’s pinned observation panel, which differs from the panel
            selected at this version.
          </Hint>
        )}
      </Section>
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
