import { signColor } from "@/components/dag/core/palette";
import { Button } from "@/components/ui/button";
import { formatPlain, formatSigned, humanize } from "../model-selection";
import { ArtifactChip, Hint, KeyValue, OwnerLink, Prose, Section, Tag } from "../scope-primitives";
import { chipFor, type ScopeContext } from "./scope-context";

export function AssetScope({ context }: { context: ScopeContext }) {
  const { model, queries } = context;
  const rawData = model.raw_data?.value;
  const fit = model.fit?.value;
  const saved = model.saved_scenarios?.value.scenarios ?? [];
  const commentary = model.baseline_report?.value.final_summary;
  const sorted = queries
    .filter((query) => query.origin === "ranking" && query.posterior)
    .sort((left, right) => Math.abs(right.posterior!.mean) - Math.abs(left.posterior!.mean));
  const mcmc = fit?.posterior.assessment.mcmc_diagnostics;
  const smc = fit?.posterior.assessment.smc_diagnostics;
  const loo = fit?.posterior.assessment.loo_diagnostics;
  return (
    <>
      {context.question ? (
        <Section title="Question" chips={<ArtifactChip {...chipFor(context, "question")} />}>
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
        <Section title="Fit" chips={<ArtifactChip {...chipFor(context, "posterior")} />}>
          <KeyValue
            rows={[
              ["method", fit.posterior.inference_metadata.method.replaceAll("_", " ")],
              ...(mcmc
                ? ([
                    [
                      "sampler",
                      `${mcmc.num_chains ?? "?"} chains × ${mcmc.num_samples ?? "?"} · accept ${mcmc.accept_prob_mean.toFixed(2)}`,
                    ],
                    [
                      "divergences",
                      <Tag key="div" tone={mcmc.num_divergences === 0 ? "success" : "destructive"}>
                        {mcmc.num_divergences}
                      </Tag>,
                    ],
                  ] as Array<[string, React.ReactNode]>)
                : []),
              ...(smc
                ? ([["SMC", `${smc.n_particles} particles · ${smc.n_levels} levels`]] as Array<
                    [string, React.ReactNode]
                  >)
                : []),
              ...(loo
                ? ([
                    [
                      "LOO",
                      `elpd ${loo.elpd_loo.toFixed(0)} ± ${loo.se.toFixed(0)} · ${loo.n_bad_k ?? 0} bad k of ${loo.n_data_points}`,
                    ],
                  ] as Array<[string, React.ReactNode]>)
                : []),
              ["PPC", `${fit.predictive_checks_passed}/${fit.predictive_checks_total} checks pass`],
            ]}
          />
        </Section>
      ) : null}
      {sorted.length > 0 ? (
        <Section
          title="Effects"
          wide
          chips={<ArtifactChip {...chipFor(context, "baseline_report")} />}
        >
          <Hint>
            do(+1 latent unit) per identified treatment, from the steady state, under the posterior
            {context.outcome ? (
              <>
                {" "}
                · outcome <b>{humanize(context.outcome)}</b>
              </>
            ) : null}{" "}
            · ranked by |effect at the steady state|
          </Hint>
          <table className="w-full table-fixed border-collapse text-[10px]">
            <colgroup>
              <col style={{ width: "36%" }} />
              <col style={{ width: "46%" }} />
              <col style={{ width: "18%" }} />
            </colgroup>
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="border-b pb-0.5 pr-1.5 font-medium">treatment</th>
                <th className="border-b pb-0.5 pr-1.5 font-medium">steady state</th>
                <th className="border-b pb-0.5 font-medium">P&gt;0</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((query) => {
                const treatment =
                  context.entities.constructById.get(query.treatmentId!)?.name ?? query.title;
                const result = query.posterior;
                if (!result) return null;
                return (
                  <tr key={query.key}>
                    <td className="truncate border-b py-0.5 pr-1.5">
                      <OwnerLink
                        onClick={() =>
                          context.select({ kind: "construct", id: query.treatmentId! })
                        }
                      >
                        {treatment}
                      </OwnerLink>
                    </td>
                    <td className="truncate border-b py-0.5 pr-1.5 font-mono">
                      <span style={{ color: signColor(result.mean) }}>
                        {formatSigned(result.mean, 3)}
                      </span>{" "}
                      <span className="text-muted-foreground">
                        [{formatPlain(result.lower_95)}, {formatPlain(result.upper_95)}]
                      </span>
                    </td>
                    <td className="border-b py-0.5 font-mono">
                      {Math.round(result.prob_positive * 100)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Section>
      ) : null}
      {commentary ? (
        <Section
          title="Commentary"
          chips={<ArtifactChip {...chipFor(context, "baseline_report")} />}
        >
          <Hint>
            <span className="line-clamp-6">{commentary}</span>
          </Hint>
        </Section>
      ) : null}
      {saved.length > 0 ? (
        <Section
          title="Saved scenarios"
          chips={<ArtifactChip {...chipFor(context, "saved_scenarios")} />}
        >
          <ul className="m-0 flex list-none flex-col gap-1 p-0 text-[11px]">
            {saved.map((scenario) => (
              <li key={scenario.query.id} className="flex flex-col">
                <Button
                  type="button"
                  variant="link"
                  size="xs"
                  className="h-auto justify-start p-0 text-[11px] font-semibold"
                  onClick={() => context.select({ kind: "query", key: scenario.query.id })}
                >
                  {scenario.label}
                </Button>
                <Hint>
                  {scenario.query.clamps.map((clamp) => clamp.variable).join(", ")} ·{" "}
                  {scenario.query.readout.horizon_days} days · {scenario.evaluations.length}{" "}
                  evaluations
                </Hint>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </>
  );
}
