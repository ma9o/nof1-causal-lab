import { signColor } from "@/components/dag/core/palette";
import { JsonViewer } from "@/components/ui/json-viewer";
import { formatPlain, formatSigned, humanize } from "../model-selection";
import {
  ArtifactChip,
  FactChip,
  Hint,
  KeyValue,
  OwnerLink,
  Prose,
  Section,
} from "../scope-primitives";
import { chipFor, type ScopeContext } from "./scope-context";

export function AssetScope({ context }: { context: ScopeContext }) {
  const { model, queries } = context;
  const rawData = model.data.raw_data?.value;
  const fit = model.findings.fit?.value;
  const commentary = model.findings.baseline_report?.value.final_summary;
  const sorted = queries
    .filter((query) => query.origin === "ranking" && query.posterior)
    .sort((left, right) => Math.abs(right.posterior!.mean) - Math.abs(left.posterior!.mean));
  const loo = fit?.report.assessment.loo_diagnostics;
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
              ["PPC", `${fit.predictive_checks_passed}/${fit.predictive_checks_total} checks pass`],
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
    </>
  );
}
