import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import { recordValue } from "@/lib/model-asset/action-presentation";
import { distributionText } from "@/lib/utils/distribution-format";
import {
  ChainLegend,
  coordinateKey,
  readChainDiagnostics,
  TraceSparkline,
  RankBars,
} from "./fit-charts";
import { LawChart } from "@/components/charts/law-density";
import { PosteriorPairsChart } from "@/components/charts/posterior-pairs-chart";
import { useParameterDraws } from "@/lib/hooks/use-visuals";
import { useState } from "react";
import type { CoefficientUse } from "@/lib/model-accessors";
import { type LawCurve, lawCurves, lawLabel } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { SimulationHistory } from "./recorded-history";

function interval(curve: LawCurve): string {
  const days = curve.parameter.reference_interval_days;
  return days == null ? "model clock tick" : `${days} d`;
}

/** Only scale changes and missing priors need words; the legend names the curves. */
function lawHint(curve: LawCurve): string | null {
  if (curve.kind === "fitted")
    return curve.prior.length > 0
      ? null
      : "The fit's input model has no prior curve on the fitted scale.";
  if (curve.parameter.distribution_transform === "dt_persistence_to_ct_decay")
    return `Authored as persistence per ${interval(curve)}; a fit reports decay rates.`;
  if (curve.parameter.distribution_transform === "dt_effect_to_ct_rate")
    return `Authored as the effect over ${interval(curve)}; a fit reports rates per day.`;
  return null;
}

/** One section per law an entity's own terms name, with its prior and any posterior. */
export function LawSections({ context, uses }: { context: ScopeContext; uses: CoefficientUse[] }) {
  const fit = context.model.findings.fit;
  const detail = useInferenceReport(context.model);
  const chains = detail.data ? readChainDiagnostics(detail.data.value.inference_diagnostics) : null;
  const mcmc = recordValue(fit?.value.report.inference_diagnostics.mcmc);
  const rows = Array.isArray(mcmc?.per_parameter) ? mcmc.per_parameter.map(recordValue) : [];
  const curves = lawCurves(context.model, uses);
  return uses.map((use) => {
    const parameter = context.entities.parameterById.get(use.parameterId)!;
    const law = parameter.distribution
      ? context.model.model!.value.distributions[parameter.distribution]
      : null;
    const curve = curves.find((item) => item.parameter.id === parameter.id);
    const label = curve ? lawLabel(curve) : humanize(use.role);
    const diagnostics = rows.filter(
      (row) => recordValue(row!.subject)!.parameter_id === parameter.id,
    );
    return (
      <Section
        key={parameter.id}
        title={label.charAt(0).toUpperCase() + label.slice(1)}
        source={curve?.kind === "fitted" ? fit?.source : undefined}
      >
        <Hint>{humanize(parameter.description)}</Hint>
        {curve ? (
          <LawChart curve={curve} caption={humanize(parameter.name)} />
        ) : (
          <p>{humanize(parameter.name)}</p>
        )}
        {(!curve || law?.distribution === "Delta") &&
          (law ? (
            <p className="break-words font-mono">{distributionText(law)}</p>
          ) : (
            <Hint>No law assigned.</Hint>
          ))}
        {curve && lawHint(curve) && <Hint>{lawHint(curve)}</Hint>}
        {diagnostics.map((row) => {
          const subject = recordValue(row!.subject)!;
          const failures = fit!.value.convergence.failures.filter(
            (failure) => failure.subject.element_id === subject.element_id,
          );
          const key = coordinateKey(row!.coordinate);
          const trace = key ? chains?.traces.get(key) : undefined;
          const ranks = key ? chains?.ranks.get(key) : undefined;
          return (
            <div key={String(subject.element_id)} className="space-y-2 border-t pt-2">
              <div className="flex items-center gap-2">
                <StatusIcon status={failures.length ? "failed" : "passed"} />
                <span>{humanize(String(row!.parameter))}</span>
              </div>
              {failures.map((failure) => (
                <Hint key={failure.criterion} issue>
                  {failure.criterion}
                </Hint>
              ))}
              <KeyValue
                rows={[
                  ["R-hat", row!.r_hat],
                  ["ESS bulk", row!.ess_bulk],
                  ["ESS tail", row!.ess_tail],
                  ["MCSE mean", row!.mcse_mean],
                ].map(([name, value]) => [
                  String(name),
                  typeof value === "number"
                    ? value.toLocaleString(undefined, { maximumSignificantDigits: 5 })
                    : "Unavailable",
                ])}
              />
              <div className="flex gap-3">
                {trace && <TraceSparkline chains={trace} />}
                {ranks && <RankBars histogram={ranks} />}
              </div>
            </div>
          );
        })}
        {diagnostics.length > 0 &&
          chains &&
          (chains.traces.size > 0 || chains.ranks.size > 0) &&
          typeof mcmc?.num_chains === "number" && <ChainLegend chains={mcmc.num_chains} />}
      </Section>
    );
  });
}

/** The saved simulation's history of one state or indicator, when it was simulated. */
export function SimulatedHistory({
  context,
  id,
  kind,
}: {
  context: ScopeContext;
  id: string;
  kind: "states" | "indicators";
}) {
  const simulation = context.model.findings.simulation;
  const series = simulation?.value.predictive[kind][id];
  if (!simulation || !series) return null;
  return (
    <Section title="Simulated history" source={simulation.source} wide>
      <SimulationHistory model={context.model} id={id} kind={kind} summary={series} />
    </Section>
  );
}

/** All retained coordinates and their empirical marginal, without a preselected report subset. */
export function PosteriorPairs({ context }: { context: ScopeContext }) {
  const fit = context.model.findings.fit;
  const draws = useParameterDraws(context.model);
  const [xId, setX] = useState<string | null>(null);
  const [yId, setY] = useState<string | null>(null);
  const columns = draws.data?.columns ?? [];
  const x = columns.find((column) => column.subject.element_id === xId) ?? columns[0];
  const y =
    columns.find((column) => column.subject.element_id === yId) ??
    columns.find((column) => column.subject.element_id !== x?.subject.element_id);
  if (!fit) return null;
  return (
    <Section title="Joint posterior" source={fit.source} wide>
      {draws.error ? (
        <Hint issue>{draws.error.message}</Hint>
      ) : draws.isLoading ? (
        <Hint>Loading retained draws…</Hint>
      ) : draws.data?.unavailable_reason ? (
        <Hint>{draws.data.unavailable_reason}</Hint>
      ) : x ? (
        <>
          {(
            [
              ["X coordinate", x, setX],
              ["Y coordinate", y, setY],
            ] as const
          ).map(
            ([label, column, set]) =>
              column && (
                <label key={label} className="flex items-center gap-2 text-[10px]">
                  {label}
                  <select
                    className="min-w-0 flex-1 rounded border bg-background p-1"
                    value={column.subject.element_id}
                    onChange={(event) => set(event.target.value)}
                  >
                    {columns.map((choice) => (
                      <option key={choice.subject.element_id} value={choice.subject.element_id}>
                        {choice.label}
                      </option>
                    ))}
                  </select>
                </label>
              ),
          )}
          {y && (
            <PosteriorPairsChart
              pair={{
                param_x: x.label,
                subject_x: x.subject,
                x_values: x.values,
                param_y: y.label,
                subject_y: y.subject,
                y_values: y.values,
              }}
            />
          )}
          <Hint>
            All {x.values.length.toLocaleString()} aligned retained draws. Choose any of{" "}
            {columns.length} parameter coordinates; no thinning or preselection.
          </Hint>
        </>
      ) : (
        <Hint>No retained parameter coordinates.</Hint>
      )}
    </Section>
  );
}
