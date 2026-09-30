import type { ParameterId } from "@nof1-causal-lab/api-types";
import { LawChart } from "@/components/charts/law-density";
import { PosteriorPairsChart } from "@/components/charts/posterior-pairs-chart";
import { useParameterDraws } from "@/lib/hooks/use-visuals";
import { useState } from "react";
import type { CoefficientUse } from "@/lib/model-accessors";
import { type LawCurve, lawCurves, lawLabel } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";
import { EmpiricalPlot, SimulationHistory } from "./recorded-history";

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
export function LawSections({
  context,
  uses,
  linked = true,
}: {
  context: ScopeContext;
  uses: CoefficientUse[];
  /** Link each law to its parameter's inspector; false inside that inspector. */
  linked?: boolean;
}) {
  return lawCurves(context.model, uses).map((curve) => {
    const label = lawLabel(curve);
    return (
      <Section
        key={curve.parameter.id}
        title={label.charAt(0).toUpperCase() + label.slice(1)}
        source={curve.kind === "fitted" ? context.model.findings.fit?.source : undefined}
      >
        <LawChart
          curve={curve}
          caption={
            linked ? (
              <OwnerLink
                onClick={() => context.select({ kind: "parameter", id: curve.parameter.id })}
              >
                {humanize(curve.parameter.name)}
              </OwnerLink>
            ) : (
              humanize(curve.parameter.name)
            )
          }
        />
        {lawHint(curve) && <Hint>{lawHint(curve)}</Hint>}
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
export function PosteriorPairs({ context, id }: { context: ScopeContext; id: ParameterId }) {
  const fit = context.model.findings.fit;
  const draws = useParameterDraws(context.model);
  const [xId, setX] = useState<string | null>(null);
  const [yId, setY] = useState<string | null>(null);
  const columns = draws.data?.columns ?? [];
  const x =
    columns.find((column) => column.subject.element_id === xId) ??
    columns.find((column) => column.subject.parameter_id === id);
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
          <EmpiricalPlot points={x.empirical} label={x.label} xLabel={x.label} />
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
