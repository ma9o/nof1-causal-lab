import { presentEntries } from "@/lib/model-accessors";
import type { IndicatorId } from "@nof1-causal-lab/api-types";
import { indicatorPresentation, dispositionLabel } from "@/lib/model-asset/inspector";
import { ownLawUses } from "@/lib/model-asset/laws";
import { formatFillNull, humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { LawSections, SimulatedHistory } from "./law-sections";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { ObservationPlots, ObservedProfile } from "./recorded-history";

const CHECK_STATUS = {
  ok: "passed",
  warning: "warning",
  error: "failed",
  not_evaluated: "not_evaluated",
} as const;

export function IndicatorScope({ context, id }: { context: ScopeContext; id: IndicatorId }) {
  const scope = indicatorPresentation(context, id);
  if (!scope) return null;
  const { indicator, disposition, audit, counts, likelihood, issues } = scope;
  const metadata = context.model.metadata;
  const preparation = metadata?.preparation.variables.find(
    (variable) => variable.observation.id === id,
  );
  const equation = context.model.observation_equations[id];
  const empirical = context.model.profile?.indicators[id]?.profile;
  return (
    <>
      <Section title="Measurement">
        <details>
          <summary className="cursor-pointer text-muted-foreground">Measurement settings</summary>
          <div className="mt-2">
            <KeyValue
              rows={[
                ["Type", indicator.observation.measurement_dtype],
                ["Aggregation", indicator.observation.aggregation],
                [
                  "Window",
                  indicator.observation.observation_window ??
                    context.model.model?.measurement_clock,
                ],
                ["Polarity", indicator.construct_polarity],
                ...(likelihood
                  ? ([
                      ["Likelihood", likelihood.law.distribution],
                      ["Standardized", likelihood.standardized ? "Yes" : "No"],
                    ] as Array<[string, string]>)
                  : []),
              ]}
            />
          </div>
        </details>
      </Section>
      {equation && (
        <Section title="Observation equation" wide>
          <Katex latex={equation} />
        </Section>
      )}
      <LawSections context={context} uses={ownLawUses(indicator)} />
      {preparation && (
        <Section title="Data preparation">
          <Hint>{preparation.extraction.how_to_measure}</Hint>
          <KeyValue
            rows={[
              ["Extraction", preparation.extraction.kind],
              ["Source columns", preparation.extraction.source_columns.join(", ")],
              ...(preparation.extraction.kind === "computed"
                ? [["Null filling", formatFillNull(preparation.extraction)] as [string, string]]
                : []),
            ]}
          />
        </Section>
      )}
      {disposition && disposition.disposition !== "manifest" && (
        <Section title={dispositionLabel(disposition.disposition)}>
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
      {(counts != null || empirical) && (
        <Section title="Observations" wide>
          {counts != null && <Hint>{counts.toLocaleString()} observations</Hint>}
          <ObservationPlots model={context.model} id={id} />
          {empirical && (
            <>
              <ObservedProfile model={context.model} id={id} profile={empirical} />
              <Hint>
                Every prepared value as a dot over the interquartile box; the bar is the median and
                the diamond the mean.
              </Hint>
            </>
          )}
        </Section>
      )}
      {audit && (
        <Section title="Validation">
          {issues.map((issue) => (
            <div key={`${issue.issue_type}-${issue.message}`} className="flex items-start gap-2">
              <StatusIcon status={issue.severity === "error" ? "failed" : "warning"} />
              <Hint issue>{issue.message}</Hint>
            </div>
          ))}
          <details>
            <summary className="cursor-pointer text-muted-foreground">All checks</summary>
            <ul className="mt-2 space-y-2">
              {presentEntries(audit.checks).map(([check, status]) => (
                <li key={check} className="flex items-center gap-2">
                  <StatusIcon status={CHECK_STATUS[status]} />
                  <span>{humanize(check)}</span>
                </li>
              ))}
            </ul>
          </details>
        </Section>
      )}
      <SimulatedHistory context={context} id={id} kind="indicators" />
    </>
  );
}
