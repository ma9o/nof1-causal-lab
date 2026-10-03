import { presentEntries } from "@/lib/model-accessors";
import type { IndicatorId } from "@nof1-causal-lab/api-types";
import { TestStatSparkline } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import { QuantileStrip } from "@/components/charts/quantile-strip";
import { indicatorPresentation, dispositionLabel } from "@/lib/model-asset/inspector";
import { ownLawUses } from "@/lib/model-asset/laws";
import { formatFillNull, humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatSignificant } from "@/lib/utils/format";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { LawSections, SimulatedHistory } from "./law-sections";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { ObservationPlots, PredictiveHistoryPlot } from "./recorded-history";
import { PredictiveFindings } from "../simulation-evidence";

const CHECK_STATUS = {
  ok: "passed",
  warning: "warning",
  error: "failed",
  not_evaluated: "not_evaluated",
} as const;

export function IndicatorScope({ context, id }: { context: ScopeContext; id: IndicatorId }) {
  const scope = indicatorPresentation(context, id);
  if (!scope) return null;
  const { indicator, disposition, audit, counts, likelihood, predictive, checks, issues } = scope;
  const preparation = context.model.metadata?.value.preparation?.variables.find(
    (variable) => variable.observation.id === id,
  );
  const equation = context.model.observation_equations[id];
  const empirical = context.model.profile?.value.indicators[id]?.profile;
  const comparison =
    predictive?.source.validity === "fresh" && predictive.value.evaluation.kind === "evaluated"
      ? predictive.value.evaluation.predictive_checks
      : null;
  const overlay = comparison?.overlays.find((item) => item.indicator_id === id);
  const statistics = comparison?.test_stats.filter((item) => item.indicator_id === id) ?? [];
  const findings = (
    predictive?.value.evaluation.kind === "evaluated" ? predictive.value.evaluation.findings : []
  ).filter(
    (finding) => typeof finding.subject.target !== "string" && finding.subject.target.id === id,
  );
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
                    context.model.model?.value.measurement_clock,
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
        <Section
          title="Data preparation"
          {...(context.model.metadata?.source === undefined
            ? {}
            : { source: context.model.metadata.source })}
        >
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
        <Section
          title={dispositionLabel(disposition.disposition)}
          {...(context.model.dispositions?.source === undefined
            ? {}
            : { source: context.model.dispositions.source })}
        >
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
      {(counts != null || empirical) && (
        <Section
          title="Observations"
          {...(context.model.measurements?.source === undefined
            ? {}
            : { source: context.model.measurements.source })}
          wide
        >
          {counts != null && <Hint>{counts.toLocaleString()} observations</Hint>}
          <ObservationPlots model={context.model} id={id} />
          {empirical && (
            <details>
              <summary className="cursor-pointer text-muted-foreground">Numerical summary</summary>
              <QuantileStrip profile={empirical} />
              <Hint>
                Range, quartiles and median of the prepared values; the dot is their mean.
              </Hint>
            </details>
          )}
        </Section>
      )}
      {audit && (
        <Section
          title="Validation"
          {...(context.model.validation_report?.source === undefined
            ? {}
            : { source: context.model.validation_report.source })}
        >
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
      {(checks.length > 0 || findings.length > 0 || overlay) && (
        <Section
          title="Predictive checks"
          {...(predictive?.source === undefined ? {} : { source: predictive.source })}
          wide
        >
          <PredictiveFindings findings={findings} entities={context.entities} />
          {checks.map((check) => (
            <div key={check.subject.check} className="flex items-center gap-2">
              <StatusIcon status={check.kind === "evaluated" ? check.outcome : "not_evaluated"} />
              <span>
                {humanize(check.subject.check)}:{" "}
                {check.kind === "evaluated" ? check.evidence.value.toFixed(2) : "Not evaluated"}
                <Hint issue={check.kind === "evaluated" && check.outcome !== "passed"}>
                  {check.kind === "evaluated" ? check.evidence.note : check.detail}
                </Hint>
              </span>
            </div>
          ))}
          {overlay && (
            <>
              <PredictiveHistoryPlot model={context.model} id={id} />
            </>
          )}
          {statistics.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {statistics.map((statistic) => (
                <div key={statistic.stat_name} className="min-w-0">
                  <Hint>
                    T({statistic.stat_name}) = {formatSignificant(statistic.observed_value)}
                  </Hint>
                  <TestStatSparkline stat={statistic} />
                </div>
              ))}
            </div>
          )}
        </Section>
      )}
      <SimulatedHistory context={context} id={id} kind="indicators" />
    </>
  );
}
