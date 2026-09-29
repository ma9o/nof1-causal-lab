import type { IndicatorId } from "@nof1-causal-lab/api-types";
import { indicatorPresentation, dispositionLabel } from "@/lib/model-asset/inspector";
import { formatFillNull, humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, KeyValue, ParameterLinks, Section, StatusIcon } from "../scope-primitives";
import { parametersForOwner } from "./parameters";

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
  const preparation = context.model.data.metadata?.value.preparation?.variables.find(
    (variable) => variable.id === id,
  );
  const parameters = parametersForOwner(context.model.model?.value, id);
  return (
    <>
      <Section title="Measurement">
        <details>
          <summary className="cursor-pointer text-muted-foreground">Measurement settings</summary>
          <div className="mt-2">
            <KeyValue
              rows={[
                ["Type", indicator.measurement_dtype],
                ["Aggregation", indicator.aggregation],
                ["Null filling", formatFillNull(indicator)],
                [
                  "Window",
                  indicator.observation_window ?? context.model.model?.value.measurement_clock,
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
      {preparation && (
        <Section title="Data preparation" source={context.model.data.metadata?.source}>
          <Hint>{preparation.how_to_measure}</Hint>
          <KeyValue
            rows={[
              ["Null filling", formatFillNull(preparation)],
              ["Extraction", preparation.extraction_mode],
              ["Source columns", preparation.source_columns.join(", ")],
            ]}
          />
        </Section>
      )}
      {disposition && disposition.disposition !== "manifest" && (
        <Section
          title={dispositionLabel(disposition.disposition)}
          source={context.model.findings.dispositions?.source}
        >
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
      {counts != null && (
        <Section title="Observations" source={context.model.data.measurements?.source}>
          <Hint>{counts.toLocaleString()} observations</Hint>
        </Section>
      )}
      {audit && (
        <Section title="Validation" source={context.model.findings.validation_report?.source}>
          {issues.map((issue) => (
            <div key={`${issue.issue_type}-${issue.message}`} className="flex items-start gap-2">
              <StatusIcon status={issue.severity === "error" ? "failed" : "warning"} />
              <Hint issue>{issue.message}</Hint>
            </div>
          ))}
          <details>
            <summary className="cursor-pointer text-muted-foreground">All checks</summary>
            <ul className="mt-2 space-y-2">
              {Object.entries(audit.checks).map(([check, status]) => (
                <li key={check} className="flex items-center gap-2">
                  <StatusIcon status={CHECK_STATUS[status]} />
                  <span>{humanize(check)}</span>
                </li>
              ))}
            </ul>
          </details>
        </Section>
      )}
      {checks.length > 0 && (
        <Section title="Predictive checks" source={predictive?.source}>
          {checks.map((check) => (
            <div key={check.check_type} className="flex items-center gap-2">
              <StatusIcon status={check.passed ? "passed" : "failed"} />
              <span>
                {humanize(check.check_type)}: {check.value.toFixed(2)}
              </span>
            </div>
          ))}
        </Section>
      )}
      {parameters.length > 0 && (
        <Section title="Parameters">
          <ParameterLinks parameters={parameters} onSelect={context.select} />
        </Section>
      )}
    </>
  );
}
