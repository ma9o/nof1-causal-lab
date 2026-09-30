import type { ValidationIssue } from "@nof1-causal-lab/api-types";
import { QuantileStrip } from "@/components/charts/quantile-strip";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatFillNull, humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { ObservationPlots } from "./recorded-history";

function DataIssues({ issues }: { issues: ValidationIssue[] }) {
  return issues.map((issue) => (
    <div key={`${issue.issue_type}-${issue.message}`} className="flex items-start gap-2">
      {issue.severity !== "info" && (
        <StatusIcon status={issue.severity === "error" ? "failed" : "warning"} />
      )}
      <Hint issue={issue.severity !== "info"}>{issue.message}</Hint>
    </div>
  ));
}

export function DataDetails({ context }: { context: ScopeContext }) {
  const metadata = context.model.data.metadata;
  const profile = context.model.data.profile;
  if (!metadata)
    return (
      <Section title="Prepared data">
        <Hint>No panel metadata is recorded at this version.</Hint>
      </Section>
    );
  const { source, variables } = metadata.value;
  const sourceRows: Array<[string, string]> =
    "files" in source
      ? [
          ["Files", source.files.join(", ")],
          ["Start (inclusive)", source.start ?? "Unbounded"],
          ["End (exclusive)", source.end ?? "Unbounded"],
        ]
      : [
          ["Simulation", source.revision],
          ["Replicate", String(source.replicate)],
        ];
  return (
    <>
      <Section title="Prepared data" source={metadata.source}>
        <KeyValue
          rows={[
            ...sourceRows,
            ["Variables", String(variables.length)],
            ["Model day zero", metadata.value.time_origin ?? "Calendar-free"],
          ]}
        />
        <DataIssues issues={profile?.value.dataset_issues ?? []} />
      </Section>
      {variables.map((variable) => {
        const audit = profile?.value.indicators[variable.id];
        const empirical = audit?.profile;
        const levels = variable.ordinal_levels ?? variable.categorical_levels;
        return (
          <Section key={variable.id} title={humanize(variable.name)} source={profile?.source} wide>
            <ObservationPlots model={context.model} id={variable.id} />
            <details>
              <summary className="cursor-pointer text-muted-foreground">
                Preparation and numerical summary
              </summary>
              <KeyValue
                rows={[
                  ["Type", variable.measurement_dtype],
                  ["Aggregation", variable.aggregation],
                  ["Window", variable.observation_window ?? "Not recorded"],
                  ["Null filling", formatFillNull(variable)],
                  ...(levels?.length ? [["Levels", levels.join(" · ")] as [string, string]] : []),
                  ["Observations", empirical?.n_obs.toLocaleString() ?? "Not recorded"],
                  [
                    "Time coverage",
                    empirical?.time_coverage_ratio != null
                      ? new Intl.NumberFormat(undefined, {
                          style: "percent",
                          maximumFractionDigits: 1,
                        }).format(empirical.time_coverage_ratio)
                      : "Not recorded",
                  ],
                ]}
              />
              {empirical && <QuantileStrip profile={empirical} />}
            </details>
            <DataIssues issues={audit?.issues ?? []} />
          </Section>
        );
      })}
    </>
  );
}
