import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { ObservationPlots, dataComparisonHistory } from "./recorded-history";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { formatModelDate } from "@/lib/utils/format";
import { Hint, KeyValue, OwnerLink, Section, StatusIcon } from "../scope-primitives";
import type { DataDiffReport, DataVariableDiff } from "@nof1-causal-lab/api-types";
import { HistoryPlot } from "@/components/charts/history-plot";
import {
  HistogramPlot,
  PPCWarningsTable,
} from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import { COMPARISON_COLORS } from "@/lib/dag/palette";

export function DataComparisonOutcome({ context }: { context: ScopeContext }) {
  const report = context.dataDiff.data!;
  return (
    <Section title="Data comparison">
      {(["left", "right"] as const).map((side) => (
        <p key={side}>
          {side === "left" ? "Left" : "Right"}: {dataSelectionLabel(report, side, context.ticks)}
        </p>
      ))}
      {report.variables.map((variable) => {
        const definition = [...variable.left, ...variable.right].find(
          (history) => history.variable !== null,
        )!.variable!;
        const reasons = [
          ...variable.comparison_issues,
          ...(variable.predictive_checks?.per_variable_warnings
            .filter((finding) => !finding.passed)
            .map((finding) => finding.message) ?? []),
          ...(variable.predictive_unavailable_reason
            ? [variable.predictive_unavailable_reason]
            : []),
        ];
        return reasons.map((reason) => (
          <p key={`${variable.indicator_id}-${reason}`}>
            <OwnerLink
              onClick={() => context.select({ kind: "indicator", id: variable.indicator_id })}
            >
              {humanize(definition.name)}
            </OwnerLink>
            : {reason}
          </p>
        ));
      })}
    </Section>
  );
}

function dataSelectionLabel(report: DataDiffReport, side: "left" | "right", ticks: JournalTick[]) {
  const sources = report[side];
  return [...new Set(sources.map((source) => source.revision))]
    .map((revision) => {
      const histories = sources.filter((source) => source.revision === revision);
      return histories[0].kind === "panel"
        ? `Panel from ${timelineTickLabel(ticks.find((tick) => tick.produced.some((artifact) => artifact.artifact_id === "panel" && artifact.revision === revision))!)} (${report.variables.filter((variable) => variable[side][sources.indexOf(histories[0])].variable !== null).length} variables)`
        : `${histories.length} replicates from simulate ${revision.slice(0, 7)}`;
    })
    .join("; ");
}

export function DataComparisonEvidence({
  context,
  selection,
}: {
  context: ScopeContext;
  selection: EntitySelection | null;
}) {
  const variables = context.dataDiff.data!.variables.filter(
    (variable) =>
      selection === null ||
      (selection.kind === "indicator" && selection.id === variable.indicator_id) ||
      (selection.kind === "construct" &&
        context.entities.indicatorOwnerById.get(variable.indicator_id)?.id === selection.id),
  );
  if (variables.length === 0) return <Hint>No observation comparison for this part.</Hint>;
  return variables.map((variable) => (
    <VariableComparison key={variable.indicator_id} variable={variable} />
  ));
}

function VariableComparison({ variable }: { variable: DataVariableDiff }) {
  const definition = [...variable.left, ...variable.right].find(
    (history) => history.variable !== null,
  )!.variable!;
  const checks = variable.predictive_checks;
  return (
    <Section title={humanize(definition.name)} wide>
      <HistoryPlot
        {...dataComparisonHistory(variable)}
        description="Every saved history. Observations are dark; the reference sits above all replicates. Changed points use green (added), red (removed) and amber (revised); left values are hollow."
      />
      {variable.comparison_issues.map((reason) => (
        <Hint key={reason} issue>
          {reason}
        </Hint>
      ))}
      {variable.predictive_unavailable_reason && (
        <Hint>{variable.predictive_unavailable_reason}</Hint>
      )}
      {checks && (
        <PPCWarningsTable
          indicators={[definition]}
          warnings={checks.per_variable_warnings}
          testStats={checks.test_stats}
          overlays={checks.overlays}
        />
      )}
      <table
        className="w-full text-left text-[10px]"
        aria-label={`${definition.name}: statistic distributions`}
      >
        <thead>
          <tr>
            <th>Statistic</th>
            <th>Left histories</th>
            <th>Right histories</th>
          </tr>
        </thead>
        <tbody>
          {variable.statistics.map((statistic) => (
            <tr key={`${statistic.statistic}-${statistic.level}`}>
              <th className="font-normal">
                {humanize(statistic.statistic)} {statistic.level}
              </th>
              <td>
                <HistogramPlot histogram={statistic.left_histogram} />
              </td>
              <td>
                <HistogramPlot histogram={statistic.right_histogram} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {variable.changes.length > 0 && (
        <table
          className="w-full text-left text-[10px]"
          aria-label={`${definition.name}: point changes`}
        >
          <thead>
            <tr>
              <th>Anchor / change</th>
              <th>Left</th>
              <th>Right</th>
            </tr>
          </thead>
          <tbody>
            {variable.changes.map((change) => (
              <tr key={change.anchor_time}>
                <th className="font-normal" style={{ color: COMPARISON_COLORS[change.change] }}>
                  <time dateTime={change.anchor_time} title={change.anchor_time}>
                    {formatModelDate(0, change.anchor_time)}
                  </time>
                  <br />
                  {change.change}
                </th>
                {(["left", "right"] as const).map((side) => (
                  <td key={side} className="break-all align-top">
                    {change[side] ? (
                      <>
                        {change[side].value ?? "Missing"}
                        {change[side].support_start && (
                          <>
                            <br />
                            {formatModelDate(0, change[side].support_start)} —{" "}
                            {formatModelDate(0, change[side].support_end!)}
                          </>
                        )}
                      </>
                    ) : (
                      "Absent"
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Section>
  );
}

export function PreparedObservations({ context }: { context: ScopeContext }) {
  const { metadata, profile } = context.model.data;
  if (!metadata) return <Hint>No observation panel recorded.</Hint>;
  return (
    <>
      {metadata.value.variables.map((variable) => (
        <Section key={variable.id} title={humanize(variable.name)} source={metadata.source} wide>
          <ObservationPlots model={context.model} id={variable.id} />
        </Section>
      ))}
      {profile && profile.value.dataset_issues.length > 0 && (
        <Section title="Dataset issues" source={profile.source}>
          {profile.value.dataset_issues.map((issue) => (
            <div key={issue.issue_type + issue.message} className="flex gap-2">
              {issue.severity !== "info" && (
                <StatusIcon status={issue.severity === "error" ? "failed" : "warning"} />
              )}
              <Hint issue={issue.severity !== "info"}>{issue.message}</Hint>
            </div>
          ))}
        </Section>
      )}
    </>
  );
}

export function DataDetails({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const metadata = context.model.data.metadata;
  const raw = context.model.data.raw_data;
  if (!metadata)
    return (
      <Section title="Prepared data" source={raw?.source}>
        {raw && tick.produced.some((artifact) => artifact.artifact_id === "raw_data") ? (
          <KeyValue
            rows={[
              ["Imported records", raw.value.n_records.toLocaleString()],
              ["Columns", String(raw.value.n_columns)],
              ["First observation", raw.value.date_range.start],
              ["Last observation", raw.value.date_range.end],
            ]}
          />
        ) : (
          <Hint>No observation panel was produced.</Hint>
        )}
      </Section>
    );
  const { source, variables } = metadata.value;
  return (
    <Section title="Prepared data" source={metadata.source}>
      <KeyValue
        rows={[
          ...("files" in source
            ? ([
                ["Files", source.files.join(", ")],
                ["Start (inclusive)", source.start ?? "Unbounded"],
                ["End (exclusive)", source.end ?? "Unbounded"],
              ] as Array<[string, string]>)
            : ([
                ["Source", "Saved simulation"],
                ["Replicate", String(source.replicate)],
              ] as Array<[string, string]>)),
          ["Variables", String(variables.length)],
          ["Observations", context.model.data.measurements?.value.n_observations.toLocaleString()],
        ]}
      />
    </Section>
  );
}
