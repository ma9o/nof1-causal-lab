import type { Applied, DataPreparationResult } from "@nof1-causal-lab/api-types";
import type { TimelineRevision } from "@nof1-causal-lab/api-types";
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

function comparisonEvaluation(variable: DataVariableDiff) {
  return variable.predictive.kind === "comparison"
    ? variable.predictive.evaluation
    : variable.predictive;
}

export function DataComparisonOutcome({
  context,
  report,
}: {
  context: ScopeContext;
  report: DataDiffReport;
}) {
  return (
    <Section title="Data comparison">
      {(["left", "right"] as const).map((side) => (
        <p key={side}>
          {side === "left" ? "Left" : "Right"}: {dataSelectionLabel(report, side, context.ticks)}
        </p>
      ))}
      {report.variables.map((variable) => {
        const evaluation = comparisonEvaluation(variable);
        const checks = evaluation.kind === "available" ? evaluation.value : null;
        const definition = [...variable.left, ...variable.right]
          .flatMap((history) => history.variable ?? [])
          .at(0);
        const reasons = [
          ...variable.comparison_issues,
          ...(checks?.per_variable_warnings
            .filter((finding) => finding.kind !== "evaluated" || finding.outcome !== "passed")
            .map((finding) =>
              finding.kind === "evaluated" ? finding.evidence.note : finding.detail,
            ) ?? []),
          ...(evaluation.kind === "unavailable" ? [evaluation.reason] : []),
        ];
        return reasons.map((reason) => (
          <p key={`${variable.indicator_id}-${reason}`}>
            <OwnerLink
              onClick={() => context.select({ kind: "indicator", id: variable.indicator_id })}
            >
              {humanize(definition?.name ?? variable.indicator_id)}
            </OwnerLink>
            : {reason}
          </p>
        ));
      })}
    </Section>
  );
}

function dataSelectionLabel(
  report: DataDiffReport,
  side: "left" | "right",
  ticks: readonly TimelineRevision[],
) {
  const sources = report[side];
  return [...new Map(sources.map((source) => [source.revision, source])).values()]
    .map((source) => {
      if (source.kind === "simulation") {
        const count = sources.filter((item) => item.revision === source.revision).length;
        return `${count} replicates from simulate ${source.revision.slice(0, 7)}`;
      }
      const tick = ticks.find(
        (tick) =>
          tick.record.attempt.outcome.status === "applied" &&
          tick.record.attempt.outcome.effects.produced.some(
            (artifact) => artifact.artifact_id === "panel" && artifact.revision === source.revision,
          ),
      );
      const variables = report.variables.filter((variable) =>
        variable[side].some(
          (history, index) =>
            history.variable !== null && sources[index]?.revision === source.revision,
        ),
      ).length;
      return `Panel from ${tick ? timelineTickLabel(tick) : source.revision.slice(0, 7)} (${variables} variables)`;
    })
    .join("; ");
}

export function DataComparisonEvidence({
  context,
  report,
  selection,
}: {
  context: ScopeContext;
  report: DataDiffReport;
  selection: EntitySelection | null;
}) {
  const variables = report.variables.filter(
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
  const definition = [...variable.left, ...variable.right]
    .flatMap((history) => history.variable ?? [])
    .at(0);
  const evaluation = comparisonEvaluation(variable);
  const checks = evaluation.kind === "available" ? evaluation.value : null;
  return (
    <Section title={humanize(definition?.name ?? variable.indicator_id)} wide>
      <HistoryPlot
        {...dataComparisonHistory(variable)}
        description="Every saved history. Observations are dark; the reference sits above all replicates. Changed points use green (added), red (removed) and amber (revised); left values are hollow."
      />
      {variable.comparison_issues.map((reason) => (
        <Hint key={reason} issue>
          {reason}
        </Hint>
      ))}
      {evaluation.kind === "unavailable" && <Hint>{evaluation.reason}</Hint>}
      {checks && (
        <PPCWarningsTable
          indicators={definition ? [definition] : []}
          warnings={checks.per_variable_warnings}
          testStats={checks.test_stats}
          overlays={checks.overlays}
        />
      )}
      <table
        className="w-full text-left text-[10px]"
        aria-label={`${definition?.name ?? variable.indicator_id}: statistic distributions`}
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
          aria-label={`${definition?.name ?? variable.indicator_id}: point changes`}
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
              <tr
                key={
                  change.kind === "removed" ? change.before.anchor_time : change.after.anchor_time
                }
              >
                <th className="font-normal" style={{ color: COMPARISON_COLORS[change.kind] }}>
                  <time
                    dateTime={
                      change.kind === "removed"
                        ? change.before.anchor_time
                        : change.after.anchor_time
                    }
                    title={
                      change.kind === "removed"
                        ? change.before.anchor_time
                        : change.after.anchor_time
                    }
                  >
                    {formatModelDate(
                      0,
                      change.kind === "removed"
                        ? change.before.anchor_time
                        : change.after.anchor_time,
                    )}
                  </time>
                  <br />
                  {change.kind}
                </th>
                {(
                  [
                    ["left", change.kind === "added" ? null : change.before],
                    ["right", change.kind === "removed" ? null : change.after],
                  ] as const
                ).map(([side, point]) => (
                  <td key={side} className="break-all align-top">
                    {point ? (
                      <>
                        {point.value ?? "Missing"}
                        {point.support_start && point.support_end && (
                          <>
                            <br />
                            {formatModelDate(0, point.support_start)} —{" "}
                            {formatModelDate(0, point.support_end)}
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
  const { metadata, profile } = context.model;
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

export function DataDetails({
  context,
  applied,
}: {
  context: ScopeContext;
  applied: Applied<DataPreparationResult>;
}) {
  const metadata = context.model.metadata;
  const raw = context.model.raw_data;
  if (!metadata)
    return (
      <Section title="Prepared data" {...(raw?.source === undefined ? {} : { source: raw.source })}>
        {raw && applied.effects.produced.some((artifact) => artifact.artifact_id === "raw_data") ? (
          <KeyValue
            rows={[
              ["Imported records", raw.value.n_records.toLocaleString()],
              ["Columns", String(raw.value.n_columns)],
              ["First observation", raw.value.date_range?.start ?? "Not recorded"],
              ["Last observation", raw.value.date_range?.end ?? "Not recorded"],
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
          ["Observations", context.model.measurements?.value.n_observations.toLocaleString()],
        ]}
      />
    </Section>
  );
}
