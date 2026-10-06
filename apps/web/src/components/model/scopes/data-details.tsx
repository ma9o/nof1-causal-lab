import type { PrepareDataOutput } from "@nof1-causal-lab/api-types";
import type { TimelineRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { ObservationPlots } from "./recorded-history";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { formatModelDate } from "@/lib/utils/format";
import { Hint, KeyValue, OwnerLink, Section, StatusIcon } from "../scope-primitives";
import type { DataDiffOutput, DataVariableDiff } from "@nof1-causal-lab/api-types";
import { PPCWarningsTable } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import { ChartFigure } from "@/components/charts/chart-figure";
import { CHART_COLORS, chainColor, cssColor } from "@/components/charts/chart-tokens";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { DrawsChart } from "@/components/charts/draws-chart";
import { extentOf } from "@/components/charts/plot-geometry";
import { dataComparisonChart } from "@/components/charts/series-adapters";

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
  report: DataDiffOutput;
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
  report: DataDiffOutput,
  side: "left" | "right",
  ticks: readonly TimelineRevision[],
) {
  const sources = report[side];
  return [...new Map(sources.map((source) => [source.revision, source])).values()]
    .map((source) => {
      if (
        ticks.some(
          (tick) => tick.commit_id === source.revision && tick.record.attempt.action === "simulate",
        )
      ) {
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
  report: DataDiffOutput;
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
  const chart = dataComparisonChart(variable);
  const span = extentOf(chart.times);
  return (
    <Section title={humanize(definition?.name ?? variable.indicator_id)} wide>
      <ChartFigure
        title="Every saved history"
        height={170}
        note="Observations are dark and sit above every replicate. Changed points use green (added), red (removed) and amber (revised)."
        {...(span ? { timeline: { domain: span, origin: chart.timeOrigin } } : {})}
      >
        {(view) => <DrawsChart {...chart} height={view.height} timeWindow={view.timeWindow} />}
      </ChartFigure>
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
          {variable.statistics.map((statistic) => {
            const left = statistic.left.flatMap((value) => (value === null ? [] : [value]));
            const right = statistic.right.flatMap((value) => (value === null ? [] : [value]));
            const frame = extentOf([...left, ...right]);
            const name = `${humanize(statistic.statistic)} ${statistic.level}`;
            return (
              <tr key={`${statistic.statistic}-${statistic.level}`}>
                <th className="font-normal">{name}</th>
                {(
                  [
                    ["left", left, 0],
                    ["right", right, 1],
                  ] as const
                ).map(([side, values, color]) => (
                  <td key={side} className="min-w-24 py-1">
                    <DistributionChart
                      compact
                      height={26}
                      label={`${name}: every ${side} history's value`}
                      dots={[{ key: side, label: side, color: chainColor(color), values }]}
                      frame={frame}
                    />
                  </td>
                ))}
              </tr>
            );
          })}
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
                <th className="font-normal" style={{ color: cssColor(CHART_COLORS[change.kind]) }}>
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
      {metadata.variables.map((variable) => (
        <Section key={variable.id} title={humanize(variable.name)} wide>
          <ObservationPlots model={context.model} id={variable.id} />
        </Section>
      ))}
      {profile && profile.dataset_issues.length > 0 && (
        <Section title="Dataset issues">
          {profile.dataset_issues.map((issue) => (
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
  applied: PrepareDataOutput;
}) {
  const metadata = applied.metadata;
  const raw = applied.raw_data;
  if (!metadata)
    return (
      <Section title="Prepared data">
        {raw ? (
          <KeyValue
            rows={[
              ["Imported records", raw.n_records.toLocaleString()],
              ["Columns", String(raw.n_columns)],
              ["First observation", raw.date_range?.start ?? "Not recorded"],
              ["Last observation", raw.date_range?.end ?? "Not recorded"],
            ]}
          />
        ) : (
          <Hint>No observation panel was produced.</Hint>
        )}
      </Section>
    );
  const { source, variables } = metadata;
  return (
    <Section title="Prepared data">
      <KeyValue
        rows={[
          ...([
            ["Files", source.files.join(", ")],
            ["Start (inclusive)", source.start ?? "Unbounded"],
            ["End (exclusive)", source.end ?? "Unbounded"],
          ] as Array<[string, string]>),
          ["Variables", String(variables.length)],
          ["Observations", context.model.measurements?.n_observations.toLocaleString()],
        ]}
      />
    </Section>
  );
}
