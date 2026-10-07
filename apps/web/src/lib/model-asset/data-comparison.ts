import type {
  ActionSuccess,
  DataComparisonReport,
  DataPoint,
  DataVariableComparison,
  GitOid,
  PrepareDataOutput,
} from "@nof1-causal-lab/api-types";
import { historyView } from "./result-values";

export type DataSourceResult = Extract<ActionSuccess, { action: "prepare_data" | "simulate" }>;

interface DataSeriesView {
  readonly variable: PrepareDataOutput["metadata"]["variables"][number] | null;
  readonly time_origin: string | null;
  readonly points: readonly DataPoint[];
}

export type DataVariableView = DataVariableComparison & {
  readonly left: readonly DataSeriesView[];
  readonly right: readonly DataSeriesView[];
};

export type DataComparisonView = Omit<DataComparisonReport, "variables"> & {
  readonly variables: readonly DataVariableView[];
};

/** Attach original histories for display; all comparisons and statistics stay in the saved report. */
export function dataComparisonView(
  report: DataComparisonReport,
  sources: ReadonlyMap<GitOid, DataSourceResult>,
): DataComparisonView {
  const side = (name: "left" | "right", id: DataVariableComparison["indicator_id"]) =>
    report[name].map((source): DataSeriesView => {
      const result = coordinate(sources.get(source.revision));
      const data =
        result.action === "prepare_data"
          ? result.body.data
          : coordinate(result.body.data[source.replicate_index]);
      const recorded = data[id];
      if (recorded === undefined) return { variable: null, time_origin: null, points: [] };
      const variables =
        result.action === "prepare_data"
          ? result.body.metadata.variables
          : result.body.report.evidence.observation_layout.variables;
      const history = historyView(recorded);
      const origin = history.time_origin === null ? 0 : Date.parse(history.time_origin);
      const instant = (day: number) => new Date(origin + day * 86_400_000).toISOString();
      const support = (day: number | null) => (day === null ? null : instant(day));
      return {
        variable: coordinate(variables.find((variable) => variable.id === id)),
        time_origin: history.time_origin,
        points: history.times.map((time, index) => ({
          anchor_time: instant(time),
          support_start: support(coordinate(history.support_start[index])),
          support_end: support(coordinate(history.support_end[index])),
          value: coordinate(history.values[index]),
        })),
      };
    });
  return {
    ...report,
    variables: report.variables.map((variable) => ({
      ...variable,
      left: side("left", variable.indicator_id),
      right: side("right", variable.indicator_id),
    })),
  };
}

function coordinate<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("Saved comparison source has a missing coordinate");
  return value;
}
