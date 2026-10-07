import type {
  ActionSuccess,
  DataComparisonReport,
  DataPoint,
  GitOid,
  PrepareDataOutput,
} from "@nof1-causal-lab/api-types";
import { historyView } from "./result-values";

export type DataSourceResult = Extract<ActionSuccess, { action: "prepare_data" | "simulate" }>;

interface DataSeriesView {
  readonly variable: PrepareDataOutput["metadata"]["variables"][number];
  readonly time_origin: string;
  readonly points: readonly DataPoint[];
}

export type DataVariableView = DataComparisonReport["variables"][number] & {
  readonly left: readonly (DataSeriesView | null)[];
  readonly right: readonly (DataSeriesView | null)[];
};

export type DataComparisonView = Omit<DataComparisonReport, "variables"> & {
  readonly variables: readonly DataVariableView[];
};

/** Attach original histories for display; all comparisons and statistics stay in the saved report. */
export function dataComparisonView(
  report: DataComparisonReport,
  sources: ReadonlyMap<GitOid, DataSourceResult>,
): DataComparisonView {
  const side = (name: "left" | "right", id: DataVariableView["indicator_id"]) =>
    report[name].map((source): DataSeriesView | null => {
      const result = coordinate(sources.get(source.revision));
      const data =
        result.action === "prepare_data"
          ? result.body.data
          : coordinate(result.body.data[source.replicate_index]);
      const recorded = data[id];
      if (recorded === undefined) return null;
      const variables =
        result.action === "prepare_data"
          ? result.body.metadata.variables
          : result.body.report.evidence.observation_layout.variables;
      const history = historyView(recorded);
      const timeOrigin =
        result.action === "prepare_data"
          ? result.body.metadata.time_origin
          : result.body.report.evidence.time_origin;
      const origin = Date.parse(timeOrigin);
      const instant = (day: number) => new Date(origin + day * 86_400_000).toISOString();
      const support = (day: number | null) => (day === null ? null : instant(day));
      return {
        variable: coordinate(variables.find((variable) => variable.id === id)),
        time_origin: timeOrigin,
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
