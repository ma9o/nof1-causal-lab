import type {
  DataComparisonReport,
  ObservationData,
  ObservationHistory,
  PreparedDataMetadata,
} from "@nof1-causal-lab/api-types";
import { dump } from "npyjs";
import { expect, it } from "vitest";
import { decodeFixture, fixtureValue } from "@/components/__fixtures__/fixture-value";
import rawReports from "@/components/dag/__fixtures__/simulation-reports.json";
import { dataComparisonChart } from "@/components/charts/series-adapters";
import { outcome } from "@/lib/__fixtures__/model";
import { dataComparisonView, type DataSourceResult } from "./data-comparison";

const reports = decodeFixture(rawReports);

const observation = {
  ...fixtureValue(outcome.indicators[0]).observation,
  observation_window: "1d",
};
const panel = "a".repeat(40);
const simulation = "b".repeat(40);
const emissions = { npy: new Uint8Array(dump([0, 1, 2, 2, 3, 4], [2, 3, 1], { dtype: "f8" })) };
const mask = { npy: new Uint8Array(dump([1, 1, 1, 1, 0, 1], [2, 3, 1], { dtype: "b1" })) };
const origin = "2026-01-01T00:00:00Z";
const metadata: PreparedDataMetadata = {
  source: { files: ["input/test.csv"], hashes: {}, start: null, end: null },
  preparation: {
    default_window: "1d",
    variables: [
      {
        observation,
        extraction: { kind: "semantic", how_to_measure: "Read Y", source_columns: [] },
      },
    ],
    context: "",
  },
  time_origin: origin,
  variables: [observation],
};
const history: ObservationHistory = {
  label: observation.name,
  times: [5, 6, 7],
  values: [1, 4, null],
  support_start: [4, 5, 6],
  support_end: [5, 6, 7],
  time_origin: origin,
  levels: null,
  empirical: [],
};
const envelope = {
  status: "success" as const,
  call_id: `call:${"a".repeat(64)}` as const,
  messages: [],
};
const replicate = (index: number): ObservationData => ({
  [observation.id]: {
    ...history,
    values: {
      array: emissions,
      indices: [index, null, 0],
      start: 0,
      stop: null,
      mask: {
        array: mask,
        indices: [index, null, 0],
        start: 0,
        stop: null,
        mask: null,
      },
    },
  },
});
const sources = new Map<string, DataSourceResult>([
  [
    panel,
    {
      ...envelope,
      commit_id: panel,
      action: "prepare_data",
      body: {
        data: { [observation.id]: history },
        metadata,
        profile: { indicators: {}, dataset_issues: [], is_valid: true },
      },
    },
  ],
  [
    simulation,
    {
      ...envelope,
      commit_id: simulation,
      action: "simulate",
      body: {
        report: {
          ...fixtureValue(reports[0]),
          evidence: {
            ...fixtureValue(reports[0]).evidence,
            observation_layout: {
              ...fixtureValue(reports[0]).evidence.observation_layout,
              variables: [observation],
            },
          },
        },
        data: [replicate(0), replicate(1)],

      },
    },
  ],
]);
const report: DataComparisonReport = {
  left: [{ revision: panel, replicate_index: 0 }],
  right: [1, 0].map((replicate_index) => ({ revision: simulation, replicate_index })),
  variables: [
    {
      indicator_id: observation.id,
      changes: [],
      comparison_issues: [],
      statistics: [{ statistic: "mean", level: null, left: [2.5], right: [3, 1] }],
      predictive: {
        kind: "comparison",
        reference_side: "left",
        evaluation: {
          kind: "unavailable",
          reason: "Replicas contain missing values at observed anchors",
        },
      },
    },
  ],
};

it("joins selected source histories in report order and retains computed evidence unchanged", () => {
  const view = dataComparisonView(report, sources);
  const variable = fixtureValue(view.variables[0]);
  expect(view.left).toBe(report.left);
  expect(view.right).toBe(report.right);
  expect(variable.statistics).toBe(report.variables[0]?.statistics);
  expect(variable.predictive).toBe(report.variables[0]?.predictive);
  expect(variable.left[0]?.points.map((point) => point.value)).toEqual([1, 4, null]);
  expect(variable.right.map((series) => series.points.map((point) => point.value))).toEqual([
    [2, null, 4],
    [0, 1, 2],
  ]);
  expect(variable.right[0]?.points[0]).toEqual({
    anchor_time: "2026-01-06T00:00:00.000Z",
    support_start: "2026-01-05T00:00:00.000Z",
    support_end: "2026-01-06T00:00:00.000Z",
    value: 2,
  });
  const missing = dataComparisonView(
    {
      ...report,
      variables: [{ ...fixtureValue(report.variables[0]), indicator_id: "indicator:absent" }],
    },
    sources,
  );
  expect(missing.variables[0]?.right).toEqual([
    { variable: null, time_origin: null, points: [] },
    { variable: null, time_origin: null, points: [] },
  ]);
});

it("plots exact point changes against canonical history coordinates without changing their values", () => {
  const point = { anchor_time: "2026-01-07T00:00:00Z", support_start: null, support_end: null };
  const selected: DataComparisonReport = {
    ...report,
    right: [{ revision: simulation, replicate_index: 0 }],
    variables: [
      {
        ...fixtureValue(report.variables[0]),
        changes: [
          { kind: "revised", before: { ...point, value: 4 }, after: { ...point, value: 1 } },
        ],
        predictive: {
          kind: "not_applicable",
          reason: "A predictive comparison requires replicated histories.",
        },
      },
    ],
  };
  const view = dataComparisonView(selected, sources);
  const chart = dataComparisonChart(fixtureValue(view.variables[0]));
  expect(chart.times).toEqual([0, 1, 2]);
  expect(chart.layers.find((layer) => layer.key === "revised-left")?.rows[0]?.values).toEqual([
    null,
    4,
    null,
  ]);
  expect(chart.layers.find((layer) => layer.key === "revised-right")?.rows[0]?.values).toEqual([
    null,
    1,
    null,
  ]);
});
