import type { paths } from "@nof1-causal-lab/api-types/src/generated/model-api";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type {
  ActionAttempt,
  ArtifactRecord,
  DataDiffReport,
  ModelDiffReport,
  ModelSnapshot,
  ModelSpec,
  SimulationReport,
  PathSeries,
  RecordDependency,
  SpecificationAssessment,
  StudyRevision,
  StudyStatus,
  TimelineResponse,
} from "@nof1-causal-lab/api-types";
import { delay, HttpResponse, http } from "msw";
import { presentEntries, modelConstructs } from "@/lib/model-accessors";
import { demoModelSnapshot, demoSnapshotAt } from "./demo-artifacts";
import { demoTraces } from "./demo-traces";
import { posterior } from "./inference-data";
import comparisonFixture from "./workbench-comparisons.json";
import visualFixture from "./workbench-visuals.json";
export const WORKBENCH_WORKSPACE = "STORYBOOK";
const stamp = "2026-09-16T12:00:00Z";
// Illustrative interface data. Branch metadata and parameter decisions are staged;
// retained DEMO evidence is reused for presentation, not claimed as new inference.
const freeModel = structuredClone(fixtureValue(demoModelSnapshot.model).value);
// Generated and validated by scripts/fixtures/study.py.
const pinnedModel = comparisonFixture.pinned_model;
const freeInputs = fixtureValue(demoModelSnapshot.state.current.model).model_inputs;
const pinnedInputs = comparisonFixture.pinned_inputs;
const definitionComparisons = comparisonFixture.comparisons;
const checks: SpecificationAssessment[] = [
  {
    kind: "not_evaluated",
    subject: "specification",
    reason: "MODEL_INCOMPLETE",
    detail: "Specification checks have not been run.",
  },
];
const commitId = (seq: number) => (0xc000000 + seq).toString(16).padEnd(40, "c");
const modelId = (ordinal: number): string =>
  ordinal <= 4
    ? fixtureValue(demoSnapshotAt(fixtureValue([2, 3, 4, 7][ordinal - 1])).state.current.model)
        .revision
    : ordinal.toString(16).padStart(40, "a");
const panelId = fixtureValue(demoModelSnapshot.state.current.panel).revision;
const modelRef = (revision: string) => ({
  workspace_id: WORKBENCH_WORKSPACE,
  revision,
  path: "model.json",
});
const logRef = (seq: number) => ({
  workspace_id: WORKBENCH_WORKSPACE,
  revision: commitId(seq),
  path: "logs/attempt.json",
});
function record(seq: number, attempt: ActionAttempt, trace_ids: string[] = []): StudyRevision {
  return {
    commit_id: commitId(seq),
    parent_ids: [commitId(seq - 1)],
    record: {
      seq,
      attempt_id: null,
      branch: "main",
      ts: stamp,
      attempt,
      trace_ids,
      messages: [{ timestamp: stamp, level: "info", label: "ACTION_COMPLETED" }],
    },
  };
}
function metadata(
  revision: string,
  parent: string,
  produced_by: string,
  model_inputs: ArtifactRecord["model_inputs"],
): ArtifactRecord {
  return {
    artifact_id: "model",
    revision,
    derived_from: produced_by === "fit" ? { model: parent, panel: panelId } : { model: parent },
    model_inputs,
    consumed_model_inputs: {},
    produced_by,
    created_at: stamp,
  };
}
function simulation(revision: string): SimulationReport {
  const outcome = fixtureValue(
    modelConstructs(freeModel).find((item) => item.name === "internalizing_symptom_burden"),
  );
  const indicator = fixtureValue(
    outcome.indicators.find((item) => item.observation.name === "gad7_screening_score"),
  );
  return {
    ...structuredClone(visualFixture.report),
    model: modelRef(revision),
    findings: [
      { kind: "construct" as const, id: outcome.id },
      { kind: "indicator" as const, id: indicator.observation.id },
    ].map((target) => ({
      kind: "not_evaluated" as const,
      subject: { check: "dispersion", target, construct_id: outcome.id },
      reason: "COMPARISON_INPUTS_MISSING" as const,
      detail: "This illustrative record has no observed comparison for the saved simulation draws.",
    })),
    ...(revision === modelId(7)
      ? {
          causal_result: null,
          causal_unavailable_reason:
            "This edited model has no committed production fit at this revision.",
        }
      : {}),
  };
}
const simulations = new Map([
  [5, simulation(modelId(5))],
  [7, simulation(modelId(7))],
]);
const models = new Map<string, ModelSpec>(
  [1, 2, 3, 4].map((revision) => {
    const seq = fixtureValue([2, 3, 4, 7][revision - 1]);
    return [modelId(revision), fixtureValue(demoSnapshotAt(seq).model).value];
  }),
);
models.set(modelId(5), freeModel);
models.set(modelId(6), freeModel);
models.set(modelId(7), pinnedModel);
const snapshots = new Map<number, ModelSnapshot>(
  [0, 1, 2, 3, 4, 5, 7].map((seq) => {
    const snapshot = demoSnapshotAt(seq);
    return [seq, { ...snapshot, workspace_id: WORKBENCH_WORKSPACE }];
  }),
);
function branchSnapshot(
  seq: number,
  info: ArtifactRecord,
  fitted: boolean,
  report?: SimulationReport,
): ModelSnapshot {
  const snapshot = demoModelSnapshot;
  const model = {
    value: fixtureValue(models.get(info.revision)),
    source: { ref: modelRef(info.revision), pointer: "", validity: "fresh" as const },
  };
  return {
    ...snapshot,
    selected_seq: seq,
    workspace_id: WORKBENCH_WORKSPACE,
    state: { ...snapshot.state, current: { ...snapshot.state.current, model: info } },
    model,
    specification: {
      value: checks,
      source: { ref: logRef(seq), pointer: "/attempt/outcome/result/checks", validity: "fresh" },
    },
    dispositions: snapshot.dispositions ? { ...snapshot.dispositions, source: model.source } : null,
    fit: snapshot.fit
      ? {
          ...snapshot.fit,
          source: {
            ref: logRef(info.revision === modelId(5) ? 8 : 10),
            pointer: "/attempt/outcome/result/report",
            validity: fitted ? "fresh" : "stale",
          },
        }
      : null,
    simulation: report
      ? {
          value: report,
          source: {
            ref: logRef(seq),
            pointer: "/attempt/outcome/result/report",
            validity: "fresh",
          },
        }
      : null,
  };
}
const v5 = metadata(modelId(5), modelId(4), "fit", freeInputs);
const v6 = metadata(modelId(6), modelId(4), "fit", freeInputs);
const v7 = metadata(modelId(7), modelId(6), "edit_model", pinnedInputs);
snapshots.set(8, branchSnapshot(8, v5, true));
snapshots.set(9, branchSnapshot(9, v5, true, simulations.get(5)));
snapshots.set(10, branchSnapshot(10, v6, true));
snapshots.set(11, branchSnapshot(11, v7, false));
snapshots.set(12, branchSnapshot(12, v7, false, simulations.get(7)));
const comparisonIndicator = fixtureValue(
  modelConstructs(freeModel)
    .flatMap((construct) => construct.indicators)
    .find((indicator) => indicator.observation.name === "gad7_screening_score"),
);
const comparedSeries = (values: number[]) => ({
  variable: comparisonIndicator.observation,
  time_origin: "2026-01-01T00:00:00Z",
  points: values.map((value, index) => ({
    anchor_time: `2026-01-0${index + 1}T00:00:00Z`,
    support_start: null,
    support_end: null,
    value,
  })),
});
const dataComparison: DataDiffReport = {
  left: [{ kind: "panel", revision: panelId }],
  right: [0, 1, 2].map((replicate) => ({ kind: "simulation", revision: commitId(9), replicate })),
  variables: [
    {
      indicator_id: comparisonIndicator.observation.id,
      left: [comparedSeries([1, 4, 3])],
      right: [comparedSeries([0, 1, 2]), comparedSeries([1, 2, 3]), comparedSeries([2, 3, 4])],
      changes: [],
      comparison_issues: [],
      reference_side: "left",
      predictive_unavailable_reason: null,
      statistics: [
        {
          statistic: "mean",
          level: null,
          left: [2.67],
          right: [1, 2, 3],
          left_histogram: [{ bin_center: 2.67, bin_start: 2.5, bin_end: 3, count: 1 }],
          right_histogram: [1, 2, 3].map((value) => ({
            bin_center: value,
            bin_start: value - 0.5,
            bin_end: value + 0.5,
            count: 1,
          })),
        },
      ],
      predictive_checks: {
        checked: true,
        n_subsample: 3,
        per_variable_warnings: [
          {
            kind: "evaluated",
            subject: {
              target: { kind: "indicator", id: comparisonIndicator.observation.id },
              check: "calibration",
            },
            outcome: "warning",
            evidence: {
              criterion: "calibration",
              note: "Observed values fall outside the replicated range.",
              value: 0.67,
              lower: 0.7,
              upper: 0.98,
              lower_inclusive: true,
              upper_inclusive: true,
              display_value: "",
              band_label: "",
            },
          },
        ],
        test_stats: [
          {
            indicator_id: comparisonIndicator.observation.id,
            stat_name: "mean",
            observed_value: 2.67,
            rep_values: [1, 2, 3],
            p_value: 0.33,
            histogram: [1, 2, 3].map((value) => ({
              bin_center: value,
              bin_start: value - 0.5,
              bin_end: value + 0.5,
              count: 1,
            })),
          },
        ],
        overlays: [
          {
            indicator_id: comparisonIndicator.observation.id,
            times: [0, 1, 2],
            time_origin: null,
            standardized: false,
            observed: [1, 4, 3],
            median: [1, 2, 3],
            spaghetti_draws: [
              [0, 1, 2],
              [1, 2, 3],
              [2, 3, 4],
            ],
          },
        ],
      },
    },
  ],
};

export const workbenchJournal: StudyRevision[] = [
  record(1, {
    action: "prepare_data",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "prepare_data",
        raw_data: null,
        model: null,
        simulation_source: null,
        n_observations: null,
        workers: [],
        ingestion_reused: null,
        extraction_reused: null,
      },
      effects: {
        produced: [fixtureValue(fixtureValue(snapshots.get(1)).state.current.raw_data)],
        retracted: [],
        checks: null,
      },
    },
  }),
  ...[2, 3, 4].map((seq) =>
    record(
      seq,
      {
        action: "edit_model",
        request: null,
        outcome: {
          status: "applied",
          result: null,
          effects: {
            produced: [fixtureValue(fixtureValue(snapshots.get(seq)).state.current.model)],
            retracted: [],
            checks: null,
          },
        },
      },
      seq === 2 ? [] : [seq === 3 ? "latent_structure" : "measurement_structure"],
    ),
  ),
  record(5, {
    action: "prepare_data",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "prepare_data",
        raw_data: null,
        model: modelRef(modelId(3)),
        simulation_source: null,
        n_observations: null,
        ingestion_reused: null,
        extraction_reused: null,
        workers: [
          {
            worker_id: 0,
            status: "failed",
            n_extractions: 0,
            n_windows: 1,
            n_llm_calls: null,
            error: "Illustrative extraction failure",
            reused: null,
          },
        ],
      },
      effects: {
        produced: [fixtureValue(fixtureValue(snapshots.get(5)).state.current.panel)],
        retracted: [],
        checks: null,
      },
    },
  }),
  record(6, {
    action: "edit_model",
    request: null,
    outcome: {
      status: "raised",
      error_type: "ProposalError",
      error_message:
        "Parameter proposal failed.\nTraceback: illustrative full failure details are retained here.",
      details: [],
    },
  }),
  record(
    7,
    {
      action: "edit_model",
      request: null,
      outcome: {
        status: "applied",
        result: null,
        effects: {
          produced: [fixtureValue(fixtureValue(snapshots.get(7)).state.current.model)],
          retracted: [],
          checks: null,
        },
      },
    },
    ["statistical_model_spec"],
  ),
  record(8, {
    action: "fit",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "fit",
        model: modelRef(modelId(4)),
        panel: { ...modelRef(panelId), path: "panel.parquet" },
        report: posterior,
        retention: "report_only",
      },
      effects: { produced: [v5], retracted: [], checks: null },
    },
  }),
  record(9, {
    action: "simulate",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "simulate",
        panel: { ...modelRef(panelId), path: "panel.parquet" },
        report: fixtureValue(simulations.get(5)),
      },
      effects: { produced: [], retracted: [], checks: null },
    },
  }),
  record(10, {
    action: "fit",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "fit",
        model: modelRef(modelId(4)),
        panel: { ...modelRef(panelId), path: "panel.parquet" },
        report: posterior,
        retention: "report_only",
      },
      effects: { produced: [v6], retracted: [], checks: null },
    },
  }),
  record(11, {
    action: "edit_model",
    request: null,
    outcome: {
      status: "applied",
      result: null,
      effects: { produced: [v7], retracted: [], checks: null },
    },
  }),
  record(12, {
    action: "simulate",
    request: null,
    outcome: {
      status: "applied",
      result: {
        action: "simulate",
        panel: { ...modelRef(panelId), path: "panel.parquet" },
        report: fixtureValue(simulations.get(7)),
      },
      effects: { produced: [], retracted: [], checks: null },
    },
  }),
  record(13, {
    action: "data_diff",
    request: {
      action: "data_diff",
      left: fixtureValue(dataComparison.left[0]),
      right: fixtureValue(dataComparison.right[0]),
    },
    outcome: {
      status: "applied",
      result: { action: "data_diff", report: dataComparison },
      effects: { produced: [], retracted: [], checks: null },
    },
  }),
];
const journal = workbenchJournal;
// This scenario includes a convergence warning and paired saved summaries.
const warned = journal.findIndex((entry) => entry.record.seq === 8);
const warnedEntry = fixtureValue(journal[warned]);
journal[warned] = {
  ...warnedEntry,
  record: {
    ...warnedEntry.record,
    messages: [
      ...warnedEntry.record.messages,
      { timestamp: stamp, level: "warn", label: "CONVERGENCE_CHECK_FAILED" },
    ],
  },
};
for (const seq of [8, 9]) {
  const fit = fixtureValue(fixtureValue(snapshots.get(seq)).fit).value;
  const row = fit.report.inference_diagnostics?.per_parameter.at(0);
  if (!row) throw new Error("The fixture fit requires recorded parameter diagnostics");
  const snapshot = fixtureValue(snapshots.get(seq));
  const sourced = fixtureValue(snapshot.fit);
  snapshots.set(seq, {
    ...snapshot,
    fit: {
      ...sourced,
      value: {
        ...fit,
        report: {
          ...fit.report,
          convergence: {
            ...fit.report.convergence,
            status: "failed",
            messages: [`R-hat fails for ${row.parameter}: 1.08`],
            assessments: [
              {
                kind: "evaluated",
                subject: { parameter: row.subject, criterion: "r_hat", label: row.parameter },
                outcome: "failed",
                evidence: {
                  criterion: "r_hat",
                  value: 1.08,
                  lower: null,
                  upper: 1.01,
                  lower_inclusive: true,
                  upper_inclusive: false,
                  note: row.parameter,
                  display_value: "",
                  band_label: "",
                },
              },
            ],
          },
        },
      },
    },
  });
}
// Explicit illustrative Git topology: alternative forks from version 7.
for (const [seq, snapshot] of snapshots) {
  snapshots.set(seq, {
    ...snapshot,
    commit_id: commitId(seq),
    branch: seq >= 10 ? "alternative" : "main",
  });
}
const commitParents: Record<number, number> = {
  1: 0,
  2: 1,
  3: 2,
  4: 3,
  5: 4,
  7: 5,
  8: 7,
  9: 8,
  10: 7,
  11: 10,
  12: 11,
  13: 9,
};
for (const [index, entry] of journal.entries()) {
  journal[index] = {
    ...entry,
    record: {
      ...entry.record,
      branch:
        entry.record.seq >= 10 && entry.record.seq <= 12 ? "alternative" : entry.record.branch,
    },
    parent_ids: [
      commitId(entry.record.seq === 6 ? 5 : fixtureValue(commitParents[entry.record.seq])),
    ],
  };
}
const branches = { main: commitId(9), alternative: commitId(12) };
// Backend-shaped argument dependencies for the journal above, as the timeline route serves them.
const dependencies: RecordDependency[] = (
  [
    [3, 2, "model"],
    [4, 3, "model"],
    [6, 4, "model"],
    [7, 4, "model"],
    [7, 5, "data_profile", true],
    [8, 4, "model"],
    [8, 5, "panel"],
    [9, 8, "model"],
    [10, 4, "model"],
    [10, 5, "panel"],
    [11, 10, "model"],
    [12, 11, "model"],
    [13, 5, "left"],
    [13, 9, "right"],
  ] as const
).map(([seq, source_seq, argument, check = false]) => ({ seq, source_seq, argument, check }));
const snapshotByCommit = (id: string | null) =>
  [...snapshots.values()].find((snapshot) => snapshot.commit_id === id);
const snapshotByModelRef = (id: string | null) =>
  snapshotByCommit(id) ??
  [...snapshots.values()].find((snapshot) => snapshot.model?.source.ref.revision === id);
export const workbenchTraces = new Map([
  [3, demoTraces.latent_structure],
  [4, demoTraces.measurement_structure],
  [7, demoTraces.statistical_model_spec],
]);
export const workbenchQuestion = demoModelSnapshot.question?.value.text;
/** Exercise the real UI requests with isolated, explicit story responses. */

export function workbenchHandlers() {
  // Only the journal fields are illustrated; this story reads no artifact freshness.
  const status = (): StudyStatus => ({
    workspace_id: WORKBENCH_WORKSPACE,
    branch: "main",
    commit_id: branches.main,
    seq: fixtureValue(journal.at(-1)).record.seq,
    state: { current: {}, checks: null },
    artifacts: [],
    actions: ["set_question", "edit_model", "prepare_data", "fit", "simulate"],
    // A fit dispatched after the alternative branch's simulation is still executing.
    running: {
      attempt_id: "0f17a770-5d1e-4c2b-9a3f-6b8e2d4c1a90",
      action: "fit",
      branch: "alternative",
      messages: [{ timestamp: "2026-09-16T12:05:00Z", level: "info", label: "FIT_STARTED" }],
    },
  });
  const timeline = (): TimelineResponse => ({
    workspace_id: WORKBENCH_WORKSPACE,
    attempts: journal,
    branches,
    dependencies,
  });
  return [
    http.get(
      `/api/studies/${WORKBENCH_WORKSPACE}/model/visuals/observations/:indicator`,
      ({ params, request }) => {
        const snapshot = snapshotByCommit(
          new URL(request.url).searchParams.get("at") ?? branches.alternative,
        );
        const histories = visualFixture.observations;
        return HttpResponse.json(
          snapshot?.metadata
            ? (presentEntries(histories).find(([id]) => id === params.indicator)?.[1] ?? null)
            : null,
        );
      },
    ),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model/visuals/parameters`, () =>
      HttpResponse.json(visualFixture.parameters),
    ),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model/visuals/predictive/:indicator`, () =>
      HttpResponse.json(null),
    ),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model/visuals/simulation`, ({ request }) => {
      const query = new URL(request.url).searchParams;
      const snapshot = snapshotByCommit(query.get("at") ?? branches.alternative);
      if (!snapshot?.simulation) return HttpResponse.json(null);
      const source = structuredClone(visualFixture.simulation);
      const start = Number(query.get("start") ?? 0),
        count = Number(query.get("count") ?? 24);
      const page = (series: PathSeries) => ({
        ...series,
        action: series.action.slice(start, start + count),
        reference: series.reference.slice(start, start + count),
      });
      return HttpResponse.json({
        ...source,
        start,
        count: Math.min(count, source.total_draws - start),
        states: Object.fromEntries(
          presentEntries(source.states).map(([id, series]) => [id, page(series)]),
        ),
        indicators: Object.fromEntries(
          presentEntries(source.indicators).map(([id, series]) => [id, page(series)]),
        ),
        effect:
          source.effect && snapshot.simulation.value.causal_result ? page(source.effect) : null,
      });
    }),
    http.post<
      Record<string, never>,
      Required<
        paths["/api/studies/{workspace_id}/model/visuals/mechanism"]["post"]["requestBody"]["content"]["application/json"]
      >
    >(`/api/studies/${WORKBENCH_WORKSPACE}/model/visuals/mechanism`, async ({ request }) => {
      const input = await request.json();
      const curves = visualFixture.mechanisms[input.owner_id];
      return curves &&
        input.lower === -3 &&
        input.upper === 3 &&
        input.start >= 0 &&
        input.start < curves.count &&
        !input.moderator &&
        (!input.axis || input.axis === curves.axis) &&
        Object.entries(input.held).every(([id, value]) =>
          presentEntries(curves.held).some(([key, held]) => key === id && held === value),
        ) &&
        input.points === 201
        ? HttpResponse.json({
            ...curves,
            total_draws: curves.count,
            start: input.start,
            count: Math.min(input.count, curves.count - input.start),
            curves: curves.curves.slice(input.start, input.start + input.count),
          })
        : HttpResponse.json(
            {
              detail:
                "This recorded story includes the default response viewport; use a live workspace to evaluate other conditions.",
            },
            { status: 422 },
          );
    }),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model/inference-report`, ({ request }) => {
      const snapshot = snapshotByCommit(
        new URL(request.url).searchParams.get("at") ?? branches.alternative,
      );
      return HttpResponse.json(
        snapshot?.fit ? { source: snapshot.fit.source, value: snapshot.fit.value.report } : null,
      );
    }),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}`, () => HttpResponse.json(status())),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/timeline`, () => HttpResponse.json(timeline())),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model`, ({ request }) => {
      const snapshot = snapshotByCommit(
        new URL(request.url).searchParams.get("at") ?? branches.alternative,
      );
      return snapshot
        ? HttpResponse.json(snapshot)
        : HttpResponse.json({ error: "Unknown story version" }, { status: 404 });
    }),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model-diff`, async ({ request }) => {
      const query = new URL(request.url).searchParams;
      await delay(180);
      const before = snapshotByModelRef(query.get("before"));
      const after = snapshotByModelRef(query.get("after"));
      if (!before?.model || !after?.model)
        return HttpResponse.json({ error: "Unknown story version" }, { status: 404 });
      const beforeVersion = fixtureValue(before.state.current.model).revision;
      const afterVersion = fixtureValue(after.state.current.model).revision;
      return HttpResponse.json({
        ...fixtureValue(definitionComparisons[`${beforeVersion}:${afterVersion}`]),
        before: { ...logRef(before.selected_seq), path: "artifacts/model/model.json" },
        after: { ...logRef(after.selected_seq), path: "artifacts/model/model.json" },
        before_checks: checks,
        after_checks: checks,
        before_fit: before.fit?.source.validity === "fresh" ? before.fit.value.report : null,
        after_fit: after.fit?.source.validity === "fresh" ? after.fit.value.report : null,
        before_simulation:
          before.simulation?.source.validity === "fresh" ? before.simulation.value : null,
        after_simulation:
          after.simulation?.source.validity === "fresh" ? after.simulation.value : null,
      } satisfies ModelDiffReport);
    }),
  ];
}
