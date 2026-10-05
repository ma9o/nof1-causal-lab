import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type {
  ActionAttempt,
  ArtifactRecord,
  DataDiffReport,
  ModelDiffReport,
  ModelSnapshot,
  ModelSpec,
  SimulationReport,
  RecordDependency,
  SpecificationAssessment,
  StudyRevision,
  CompletedPoll,
  PrepareDataRequest,
  TimelineResponse,
  TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { delay, HttpResponse, http } from "msw";
import { modelConstructs } from "@/lib/model-accessors";
import { demoModelSnapshot, demoSnapshotAt } from "./demo-artifacts";
import { demoTraces } from "./demo-traces";
import comparisonFixture from "./workbench-comparisons.json";
import visualFixture from "./workbench-visuals.json";
export const WORKBENCH_WORKSPACE = "STORYBOOK";
const stamp = "2026-09-16T12:00:00Z";
// Illustrative interface data. Parameter decisions are staged;
// retained DEMO evidence is reused for presentation, not claimed as new inference.
const freeModel = structuredClone(fixtureValue(demoModelSnapshot.model).value);
// Generated and validated by scripts/fixtures/study.py.
const pinnedModel = comparisonFixture.pinned_model;
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
      ts: stamp,
      attempt,
      trace_ids,
      messages: [{ timestamp: stamp, level: "info", label: "ACTION_COMPLETED" }],
    },
  };
}
function metadata(revision: string, parent: string, produced_by: string): ArtifactRecord {
  return {
    artifact_id: "model",
    revision,
    derived_from: produced_by === "fit" ? { model: parent, panel: panelId } : { model: parent },
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
    evidence: { ...structuredClone(visualFixture.report.evidence), model: modelRef(revision) },
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
          causal: {
            kind: "unavailable" as const,
            reason: "This edited model has no committed production fit at this revision.",
          },
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
function illustratedSnapshot(
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
      source: { ref: modelRef(info.revision), pointer: "", validity: "fresh" },
    },
    dispositions: snapshot.dispositions ? { ...snapshot.dispositions, source: model.source } : null,
    fit: snapshot.fit
      ? {
          ...snapshot.fit,
          source: {
            ref: logRef(info.revision === modelId(5) ? 8 : 10),
            pointer: "/attempt/outcome/result/evidence",
            validity: fitted ? "fresh" : "stale",
          },
        }
      : null,
    simulation: report
      ? {
          value: report,
          source: {
            ref: logRef(seq),
            pointer: "/attempt/outcome/result/evidence",
            validity: "fresh",
          },
        }
      : null,
  };
}
const v5 = metadata(modelId(5), modelId(4), "fit");
const v6 = metadata(modelId(6), modelId(4), "fit");
const v7 = metadata(modelId(7), modelId(6), "edit_model");
snapshots.set(8, illustratedSnapshot(8, v5, true));
snapshots.set(9, illustratedSnapshot(9, v5, true, simulations.get(5)));
snapshots.set(10, illustratedSnapshot(10, v6, true));
snapshots.set(11, illustratedSnapshot(11, v7, false));
snapshots.set(12, illustratedSnapshot(12, v7, false, simulations.get(7)));
const comparisonIndicator = fixtureValue(
  modelConstructs(freeModel)
    .flatMap((construct) => construct.indicators)
    .find((indicator) => indicator.observation.name === "gad7_screening_score"),
);
const comparedSeries = (values: number[]) => ({
  variable: {
    ...comparisonIndicator.observation,
    observation_window: "1d",
  },
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
      predictive: {
        kind: "comparison",
        reference_side: "left",
        evaluation: {
          kind: "available",
          value: {
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
                frame: [1, 3],
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
                frame: [0, 4],
              },
            ],
          },
        },
      },
    },
  ],
};

const demoMetadata = fixtureValue(demoModelSnapshot.metadata).value;
if (demoMetadata.kind !== "file")
  throw new Error("The workbench requires a file preparation recipe");
const retainedMetadata = demoMetadata;
function preparation(seq: number): PrepareDataRequest {
  return {
    action: "prepare_data",
    reasoning: null,
    input: {
      source: {
        ...retainedMetadata.source,
        hashes: Object.fromEntries(
          retainedMetadata.source.files.map((name) => [name, "0".repeat(64)]),
        ),
      },
      definition: { ...retainedMetadata.preparation, context: `Illustrative preparation ${seq}` },
    },
  };
}
const settings = (seed: number) => ({
  num_samples: null,
  num_warmup: null,
  num_chains: null,
  n_particles: null,
  seed,
});
const workbenchRecords: StudyRevision[] = [
  record(1, {
    action: "prepare_data",
    request: preparation(1),
    outcome: {
      status: "applied",
      result: {
        workers: [],
        ingestion_reused: null,
        extraction_reused: null,
      },
      effects: {
        produced: [fixtureValue(fixtureValue(snapshots.get(1)).state.current.raw_data)],
        retracted: [],
        reports: {},
      },
    },
  }),
  ...[2, 3, 4].map((seq) =>
    record(
      seq,
      {
        action: "edit_model",
        request: {
          action: "edit_model",
          reasoning: null,
          expected_revision: seq === 2 ? null : modelId(seq - 2),
          panel_revision: null,
          model: fixtureValue(fixtureValue(snapshots.get(seq)).model).value,
        },
        outcome: {
          status: "applied",
          result: null,
          effects: {
            produced: [fixtureValue(fixtureValue(snapshots.get(seq)).state.current.model)],
            retracted: [],
            reports: {},
          },
        },
      },
      seq === 2 ? [] : [seq === 3 ? "latent_structure" : "measurement_structure"],
    ),
  ),
  record(5, {
    action: "prepare_data",
    request: preparation(5),
    outcome: {
      status: "applied",
      result: {
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
        reports: {},
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
      request: {
        action: "edit_model",
        reasoning: null,
        expected_revision: modelId(3),
        panel_revision: panelId,
        model: fixtureValue(models.get(modelId(4))),
      },
      outcome: {
        status: "applied",
        result: null,
        effects: {
          produced: [fixtureValue(fixtureValue(snapshots.get(7)).state.current.model)],
          retracted: [],
          reports: {},
        },
      },
    },
    ["statistical_model_spec"],
  ),
  record(8, {
    action: "fit",
    request: {
      action: "fit",
      reasoning: null,
      model_revision: modelId(4),
      panel_revision: panelId,
      settings: settings(8),
    },
    outcome: {
      status: "applied",
      result: null,
      effects: { produced: [v5], retracted: [], reports: {} },
    },
  }),
  record(9, {
    action: "simulate",
    request: {
      action: "simulate",
      reasoning: null,
      model_revision: modelId(5),
      panel_revision: panelId,
      ...fixtureValue(simulations.get(5)).evidence.design,
    },
    outcome: {
      status: "applied",
      result: {
        evidence: fixtureValue(simulations.get(5)).evidence,
      },
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
  record(10, {
    action: "fit",
    request: {
      action: "fit",
      reasoning: null,
      model_revision: modelId(4),
      panel_revision: panelId,
      settings: settings(10),
    },
    outcome: {
      status: "applied",
      result: null,
      effects: { produced: [v6], retracted: [], reports: {} },
    },
  }),
  record(11, {
    action: "edit_model",
    request: {
      action: "edit_model",
      reasoning: null,
      expected_revision: modelId(6),
      panel_revision: panelId,
      model: pinnedModel,
    },
    outcome: {
      status: "applied",
      result: null,
      effects: { produced: [v7], retracted: [], reports: {} },
    },
  }),
  record(12, {
    action: "simulate",
    request: {
      action: "simulate",
      reasoning: null,
      model_revision: modelId(7),
      panel_revision: panelId,
      ...fixtureValue(simulations.get(7)).evidence.design,
    },
    outcome: {
      status: "applied",
      result: {
        evidence: fixtureValue(simulations.get(7)).evidence,
      },
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
  record(13, {
    action: "data_diff",
    request: {
      action: "data_diff",
      reasoning: null,
      left: fixtureValue(dataComparison.left[0]),
      right: fixtureValue(dataComparison.right[0]),
    },
    outcome: {
      status: "applied",
      result: null,
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
];
const journal = workbenchRecords;
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
// Failed attempts and comparisons are leaves; successful scientific calls form one journal.
for (const [seq, snapshot] of snapshots)
  snapshots.set(seq, { ...snapshot, commit_id: commitId(seq) });
const commitParents: Record<number, number> = {
  1: 0,
  2: 1,
  3: 2,
  4: 3,
  5: 4,
  6: 5,
  7: 5,
  8: 7,
  9: 8,
  10: 9,
  11: 10,
  12: 11,
  13: 12,
};
for (const [index, entry] of journal.entries())
  journal[index] = {
    ...entry,
    parent_ids: [commitId(fixtureValue(commitParents[entry.record.seq]))],
  };
snapshots.set(13, {
  ...fixtureValue(snapshots.get(12)),
  selected_seq: 13,
  commit_id: commitId(13),
});
// Backend-shaped argument dependencies for the journal above, as the timeline route serves them.
const dependencies: RecordDependency[] = (
  [
    [3, 2, "model"],
    [4, 3, "model"],
    [7, 4, "model"],
    [7, 5, "panel"],
    [8, 7, "model"],
    [8, 5, "panel"],
    [9, 8, "model"],
    [9, 5, "panel"],
    [10, 7, "model"],
    [10, 5, "panel"],
    [11, 10, "model"],
    [11, 5, "panel"],
    [12, 11, "model"],
    [12, 5, "panel"],
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
export const workbenchJournal: TimelineRevision[] = journal.map((entry) => {
  const outcome = entry.record.attempt.outcome;
  return {
    ...entry,
    record: {
      seq: entry.record.seq,
      ts: entry.record.ts,
      messages: entry.record.messages,
      trace_ids: entry.record.trace_ids,
      attempt: {
        ...entry.record.attempt,
        outcome:
          outcome.status === "applied"
            ? { ...outcome, result: null, effects: { ...outcome.effects } }
            : outcome,
      },
    },
  };
});

/** Complete retained results backing the seven public call routes. */
export function workbenchResult(seq: number): CompletedPoll {
  const entry = fixtureValue(journal.find((item) => item.record.seq === seq));
  const snapshot = snapshots.get(seq) ?? null;
  const paths = snapshot?.simulation
    ? {
        ...visualFixture.simulation,
        effect:
          snapshot.simulation.value.causal.kind === "available"
            ? visualFixture.simulation.effect
            : null,
      }
    : null;
  return {
    kind: "completed",
    commit_id: entry.commit_id,
    attempt: entry.record.attempt,
    messages: entry.record.messages,
    snapshot,
    inference_report: null,
    data_comparison: entry.record.attempt.action === "data_diff" ? dataComparison : null,
    model_comparison: null,
    checks:
      entry.record.attempt.action === "edit_model" || entry.record.attempt.action === "fit"
        ? {
            specification: fixtureValue(snapshot).specification?.value ?? [],
            question: fixtureValue(snapshot).question_checks?.value ?? null,
            predictive: fixtureValue(snapshot).predictive?.value ?? null,
            reused: [],
          }
        : null,
    observation_histories: snapshot?.metadata ? visualFixture.observations : {},
    predictive_overlays: {},
    simulation_paths: paths,
    parameter_draws: visualFixture.parameters,
    traces: Object.fromEntries(
      entry.record.trace_ids.map((id) => [id, fixtureValue(workbenchTraces.get(seq))]),
    ),
    artifacts: {},
    arrays: {},
  };
}

/** Exercise the real UI requests with isolated, explicit story responses. */
export function workbenchHandlers() {
  const timeline = (): TimelineResponse => ({
    attempts: workbenchJournal,
    dependencies,
    running: {
      attempt_id: "0f17a770-5d1e-4c2b-9a3f-6b8e2d4c1a90",
      action: "fit",
      events: [],
      request: {
        action: "fit",
        reasoning: null,
        model_revision: modelId(7),
        panel_revision: panelId,
        settings: settings(13),
      },
      messages: [{ timestamp: "2026-09-16T12:05:00Z", level: "info", label: "FIT_STARTED" }],
    },
  });
  return [
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/timeline`, () => HttpResponse.json(timeline())),
    ...[
      "set_question",
      "edit_model",
      "prepare_data",
      "fit",
      "simulate",
      "data_diff",
      "model_diff",
    ].map((action) =>
      http.post(`/api/studies/${WORKBENCH_WORKSPACE}/${action}`, async ({ request }) => {
        const body = await request.json();
        const entry = journal.find(
          (item) => JSON.stringify(item.record.attempt.request) === JSON.stringify(body),
        );
        return entry
          ? HttpResponse.json(workbenchResult(entry.record.seq))
          : HttpResponse.json({ detail: "No saved story call" }, { status: 403 });
      }),
    ),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/model-comparison`, async ({ request }) => {
      const query = new URL(request.url).searchParams;
      await delay(180);
      const before = snapshotByModelRef(query.get("before"));
      const after = snapshotByModelRef(query.get("after"));
      if (!before || !after)
        return HttpResponse.json({ error: "Unknown story version" }, { status: 404 });
      const beforeVersion = before.model?.source.ref.revision ?? "no-model";
      const afterVersion = after.model?.source.ref.revision ?? "no-model";
      return HttpResponse.json({
        ...fixtureValue(definitionComparisons[`${beforeVersion}:${afterVersion}`]),
        before_model: before.model?.value ?? null,
        after_model: after.model?.value ?? null,
        before: before.model?.source.ref ?? null,
        after: after.model?.source.ref ?? null,
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
