import type {
  ArtifactRecord,
  ModelDiffReport,
  ModelSnapshot,
  ModelSpec,
  SimulationReport,
  SimulationTrajectories,
  SpecificationReport,
  StudyRevision,
} from "@nof1-causal-lab/api-types";
import { delay, HttpResponse, http } from "msw";
import type { EpisodeProgressPayload } from "@/lib/api/analysis";
import { demoModelSnapshot, demoSnapshotAt } from "./demo-artifacts";
import { demoTraces } from "./demo-traces";
import comparisonFixture from "./workbench-comparisons.json";
import simulationFixture from "./workbench-simulation.json";
export const WORKBENCH_WORKSPACE = "STORYBOOK";
const stamp = "2026-09-16T12:00:00Z";
// Illustrative interface data. Branch metadata and parameter decisions are staged;
// retained DEMO evidence is reused for presentation, not claimed as new inference.
const freeModel = structuredClone(demoModelSnapshot.model!.value);
// Generated and validated by scripts/fixtures/study.py.
const pinnedModel = comparisonFixture.pinned_model as unknown as ModelSpec;
const freeInputs = demoModelSnapshot.context.state.current.model!.model_inputs;
const pinnedInputs = comparisonFixture.pinned_inputs;
const definitionComparisons = comparisonFixture.comparisons as unknown as Record<
  string,
  Pick<ModelDiffReport, "graph" | "parameters" | "changed_inputs" | "definition_changes">
>;
const checks: SpecificationReport = {
  findings: [
    {
      check: "specification",
      status: "not_evaluated",
      message: "Specification checks have not been run.",
    },
  ],
};
const commitId = (seq: number) => seq.toString(16).padStart(40, "c");
const modelId = (ordinal: number): string =>
  ordinal <= 4
    ? demoSnapshotAt([2, 3, 4, 7][ordinal - 1]).context.state.current.model!.revision
    : ordinal.toString(16).padStart(40, "a");
const panelId = demoModelSnapshot.context.state.current.panel!.revision;
const modelRef = (revision: string) => ({
  workspace_id: WORKBENCH_WORKSPACE,
  revision,
  path: "model.json",
});
const logRef = (seq: number) => ({
  workspace_id: WORKBENCH_WORKSPACE,
  revision: commitId(seq),
  path: "logs/transition.json",
});
function record(
  seq: number,
  action: Pick<StudyRevision, "action" | "inputs" | "operation_id">,
  produced: ArtifactRecord[] = [],
  diagnostics: StudyRevision["diagnostics"] = {},
): StudyRevision {
  return {
    seq,
    commit_id: commitId(seq),
    parent_ids: [commitId(seq - 1)],
    branch: "main",
    ts: stamp,
    ...action,
    produced,
    diagnostics,
    messages: [{ timestamp: stamp, level: "info", label: "ACTION_COMPLETED" }],
    status: "applied",
    retracted: [],
    trace_ids: [],
    reason: null,
    error_type: null,
    error_message: null,
    resume: null,
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
    derived_from:
      produced_by === "run:posterior" ? { model: parent, panel: panelId } : { model: parent },
    model_inputs,
    consumed_model_inputs: {},
    produced_by,
    created_at: stamp,
  };
}
function simulation(revision: string): SimulationReport {
  return {
    ...structuredClone(simulationFixture.report as unknown as SimulationReport),
    model: modelRef(revision),
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
function simulationInputs(report: SimulationReport) {
  return {
    model_revision: report.model.revision,
    start: report.design.start ?? null,
    end: report.design.end,
    interventions: report.design.interventions.map((event) => ({ ...event })),
  };
}
const models = new Map<string, ModelSpec>(
  [1, 2, 3, 4].map((revision) => {
    const seq = [2, 3, 4, 7][revision - 1];
    return [modelId(revision), demoSnapshotAt(seq).model!.value];
  }),
);
models.set(modelId(5), freeModel);
models.set(modelId(6), freeModel);
models.set(modelId(7), pinnedModel);
const snapshots = new Map<number, ModelSnapshot>(
  [0, 1, 2, 3, 4, 5, 7].map((seq) => {
    const snapshot = structuredClone(demoSnapshotAt(seq));
    snapshot.context.workspace_id = WORKBENCH_WORKSPACE;
    return [seq, snapshot];
  }),
);
function branchSnapshot(
  seq: number,
  info: ArtifactRecord,
  fitted: boolean,
  report?: SimulationReport,
): ModelSnapshot {
  const snapshot = structuredClone(demoModelSnapshot);
  snapshot.context.workspace_id = WORKBENCH_WORKSPACE;
  snapshot.context.seq = seq;
  snapshot.context.state.current.model = info;
  snapshot.context.artifacts = snapshot.context.artifacts.map((artifact) =>
    artifact.artifact_id === "model"
      ? {
          ...artifact,
          revision: info.revision,
          produced_by: info.produced_by,
        }
      : artifact,
  );
  snapshot.model = {
    value: models.get(info.revision)!,
    source: {
      ref: modelRef(info.revision),
      pointer: "",
      validity: "fresh",
    },
  };
  snapshot.findings.specification = {
    value: checks,
    source: {
      ref: logRef(seq),
      pointer: "/diagnostics/checks",
      validity: "fresh",
    },
  };
  if (snapshot.findings.dispositions) snapshot.findings.dispositions.source = snapshot.model.source;
  if (snapshot.findings.fit)
    snapshot.findings.fit.source = {
      ref: logRef(info.revision === modelId(5) ? 8 : 10),
      pointer: "/diagnostics/report",
      validity: fitted ? "fresh" : "stale",
    };
  snapshot.findings.simulation = report
    ? {
        value: report,
        source: { ref: logRef(seq), pointer: "/diagnostics/report", validity: "fresh" },
      }
    : null;
  return snapshot;
}
const v5 = metadata(modelId(5), modelId(4), "run:posterior", freeInputs);
const v6 = metadata(modelId(6), modelId(4), "run:posterior", freeInputs);
const v7 = metadata(modelId(7), modelId(6), "write:model", pinnedInputs);
snapshots.set(8, branchSnapshot(8, v5, true));
snapshots.set(9, branchSnapshot(9, v5, true, simulations.get(5)));
snapshots.set(10, branchSnapshot(10, v6, true));
snapshots.set(11, branchSnapshot(11, v7, false));
snapshots.set(12, branchSnapshot(12, v7, false, simulations.get(7)));
const journal: StudyRevision[] = [
  record(
    1,
    {
      action: "prepare_data",
      operation_id: "raw_data",
      inputs: {},
    },
    [snapshots.get(1)!.context.state.current.raw_data!],
  ),
  record(
    2,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: null },
    },
    [snapshots.get(2)!.context.state.current.model!],
  ),
  {
    ...record(
      3,
      {
        action: "edit_model",
        operation_id: "latent_structure",
        inputs: {},
      },
      [snapshots.get(3)!.context.state.current.model!],
    ),
    trace_ids: ["latent_structure"],
  },
  {
    ...record(
      4,
      {
        action: "edit_model",
        operation_id: "measurement_structure",
        inputs: {},
      },
      [snapshots.get(4)!.context.state.current.model!],
    ),
    trace_ids: ["measurement_structure"],
  },
  record(
    5,
    {
      action: "prepare_data",
      operation_id: "measurements",
      inputs: {
        input: {
          source: { file: "demo-observations.parquet" },
          variables: snapshots
            .get(5)!
            .data.metadata!.value.variables.map((variable) => ({ ...variable })),
        },
      },
    },
    [snapshots.get(5)!.context.state.current.panel!],
  ),
  {
    ...record(6, {
      action: "edit_model",
      operation_id: "statistical_model_spec",
      inputs: {},
    }),
    status: "raised",
    error_type: "ValueError",
    error_message:
      "Parameter proposal was rejected.\nTraceback: illustrative full failure details are retained here, never on the timeline tick.",
  },
  {
    ...record(
      7,
      {
        action: "edit_model",
        operation_id: "statistical_model_spec",
        inputs: {},
      },
      [snapshots.get(7)!.context.state.current.model!],
    ),
    trace_ids: ["statistical_model_spec"],
  },
  record(
    8,
    {
      action: "fit",
      operation_id: "posterior",
      inputs: { model_revision: modelId(4), panel_revision: panelId },
    },
    [v5],
  ),
  record(9, {
    action: "simulate",
    operation_id: "simulate",
    inputs: simulationInputs(simulations.get(5)!),
  }),
  record(
    10,
    {
      action: "fit",
      operation_id: "posterior",
      inputs: { model_revision: modelId(4), panel_revision: panelId },
    },
    [v6],
  ),
  record(
    11,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: modelId(6) },
    },
    [v7],
  ),
  record(12, {
    action: "simulate",
    operation_id: "simulate",
    inputs: simulationInputs(simulations.get(7)!),
  }),
];
// This single scenario includes a table import, a backend convergence warning,
// paired trajectories with bands, and a later simulation without a certified effect.
for (const snapshot of snapshots.values()) {
  if (snapshot.data.metadata)
    snapshot.data.metadata.value.source = { file: "demo-observations.parquet" };
}
journal
  .find((entry) => entry.seq === 8)!
  .messages.push({ timestamp: stamp, level: "warn", label: "CONVERGENCE_CHECK_FAILED" });
// Explicit illustrative Git topology: alternative forks from version 7.
for (const [seq, snapshot] of snapshots) {
  snapshot.context.commit_id = commitId(seq);
  snapshot.context.branch = seq >= 10 ? "alternative" : "main";
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
};
for (const entry of journal) {
  if (entry.seq >= 10) entry.branch = "alternative";
  entry.parent_ids = [commitId(entry.seq === 6 ? 5 : commitParents[entry.seq])];
}
const branches = { main: commitId(9), alternative: commitId(12) };
const snapshotByCommit = (id: string | null) =>
  [...snapshots.values()].find((snapshot) => snapshot.context.commit_id === id);
const snapshotByModelRef = (id: string | null) =>
  snapshotByCommit(id) ??
  [...snapshots.values()].find((snapshot) => snapshot.context.state.current.model?.revision === id);
export const workbenchTraces = new Map([
  [3, demoTraces.latent_structure],
  [4, demoTraces.measurement_structure],
  [7, demoTraces.statistical_model_spec],
]);
export const workbenchQuestion = freeModel.question ?? undefined;
/** Exercise the real UI requests with isolated, explicit story responses. */
export function workbenchHandlers() {
  const latest = snapshots.get(12)!;
  const progress = (): EpisodeProgressPayload => ({
    workspaceId: WORKBENCH_WORKSPACE,
    seq: journal.at(-1)!.seq,
    artifacts: latest.context.artifacts,
    actions: ["edit_model", "prepare_data", "fit", "simulate"],
    transitions: journal,
    branches,
    events: [],
  });
  return [
    http.get(`/api/analysis/${WORKBENCH_WORKSPACE}/progress`, () => HttpResponse.json(progress())),
    http.get(`/api/episodes/${WORKBENCH_WORKSPACE}/model`, ({ request }) => {
      const snapshot = snapshotByCommit(
        new URL(request.url).searchParams.get("at") ?? branches.alternative,
      );
      return snapshot
        ? HttpResponse.json(snapshot)
        : HttpResponse.json({ error: "Unknown story version" }, { status: 404 });
    }),
    http.get(`/api/episodes/${WORKBENCH_WORKSPACE}/model-diff`, async ({ request }) => {
      const query = new URL(request.url).searchParams;
      await delay(180);
      const before = snapshotByModelRef(query.get("before"));
      const after = snapshotByModelRef(query.get("after"));
      if (!before?.model || !after?.model)
        return HttpResponse.json({ error: "Unknown story version" }, { status: 404 });
      const beforeVersion = before.context.state.current.model!.revision;
      const afterVersion = after.context.state.current.model!.revision;
      return HttpResponse.json({
        ...definitionComparisons[`${beforeVersion}:${afterVersion}`],
        before: { ...logRef(before.context.seq), path: "artifacts/model/model.json" },
        after: { ...logRef(after.context.seq), path: "artifacts/model/model.json" },
        before_checks: checks,
        after_checks: checks,
        before_fit:
          before.findings.fit?.source.validity === "fresh"
            ? before.findings.fit.value.report
            : null,
        after_fit:
          after.findings.fit?.source.validity === "fresh" ? after.findings.fit.value.report : null,
        before_simulation:
          before.findings.simulation?.source.validity === "fresh"
            ? before.findings.simulation.value
            : null,
        after_simulation:
          after.findings.simulation?.source.validity === "fresh"
            ? after.findings.simulation.value
            : null,
      } satisfies ModelDiffReport);
    }),
    http.get(
      `/api/episodes/${WORKBENCH_WORKSPACE}/model/simulation-trajectories`,
      ({ request }) => {
        const snapshot = snapshotByCommit(new URL(request.url).searchParams.get("at"));
        const simulation = snapshot?.findings.simulation;
        return HttpResponse.json(
          simulation
            ? {
                value: simulationFixture.trajectories as unknown as SimulationTrajectories,
                source: simulation.source,
              }
            : null,
        );
      },
    ),
  ];
}
