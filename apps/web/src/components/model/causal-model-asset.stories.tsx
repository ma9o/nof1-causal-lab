import { demoModelSnapshot, demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { demoTraces } from "@/components/__fixtures__/demo-traces";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import type { ArtifactFreshness, TransitionRecord } from "@nof1-causal-lab/api-types";
import { ARTIFACT_VIEW_IDS, type LLMTrace } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { CausalModelAssetView } from "./causal-model-asset";
import type { MoveTraceState } from "./conversation-pane";

type Produced = TransitionRecord["produced"][number];
type Artifact = Produced["artifact_id"];

const START = Date.parse("2026-07-08T11:40:00Z");
let clock = START;

function produced(artifactId: Artifact, version: number): Produced {
  return {
    artifact_id: artifactId,
    version,
    provenance: "computed",
    derived_from: {},
    produced_by: null,
    created_at: new Date(clock).toISOString(),
  };
}

function move(
  seq: number,
  seconds: number,
  record: Omit<
    TransitionRecord,
    "seq" | "ts" | "reason" | "error_type" | "error_message" | "diagnostics" | "resume"
  > &
    Partial<Pick<TransitionRecord, "error_type" | "error_message">>,
): TransitionRecord {
  clock += seconds * 1000;
  return {
    seq,
    ts: new Date(clock).toISOString(),
    reason: null,
    diagnostics: {},
    resume: null,
    error_type: null,
    error_message: null,
    ...record,
  };
}

/** The DEMO fixture's journal, reconstructed: the same moves the canonical episode applied. */
const JOURNAL: TransitionRecord[] = [
  move(1, 212, {
    move: { kind: "run", artifact_id: "raw_data" },
    status: "applied",
    produced: [produced("raw_data", 1)],
    retracted: [],
    trace_ids: ["raw_data"],
  }),
  move(2, 3, {
    move: { kind: "write", artifact_id: "question", provenance: "human" },
    status: "applied",
    produced: [produced("question", 1)],
    retracted: [],
    trace_ids: [],
  }),
  move(3, 251, {
    move: { kind: "run", artifact_id: "latent_structure" },
    status: "applied",
    produced: [produced("latent_structure", 1)],
    retracted: [],
    trace_ids: ["latent_structure"],
  }),
  move(4, 175, {
    move: { kind: "run", artifact_id: "measurement_structure" },
    status: "applied",
    produced: [
      produced("measurement_structure", 1),
      produced("causal_design", 1),
      produced("structural_plan", 1),
      produced("identification_report", 1),
    ],
    retracted: [],
    trace_ids: ["measurement_structure"],
  }),
  move(5, 318, {
    move: { kind: "run", artifact_id: "measurements" },
    status: "applied",
    produced: [produced("measurements", 1), produced("panel", 1), produced("validation_report", 1)],
    retracted: [],
    trace_ids: ["measurements"],
  }),
  move(6, 74, {
    move: { kind: "run", artifact_id: "statistical_model_spec" },
    status: "raised",
    produced: [],
    retracted: [],
    trace_ids: [],
    error_type: "ValueError",
    error_message: "ValueError: prior admission rejected a channel",
  }),
  move(7, 188, {
    move: { kind: "run", artifact_id: "statistical_model_spec" },
    status: "applied",
    produced: [produced("statistical_model_spec", 1), produced("compiled_ssm", 1)],
    retracted: [],
    trace_ids: ["statistical_model_spec"],
  }),
  move(8, 1843, {
    move: { kind: "run", artifact_id: "posterior" },
    status: "applied",
    produced: [produced("posterior", 1)],
    retracted: [],
    trace_ids: [],
  }),
  move(9, 133, {
    move: { kind: "run", artifact_id: "baseline_report" },
    status: "applied",
    produced: [produced("baseline_report", 1)],
    retracted: [],
    trace_ids: ["baseline_report"],
  }),
  move(10, 40, {
    move: { kind: "write", artifact_id: "saved_scenarios", provenance: "human" },
    status: "applied",
    produced: [produced("saved_scenarios", 1)],
    retracted: [],
    trace_ids: [],
  }),
];

const TRACE_BY_SEQ: Record<number, LLMTrace> = {
  1: demoTraces.raw_data,
  3: demoTraces.latent_structure,
  4: demoTraces.measurement_structure,
  5: demoTraces.measurements,
  7: demoTraces.statistical_model_spec,
  9: demoTraces.baseline_report,
};

function useFixtureMoveTrace(seq: number, enabled: boolean): MoveTraceState {
  const trace = TRACE_BY_SEQ[seq];
  if (!enabled || !trace) return { status: "absent" };
  return { status: "ready", trace };
}

const ARTIFACTS: ArtifactFreshness[] = JOURNAL.flatMap((record) =>
  record.produced.map((info) => ({
    artifact_id: info.artifact_id,
    exists: true,
    stale: false,
    version: info.version,
    provenance: "computed" as const,
    produced_by: `run:${record.move.artifact_id}`,
  })),
);

const PROGRESS: PipelineProgress = {
  artifacts: Object.fromEntries(
    ARTIFACT_VIEW_IDS.map((id) => [id, "completed"]),
  ) as PipelineProgress["artifacts"],
  timings: {},
  transitionErrors: {},
  staleArtifactsByProducer: {},
  autoRunning: false,
  transitionOrder: [...ARTIFACT_VIEW_IDS],
  runningTransitions: [],
  isComplete: true,
  isFailed: false,
};

const QUESTION =
  "I've been on escitalopram for about two and a half years. I'm thinking about tapering. Has the medication actually been moving the needle, or did I just get better on my own — and what would my likely trajectory look like over the next two months if I taper off now versus stay on it?";

const meta = {
  title: "Model/Causal Model Asset",
  component: CausalModelAssetView,
  parameters: { layout: "fullscreen" },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  args: {
    workspaceId: "DEMO",
    useSnapshot: (atSeq: number) => ({ data: demoSnapshotAt(atSeq), error: null }),
    question: QUESTION,
    readOnly: false,
    transitions: JOURNAL,
    artifacts: ARTIFACTS,
    legal: [],
    progress: PROGRESS,
    analysisTrace: demoTraces.baseline_report,
    useMoveTrace: useFixtureMoveTrace,
    onRun: null,
  },
} satisfies Meta<typeof CausalModelAssetView>;

export default meta;

type Story = StoryObj<typeof meta>;

/** v10 · everything materialized; nothing selected shows the asset. */
export const FinalState: Story = {};

/** The same asset before the specification: no queries, no fitted layer. */
export const Measured: Story = {
  args: {
    transitions: JOURNAL.slice(0, 5),
    artifacts: ARTIFACTS.filter((artifact) =>
      [
        "raw_data",
        "question",
        "latent_structure",
        "measurement_structure",
        "causal_design",
        "structural_plan",
        "identification_report",
        "measurements",
        "panel",
        "validation_report",
      ].includes(artifact.artifact_id),
    ),
    progress: {
      ...PROGRESS,
      artifacts: {
        ...PROGRESS.artifacts,
        statistical_model_spec: "pending",
        posterior: "pending",
        baseline_report: "pending",
      },
      isComplete: false,
    },
    analysisTrace: undefined,
  },
};

/** Inference in flight: the running move pulses on the scrubber and in the thread. */
export const Materializing: Story = {
  args: {
    transitions: JOURNAL.slice(0, 7),
    artifacts: ARTIFACTS.filter(
      (artifact) =>
        !["posterior", "baseline_report", "saved_scenarios"].includes(artifact.artifact_id),
    ),
    progress: {
      ...PROGRESS,
      artifacts: { ...PROGRESS.artifacts, posterior: "running", baseline_report: "pending" },
      runningTransitions: ["posterior"],
      autoRunning: true,
      isComplete: false,
    },
    analysisTrace: undefined,
  },
};

/** Illustrative second fit: one scientific question retains both evaluations. */
const comparisonSnapshot = structuredClone(demoModelSnapshot);
const comparedScenario = comparisonSnapshot.saved_scenarios!.value.scenarios[0];
const alternative = structuredClone(comparedScenario.evaluations[0]);
alternative.evaluation.id = `evaluation:${"a".repeat(64)}`;
alternative.evaluation.model.id = "ALTERNATIVE";
alternative.result.evaluation_id = alternative.evaluation.id;
alternative.result.summary = {
  mean: 0.42,
  median: 0.4,
  lower_95: 0.1,
  upper_95: 0.8,
  prob_positive: 0.97,
};
alternative.result.effect_trajectory = null;
alternative.result.trajectory_peak = null;
comparedScenario.evaluations.push(alternative);

export const AcrossFits: Story = {
  args: {
    useSnapshot: (atSeq: number) => ({
      data: atSeq === 10 ? comparisonSnapshot : demoSnapshotAt(atSeq),
      error: null,
    }),
  },
};
