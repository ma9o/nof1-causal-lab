import { primaryArtifact } from "@/lib/model-asset/journal";
import { demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { demoTraces } from "@/components/__fixtures__/demo-traces";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import type { ArtifactFreshness, TransitionRecord } from "@nof1-causal-lab/api-types";
import { TRANSITIONS, type LLMTrace } from "@nof1-causal-lab/api-types";
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
    model_inputs: {},
    consumed_model_inputs: {},
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
    move: { kind: "run", operation_id: "raw_data" },
    status: "applied",
    produced: [produced("raw_data", 1)],
    retracted: [],
    trace_ids: ["raw_data"],
  }),
  move(2, 3, {
    move: { kind: "write", artifact_id: "model", provenance: "human", expected_model_version: 0 },
    status: "applied",
    produced: [produced("model", 1)],
    retracted: [],
    trace_ids: [],
  }),
  move(3, 251, {
    move: { kind: "run", operation_id: "latent_structure" },
    status: "applied",
    produced: [produced("model", 2)],
    retracted: [],
    trace_ids: ["latent_structure"],
  }),
  move(4, 175, {
    move: { kind: "run", operation_id: "measurement_structure" },
    status: "applied",
    produced: [produced("model", 3), produced("identification_report", 1)],
    retracted: [],
    trace_ids: ["measurement_structure"],
  }),
  move(5, 318, {
    move: { kind: "run", operation_id: "measurements" },
    status: "applied",
    produced: [produced("panel", 1), produced("validation_report", 1)],
    retracted: [],
    trace_ids: ["measurements"],
  }),
  move(6, 74, {
    move: { kind: "run", operation_id: "statistical_model_spec" },
    status: "raised",
    produced: [],
    retracted: [],
    trace_ids: [],
    error_type: "ValueError",
    error_message: "ValueError: prior admission rejected a channel",
  }),
  move(7, 188, {
    move: { kind: "run", operation_id: "statistical_model_spec" },
    status: "applied",
    produced: [produced("model", 4)],
    retracted: [],
    trace_ids: ["statistical_model_spec"],
  }),
  move(8, 1843, {
    move: { kind: "run", operation_id: "posterior" },
    status: "applied",
    produced: [],
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
    produced_by: `run:${primaryArtifact(record.move)}`,
  })),
);

const PROGRESS: PipelineProgress = {
  artifacts: Object.fromEntries(
    TRANSITIONS.map((section) => [section.id, "completed"]),
  ) as PipelineProgress["artifacts"],
  timings: {},
  transitionErrors: {},
  staleArtifactsByProducer: {},
  autoRunning: false,
  transitionOrder: TRANSITIONS.map((section) => section.id),
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
    nextOperation: null,
    progress: PROGRESS,
    useMoveTrace: useFixtureMoveTrace,
    onRun: null,
  },
} satisfies Meta<typeof CausalModelAssetView>;

export default meta;

type Story = StoryObj<typeof meta>;

/** The fitted model; nothing selected shows the asset. */
export const FinalState: Story = {};

/** The same asset before the specification: no queries, no fitted layer. */
export const Measured: Story = {
  args: {
    transitions: JOURNAL.slice(0, 5),
    artifacts: ARTIFACTS.filter((artifact) =>
      [
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "causal_design",
        "model",
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
      },
      isComplete: false,
    },
  },
};

/** Inference in flight: the running move pulses on the scrubber and in the thread. */
export const Materializing: Story = {
  args: {
    transitions: JOURNAL.slice(0, 7),
    artifacts: ARTIFACTS,
    progress: {
      ...PROGRESS,
      artifacts: { ...PROGRESS.artifacts, posterior: "running" },
      runningTransitions: ["posterior"],
      autoRunning: true,
      isComplete: false,
    },
  },
};
