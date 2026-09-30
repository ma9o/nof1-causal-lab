import type { StudyRevision } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { journalTicks, latestSeq } from "./journal";
type Produced = StudyRevision["produced"][number];
function produced(artifactId: Produced["artifact_id"], revision: string): Produced {
  return {
    artifact_id: artifactId,
    revision,
    derived_from: {},
    model_inputs: {},
    consumed_model_inputs: {},
    produced_by: null,
    created_at: "2026-07-08T11:57:25Z",
  };
}
function record(
  seq: number,
  action: Pick<StudyRevision, "action" | "inputs" | "operation_id">,
  status: StudyRevision["status"],
  extra: Partial<StudyRevision> = {},
): StudyRevision {
  return {
    seq,
    commit_id: String(seq).padStart(40, "a"),
    parent_ids: [],
    ts: `2026-07-08T11:57:${String(seq).padStart(2, "0")}Z`,
    ...action,
    status,
    branch: "main",
    reason: null,
    diagnostics: {},
    messages: [],
    resume: null,
    error_type: null,
    error_message: null,
    produced: [],
    retracted: [],
    trace_ids: [],
    ...extra,
  };
}
const JOURNAL: StudyRevision[] = [
  record(
    1,
    {
      action: "prepare_data",
      operation_id: "raw_data",
      inputs: {},
    },
    "applied",
    {
      produced: [produced("raw_data", "0000000000000000000000000000000000000001")],
      trace_ids: ["raw_data"],
    },
  ),
  record(
    2,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: null },
    },
    "applied",
    {
      produced: [produced("model", "0000000000000000000000000000000000000001")],
    },
  ),
  record(
    3,
    {
      action: "edit_model",
      operation_id: "measurement_structure",
      inputs: {},
    },
    "applied",
    {
      produced: [
        produced("model", "0000000000000000000000000000000000000002"),
        produced("identification_report", "0000000000000000000000000000000000000001"),
      ],
      trace_ids: ["measurement_structure"],
    },
  ),
  record(
    4,
    {
      action: "edit_model",
      operation_id: "statistical_model_spec",
      inputs: {},
    },
    "raised",
    {
      error_type: "ValueError",
      error_message: "prior admission failed",
    },
  ),
  record(
    5,
    {
      action: "edit_model",
      operation_id: "statistical_model_spec",
      inputs: {},
    },
    "rejected",
    {
      reason: "inputs missing",
    },
  ),
  record(
    6,
    {
      action: "edit_model",
      operation_id: "statistical_model_spec",
      inputs: {},
    },
    "applied",
    {
      produced: [
        produced("model", "0000000000000000000000000000000000000003"),
        produced("identification_report", "0000000000000000000000000000000000000001"),
      ],
    },
  ),
  record(
    7,
    {
      action: "edit_model",
      operation_id: null,
      inputs: {},
    },
    "applied",
    {
      produced: [
        produced("model", "0000000000000000000000000000000000000004"),
        produced("identification_report", "0000000000000000000000000000000000000002"),
      ],
      retracted: [{ artifact_id: "identification_report", reason_ref: "stale-spec" }],
    },
  ),
];
describe("journalTicks", () => {
  it("keeps every attempt for activity, including rejection reasons", () => {
    const ticks = journalTicks(JOURNAL);
    expect(ticks.map((tick) => tick.seq)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(ticks[4].error).toBe("inputs missing");
  });
  it("retains produced artifacts for the action record", () => {
    const [, , design, raised] = journalTicks(JOURNAL);
    expect(design.produced.map((info) => [info.artifact_id, info.revision])).toEqual([
      ["model", "2".padStart(40, "0")],
      ["identification_report", "1".padStart(40, "0")],
    ]);
    expect(raised.error).toBe("prior admission failed");
  });
});
it("retains simulation findings as a committed checkpoint without an artifact output", () => {
  const simulation = record(
    8,
    {
      action: "simulate",
      operation_id: "simulate",
      inputs: { model_revision: "4".padStart(40, "0") },
    },
    "applied",
    { diagnostics: { simulation: { findings: [] } } },
  );
  const [tick] = journalTicks([simulation]);
  expect(tick.produced).toEqual([]);
  expect(latestSeq([...JOURNAL, simulation])).toBe(8);
});
describe("latestSeq", () => {
  it("excludes failed attempts from model revisions", () => {
    expect(latestSeq(JOURNAL.slice(0, 5))).toBe(3);
  });
});
