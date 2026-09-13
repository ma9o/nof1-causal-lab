import type { TransitionRecord } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { journalTicks, latestSeq } from "./journal";

type Produced = TransitionRecord["produced"][number];

function produced(artifactId: Produced["artifact_id"], version: number): Produced {
  return {
    artifact_id: artifactId,
    version,
    provenance: "computed",
    derived_from: {},
    produced_by: null,
    created_at: "2026-07-08T11:57:25Z",
  };
}

function record(
  seq: number,
  move: TransitionRecord["move"],
  status: TransitionRecord["status"],
  extra: Partial<TransitionRecord> = {},
): TransitionRecord {
  return {
    seq,
    ts: `2026-07-08T11:57:${String(seq).padStart(2, "0")}Z`,
    move,
    status,
    reason: null,
    diagnostics: {},
    resume: null,
    error_type: null,
    error_message: null,
    produced: [],
    retracted: [],
    trace_ids: [],
    ...extra,
  };
}

const JOURNAL: TransitionRecord[] = [
  record(1, { kind: "run", artifact_id: "raw_data" }, "applied", {
    produced: [produced("raw_data", 1)],
    trace_ids: ["raw_data"],
  }),
  record(2, { kind: "write", artifact_id: "question", provenance: "human" }, "applied", {
    produced: [produced("question", 1)],
  }),
  record(3, { kind: "run", artifact_id: "measurement_structure" }, "applied", {
    produced: [
      produced("measurement_structure", 1),
      produced("causal_design", 1),
      produced("structural_plan", 1),
    ],
    trace_ids: ["measurement_structure"],
  }),
  record(4, { kind: "run", artifact_id: "statistical_model_spec" }, "raised", {
    error_type: "ValueError",
    error_message: "prior admission failed",
  }),
  record(5, { kind: "run", artifact_id: "statistical_model_spec" }, "rejected", {
    reason: "inputs missing",
  }),
  record(6, { kind: "run", artifact_id: "statistical_model_spec" }, "applied", {
    produced: [produced("statistical_model_spec", 1), produced("compiled_ssm", 1)],
  }),
  record(
    7,
    { kind: "write", artifact_id: "measurement_structure", provenance: "human" },
    "applied",
    {
      produced: [
        produced("measurement_structure", 2),
        produced("causal_design", 2),
        produced("structural_plan", 2),
      ],
      retracted: [{ artifact_id: "compiled_ssm", reason_ref: "stale-spec" }],
    },
  ),
];

describe("journalTicks", () => {
  it("keeps applied and raised moves, drops rejected attempts", () => {
    const ticks = journalTicks(JOURNAL);
    expect(ticks.map((tick) => tick.seq)).toEqual([1, 2, 3, 4, 6, 7]);
  });

  it("reads the installed version, derived co-outputs and retractions off the record", () => {
    const [, , design, raised, , rewrite] = journalTicks(JOURNAL);
    expect(design.version).toBe(1);
    expect(design.derived).toEqual(["causal_design", "structural_plan"]);
    expect(raised.version).toBeNull();
    expect(raised.error).toBe("prior admission failed");
    expect(rewrite.version).toBe(2);
    expect(rewrite.retracted).toEqual(["compiled_ssm"]);
  });
});

describe("latestSeq", () => {
  it("excludes failed attempts from model revisions", () => {
    expect(latestSeq(JOURNAL.slice(0, 5))).toBe(3);
  });
});
