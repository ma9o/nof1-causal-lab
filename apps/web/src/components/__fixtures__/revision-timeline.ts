export const fixtureOid = (n: number) => n.toString(16).padStart(40, "a");
import type { ArtifactRecord, StudyRevision } from "@nof1-causal-lab/api-types";
function produced(
  artifact_id: ArtifactRecord["artifact_id"],
  revision: string,
  derived_from: ArtifactRecord["derived_from"] = {},
): ArtifactRecord {
  return {
    artifact_id,
    revision,
    derived_from,
    model_inputs: {},
    consumed_model_inputs: {},
    produced_by: null,
    created_at: "2026-09-16T12:00:00Z",
  };
}
function entry(
  seq: number,
  action: Pick<StudyRevision, "action" | "inputs" | "operation_id">,
  output: ArtifactRecord[] = [],
  diagnostics: StudyRevision["diagnostics"] = {},
): StudyRevision {
  return {
    seq,
    commit_id: fixtureOid(100 + seq),
    parent_ids: [fixtureOid(99 + seq)],
    branch: "main",
    ts: `2026-09-16T12:${String(seq).padStart(2, "0")}:00Z`,
    ...action,
    produced: output,
    diagnostics,
    messages: [],
    status: "applied",
    retracted: [],
    trace_ids: [],
    reason: null,
    error_type: null,
    error_message: null,
    resume: null,
  };
}
/** Illustrative action history only: no fabricated model definitions or scientific results. */
export const branchedRevisionRecords: StudyRevision[] = [
  entry(
    1,
    {
      action: "prepare_data",
      operation_id: "raw_data",
      inputs: {},
    },
    [produced("raw_data", fixtureOid(1))],
  ),
  entry(
    2,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: null },
    },
    [produced("model", fixtureOid(1))],
  ),
  entry(
    3,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: fixtureOid(1) },
    },
    [produced("model", fixtureOid(2), { model: fixtureOid(1) })],
  ),
  entry(
    4,
    {
      action: "prepare_data",
      operation_id: "measurements",
      inputs: { model_revision: fixtureOid(2), raw_data_revision: fixtureOid(1) },
    },
    [produced("panel", fixtureOid(1), { model: fixtureOid(2), raw_data: fixtureOid(1) })],
  ),
  entry(
    5,
    {
      action: "fit",
      operation_id: "posterior",
      inputs: { model_revision: fixtureOid(2), panel_revision: fixtureOid(1) },
    },
    [produced("model", fixtureOid(3), { model: fixtureOid(2), panel: fixtureOid(1) })],
  ),
  entry(
    6,
    {
      action: "simulate",
      operation_id: "simulate",
      inputs: { model_revision: fixtureOid(3) },
    },
    [],
    {
      input_pins: { model: fixtureOid(3) },
    },
  ),
  entry(
    7,
    {
      action: "fit",
      operation_id: "posterior",
      inputs: { model_revision: fixtureOid(2), panel_revision: fixtureOid(1) },
    },
    [produced("model", fixtureOid(4), { model: fixtureOid(2), panel: fixtureOid(1) })],
  ),
  entry(
    8,
    {
      action: "simulate",
      operation_id: "simulate",
      inputs: { model_revision: fixtureOid(4) },
    },
    [],
    {
      input_pins: { model: fixtureOid(4) },
    },
  ),
  entry(
    9,
    {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: fixtureOid(4) },
    },
    [produced("model", fixtureOid(5), { model: fixtureOid(4) })],
  ),
  entry(
    10,
    {
      action: "simulate",
      operation_id: "simulate",
      inputs: { model_revision: fixtureOid(3) },
    },
    [],
    {
      input_pins: { model: fixtureOid(3) },
    },
  ),
  {
    ...entry(11, {
      action: "edit_model",
      operation_id: null,
      inputs: { expected_revision: fixtureOid(2) },
    }),
    status: "rejected",
    reason: "Model revision conflict: expected 2, current 5",
  },
];
