import { describe, expect, it } from "vitest";
import {
  MODEL_SPEC_ADMISSION_EVENT_PREFIX,
  EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE,
  applyModelSpecAdmissionEvent,
  parseModelSpecAdmissionEvent,
} from "@/lib/model-spec-admission-runtime";
import { buildTimeline } from "./presentation";

function replayModelSpecAdmissionEvents(
  records: Parameters<typeof parseModelSpecAdmissionEvent>[0][],
) {
  return records.reduce((state, record) => {
    const event = parseModelSpecAdmissionEvent(record)!;
    return applyModelSpecAdmissionEvent(state, event);
  }, EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE);
}

describe("admission presentation", () => {
  it("presents attempts and coupled rechecks using the modes recorded by the server", () => {
    const event = (type: string, payload: Record<string, unknown>) => ({
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}${type}`,
      payload,
    });
    const state = replayModelSpecAdmissionEvents([
      event("plan", {
        constructs: [{ name: "sleep" }, { name: "stress" }],
        edges: [],
        max_attempts: 4,
      }),
      event("construct_report", {
        name: "sleep",
        attempt: 1,
        outcome: "REVISE",
        admitted: false,
        results: [{ check: "C1a finiteness", passed: false, mode: "soft" }],
      }),
      event("construct_report", {
        name: "stress",
        attempt: 1,
        outcome: "ADMITTED",
        admitted: true,
        results: [],
        coupled_recheck: {
          constructs: ["sleep"],
          closing_edges: ["stress->sleep"],
          results: [{ check: "New server check", passed: false, mode: "hard" }],
          timings: [],
        },
      }),
    ]);
    const timeline = buildTimeline(state, state.constructs[0]);
    expect(timeline).toMatchObject([
      { kind: "attempt", attempt: 1, status: "revising" },
      { kind: "recheck", originator: "stress", closingEdges: ["stress->sleep"], status: "blocked" },
    ]);
    expect(timeline[0].results).toBe(state.constructs[0].reports[0].results);
  });

  it("does not infer a missing check mode from a known scientific check name", () => {
    expect(() =>
      replayModelSpecAdmissionEvents([
        {
          event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
          payload: {
            name: "sleep",
            outcome: "REVISE",
            results: [{ check: "C1a finiteness", passed: false }],
          },
        },
      ]),
    ).toThrow("server-provided mode");
  });
});
