import type { EpisodeStatus, StudyRevision } from "@/lib/server/episode-runs";
import { MACHINE_DESCRIPTION } from "@nof1-causal-lab/api-types";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
vi.mock("@/lib/server/episode-runs", () => ({
  getMachineDescription: vi.fn(),
  getEpisodeStatus: vi.fn(),
  getEpisodeTimeline: vi.fn(),
}));
vi.mock("@/lib/server/artifacts", () => ({
  ArtifactNotFoundError: class ArtifactNotFoundError extends Error {},
  readArtifactJson: vi.fn(),
}));
import { readArtifactJson } from "@/lib/server/artifacts";
import {
  getEpisodeStatus,
  getEpisodeTimeline,
  getMachineDescription,
} from "@/lib/server/episode-runs";
import { buildAnalysisManifest } from "./_shared";
function emptyStatus(workspaceId: string): EpisodeStatus {
  return {
    workspace_id: workspaceId,
    branch: "main",
    seq: 0,
    state: { current: {} },
    artifacts: [],
    actions: ["edit_model", "prepare_data", "fit", "simulate"],
    running: null,
  };
}
function statusWithQuestion(workspaceId: string, revision = "a".repeat(40)): EpisodeStatus {
  return {
    ...emptyStatus(workspaceId),
    state: {
      current: {
        model: {
          artifact_id: "model",
          revision,
          derived_from: {},
          model_inputs: {},
          consumed_model_inputs: {},
          produced_by: null,
          created_at: "2026-07-01T00:00:00+00:00",
        },
      },
    },
  };
}
function transition(
  overrides: Partial<StudyRevision> & Pick<StudyRevision, "seq" | "ts" | "action" | "status">,
): StudyRevision {
  return {
    commit_id: String(overrides.seq).padStart(40, "a"),
    parent_ids: [],
    branch: "main",
    reason: null,
    error_type: null,
    error_message: null,
    diagnostics: {},
    messages: [],
    produced: [],
    inputs: {},
    retracted: [],
    trace_ids: [],
    resume: null,
    ...overrides,
  };
}
describe("buildAnalysisManifest", () => {
  beforeEach(() => {
    vi.mocked(getMachineDescription).mockResolvedValue(MACHINE_DESCRIPTION);
  });
  afterEach(() => {
    vi.clearAllMocks();
  });
  it("returns null when the episode journal is empty", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(emptyStatus("user-1"));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      branches: {},
      workspace_id: "user-1",
      transitions: [],
    });
    await expect(buildAnalysisManifest("user-1")).resolves.toBeNull();
  });
  it("builds transition executions from journal run transitions", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(statusWithQuestion("user-1"));
    vi.mocked(readArtifactJson).mockResolvedValue({ question: "Does exercise help sleep?" });
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      branches: {},
      workspace_id: "user-1",
      transitions: [
        transition({
          seq: 1,
          commit_id: "a".repeat(40),
          parent_ids: [],
          ts: "2026-07-01T00:00:00+00:00",
          ...{
            action: "edit_model",
            operation_id: null,
            inputs: {},
          },
          status: "applied",
        }),
        transition({
          seq: 2,
          ts: "2026-07-01T00:01:00+00:00",
          ...{
            action: "prepare_data",
            operation_id: "raw_data",
            inputs: {},
          },
          status: "applied",
        }),
        transition({
          seq: 3,
          ts: "2026-07-01T00:02:00+00:00",
          ...{
            action: "simulate",
            operation_id: "simulate",
            inputs: {},
          },
          status: "raised",
          error_type: "SchemaValidationError",
          error_message: "simulate payload failed validation",
        }),
        transition({
          seq: 4,
          ts: "2026-07-01T00:03:00+00:00",
          ...{
            action: "simulate",
            operation_id: "simulate",
            inputs: {},
          },
          status: "rejected",
          reason: "simulate requires artifacts that do not exist: model",
        }),
      ],
    });
    const manifest = await buildAnalysisManifest("user-1");
    expect(manifest).not.toBeNull();
    expect(manifest?.createdAt).toBe("2026-07-01T00:00:00+00:00");
    expect(manifest?.question).toBe("Does exercise help sleep?");
    expect(manifest?.transitionOrder).toEqual(
      expect.arrayContaining([
        "raw_data",
        "measurements",
        "validation_report",
        "posterior",
        "simulate",
        "simulated_measurements",
      ]),
    );
    expect(vi.mocked(readArtifactJson)).toHaveBeenCalledWith(
      "user-1",
      "model",
      "model",
      "a".repeat(40),
    );
    expect(manifest?.transitionRuns["raw_data"]).toEqual({
      execution: {
        stateType: "COMPLETED",
        startTime: "2026-07-01T00:01:00+00:00",
        endTime: "2026-07-01T00:01:00+00:00",
      },
    });
    // Raised run marks the transition failed; the later rejected attempt never executed.
    expect(manifest?.transitionRuns["simulate"]?.execution?.stateType).toBe("FAILED");
    expect(manifest?.transitionRuns["measurements"]).toEqual({ execution: null });
  });
  it("prefers the latest run attempt per artifact", async () => {
    vi.mocked(getEpisodeStatus).mockResolvedValue(emptyStatus("user-1"));
    vi.mocked(getEpisodeTimeline).mockResolvedValue({
      branches: {},
      workspace_id: "user-1",
      transitions: [
        transition({
          seq: 1,
          commit_id: "a".repeat(40),
          parent_ids: [],
          ts: "2026-07-01T00:00:00+00:00",
          ...{
            action: "prepare_data",
            operation_id: "raw_data",
            inputs: {},
          },
          status: "raised",
          error_type: "RuntimeError",
        }),
        transition({
          seq: 2,
          ts: "2026-07-01T00:05:00+00:00",
          ...{
            action: "prepare_data",
            operation_id: "raw_data",
            inputs: {},
          },
          status: "applied",
        }),
      ],
    });
    const manifest = await buildAnalysisManifest("user-1");
    expect(manifest?.transitionRuns["raw_data"]?.execution?.stateType).toBe("COMPLETED");
  });
});
