import { describe, expect, it } from "vitest";
import type { PipelineSectionId } from "@nof1-causal-lab/api-types";
import {
  applyTransitionUpdate,
  initialProgress,
  restartTransitionAttempt,
  type PipelineProgress,
} from "./pipeline-progress";

const TEST_ORDER: PipelineSectionId[] = [
  "raw_data",
  "simulate",
  "simulated_measurements",
  "measurements",
  "validation_report",
  "posterior",
];

function applyUpdate(
  prev: PipelineProgress | undefined,
  artifactId: PipelineSectionId,
  status: "pending" | "running" | "completed" | "failed",
  eventTime?: number,
  errorMessage?: string,
): PipelineProgress {
  return applyTransitionUpdate(prev, artifactId, status, eventTime, errorMessage, TEST_ORDER);
}

function restartAttempt(
  prev: PipelineProgress | undefined,
  artifactId: PipelineSectionId,
  eventTime?: number,
): PipelineProgress {
  return restartTransitionAttempt(prev, artifactId, eventTime, TEST_ORDER);
}

describe("initialProgress", () => {
  it("starts every artifact view pending with no errors", () => {
    const progress = initialProgress(TEST_ORDER);

    expect(progress.artifacts["raw_data"]).toBe("pending");
    expect(progress.artifacts["posterior"]).toBe("pending");
    expect(progress.transitionErrors).toEqual({});
    expect(progress.transitionOrder).toEqual(TEST_ORDER);
    expect(progress.runningTransitions).toEqual([]);
    expect(progress.isComplete).toBe(false);
    expect(progress.isFailed).toBe(false);
  });
});

describe("applyTransitionUpdate", () => {
  it("records running and completion timings", () => {
    let progress: PipelineProgress | undefined;
    progress = applyUpdate(progress, "raw_data", "running", 1_000);
    progress = applyUpdate(progress, "raw_data", "completed", 5_000);

    expect(progress.artifacts["raw_data"]).toBe("completed");
    expect(progress.timings["raw_data"]).toEqual({
      startedAt: 1_000,
      completedAt: 5_000,
    });
  });

  it("never regresses a completed transition back to running", () => {
    let progress = applyUpdate(undefined, "raw_data", "completed", 5_000);
    progress = applyUpdate(progress, "raw_data", "running", 6_000);

    expect(progress.artifacts["raw_data"]).toBe("completed");
  });

  it("tracks every running transition in machine order", () => {
    let progress = applyUpdate(undefined, "raw_data", "completed", 1_000);
    progress = applyUpdate(progress, "measurements", "running", 2_000);
    progress = applyUpdate(progress, "simulate", "running", 3_000);

    expect(progress.runningTransitions).toEqual(["simulate", "measurements"]);
  });

  it("stores the failure detail and flips isFailed", () => {
    const progress = applyUpdate(
      undefined,
      "simulate",
      "failed",
      2_000,
      "SchemaValidationError: bad payload",
    );

    expect(progress.artifacts["simulate"]).toBe("failed");
    expect(progress.transitionErrors["simulate"]).toBe("SchemaValidationError: bad payload");
    expect(progress.isFailed).toBe(true);
  });

  it("restarts a failed transition on a new running attempt", () => {
    let progress = applyUpdate(undefined, "simulate", "failed", 2_000, "boom");
    progress = restartAttempt(progress, "simulate", 3_000);

    expect(progress.artifacts["simulate"]).toBe("running");
    expect(progress.timings["simulate"]).toEqual({ startedAt: 3_000 });
    expect(progress.transitionErrors["simulate"]).toBeUndefined();
    expect(progress.isFailed).toBe(false);
  });

  it("restarts a completed transition when its inputs are recomputed", () => {
    let progress = applyUpdate(undefined, "simulated_measurements", "completed", 2_000);
    progress = restartAttempt(progress, "simulated_measurements", 9_000);
    progress = applyUpdate(progress, "simulated_measurements", "completed", 12_000);

    expect(progress.artifacts["simulated_measurements"]).toBe("completed");
    expect(progress.timings["simulated_measurements"]).toEqual({
      startedAt: 9_000,
      completedAt: 12_000,
    });
  });

  it("keeps the original start when the transition is already running", () => {
    let progress = applyUpdate(undefined, "measurements", "running", 1_000);
    progress = restartAttempt(progress, "measurements", 4_000);

    expect(progress.timings["measurements"]).toEqual({
      startedAt: 1_000,
      completedAt: undefined,
    });
  });

  it("marks the pipeline complete when every artifact view completed", () => {
    let progress: PipelineProgress | undefined;
    for (const artifactId of TEST_ORDER) {
      progress = applyUpdate(progress, artifactId, "completed");
    }

    expect(progress?.isComplete).toBe(true);
  });

  it("flips a failed transition to completed when a later attempt succeeds (latest wins)", () => {
    let progress = applyUpdate(undefined, "simulate", "failed", 2_000, "boom");
    progress = applyUpdate(progress, "simulate", "completed", 5_000);

    expect(progress.artifacts["simulate"]).toBe("completed");
  });

  it("flips a completed transition to failed on a later failure", () => {
    let progress = applyUpdate(undefined, "simulate", "completed", 2_000);
    progress = applyUpdate(progress, "simulate", "failed", 5_000, "regressed");

    expect(progress.artifacts["simulate"]).toBe("failed");
    expect(progress.transitionErrors["simulate"]).toBe("regressed");
  });

  it("ignores a stale terminal signal that predates the recorded outcome", () => {
    let progress = applyUpdate(undefined, "simulate", "failed", 5_000, "boom");
    progress = applyUpdate(progress, "simulate", "completed", 3_000);

    expect(progress.artifacts["simulate"]).toBe("failed");
  });

  it("ignores an earlier failed journal record after a newer attempt starts", () => {
    let progress = applyUpdate(undefined, "posterior", "failed", 2_000, "old failure");
    progress = restartAttempt(progress, "posterior", 5_000);
    progress = applyUpdate(progress, "posterior", "failed", 2_000, "old failure");

    expect(progress.artifacts["posterior"]).toBe("running");
    expect(progress.transitionErrors["posterior"]).toBeUndefined();
    expect(progress.isFailed).toBe(false);
  });
});
