import { describe, expect, it } from "vitest";
import {
  cursorTimestampMs,
  parseTransitionProgressEvent,
  progressPollIntervalMs,
} from "./use-run-events";

describe("progressPollIntervalMs", () => {
  it("keeps polling idle viewers so externally started runs are discovered", () => {
    expect(progressPollIntervalMs(false)).toBe(10_000);
    expect(progressPollIntervalMs(true)).toBe(2_000);
  });
});

describe("cursorTimestampMs", () => {
  it("parses the nanosecond prefix of an event cursor", () => {
    const cursor = "01750000000000000000-abcd1234.json";
    expect(cursorTimestampMs(cursor)).toBe(Math.floor(1_750_000_000_000_000_000 / 1_000_000));
  });

  it("returns undefined for malformed cursors", () => {
    expect(cursorTimestampMs("not-a-cursor.json")).toBeUndefined();
  });
});

describe("parseTransitionProgressEvent", () => {
  const cursor = "01750000000000000000-abcd1234.json";

  it("parses running events", () => {
    const event = parseTransitionProgressEvent({
      event: "nof1-causal-lab.transition.running",
      transition_id: "latent_structure",
      cursor,
    } as const);

    expect(event).toEqual({
      artifactId: "latent_structure",
      status: "running",
      eventTime: cursorTimestampMs(cursor),
      error: undefined,
    });
  });

  it("parses failed events with error detail", () => {
    const event = parseTransitionProgressEvent({
      event: "nof1-causal-lab.transition.failed",
      transition_id: "measurements",
      error: { type: "RuntimeError", message: "boom" },
      cursor,
    } as const);

    expect(event?.status).toBe("failed");
    expect(event?.error).toEqual({ type: "RuntimeError", message: "boom" });
  });

  it("ignores non-transition-progress events", () => {
    expect(
      parseTransitionProgressEvent({
        event: "nof1-causal-lab.extraction.worker",
        worker_id: 1,
        state: "running",
        n_windows: 4,
        cursor,
      } as const),
    ).toBeNull();
  });

  it("ignores malformed payloads", () => {
    expect(
      parseTransitionProgressEvent({
        event: "nof1-causal-lab.transition.running",
        transition_id: "not-an-artifact",
        cursor,
      } as const),
    ).toBeNull();
  });
});
