import type { ArtifactFreshness } from "@/lib/api/analysis";
import { describe, expect, it } from "vitest";
import { groupStaleArtifactsByProducer } from "./artifact-staleness";

function artifact(overrides: Partial<ArtifactFreshness>): ArtifactFreshness {
  return {
    artifact_id: "model",
    exists: true,
    stale: false,
    retracted: false,
    revision: "0000000000000000000000000000000000000001",

    produced_by: "run:posterior",
    ...overrides,
  };
}

describe("groupStaleArtifactsByProducer", () => {
  it("groups stale existing artifacts by producing artifact", () => {
    const report = [
      artifact({
        artifact_id: "model",
        stale: true,
        produced_by: "run:posterior",
      }),
      artifact({
        artifact_id: "panel",
        stale: true,
        produced_by: "run:measurements",
      }),
      artifact({ artifact_id: "panel", stale: false, produced_by: "run:measurements" }),
    ];

    expect(groupStaleArtifactsByProducer(report)).toEqual({
      posterior: ["model"],
      measurements: ["panel"],
    });
  });

  it("ignores absent artifacts even when flagged stale", () => {
    const report = [
      artifact({
        artifact_id: "panel",
        exists: false,
        stale: true,
        produced_by: "run:measurements",
      }),
    ];

    expect(groupStaleArtifactsByProducer(report)).toEqual({});
  });

  it("ignores root artifacts with no producer", () => {
    const report = [artifact({ artifact_id: "model", stale: true, produced_by: null })];

    expect(groupStaleArtifactsByProducer(report)).toEqual({});
  });
});
