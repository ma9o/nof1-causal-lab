import { describe, expect, it } from "vitest";
import { demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { availableGraphLayers } from "./layered-causal-graph-model";

describe("semantic graph layers", () => {
  it("uses the facts available at each committed revision", () => {
    expect(availableGraphLayers(demoSnapshotAt(0))).toEqual([]);
    expect(availableGraphLayers(demoSnapshotAt(3))).toEqual(["structure"]);
    expect(availableGraphLayers(demoSnapshotAt(4))).toEqual(["structure", "measurement", "design"]);
    expect(availableGraphLayers(demoSnapshotAt(8))).toEqual([
      "structure",
      "measurement",
      "design",
      "specification",
      "fit",
    ]);
  });
});
