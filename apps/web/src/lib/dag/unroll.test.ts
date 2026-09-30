import { describe, expect, it } from "vitest";
import { unrollCausalLinks } from "@/lib/dag/unroll";

describe("unrollCausalLinks", () => {
  it("routes dynamic causes from history and static causes from their single node", () => {
    const built = unrollCausalLinks(
      [
        { cause: "varying", effect: "outcome" },
        { cause: "another_dynamic", effect: "outcome" },
        { cause: "stable", effect: "outcome" },
      ],
      new Set(["varying", "another_dynamic"]),
    );

    expect(built.ghosts).toEqual(["varying__p", "another_dynamic__p"]);
    expect(built.edges).toEqual([
      {
        cause: "varying",
        effect: "outcome",
        crossSlice: true,
        source: "varying__p",
        target: "outcome",
      },
      {
        cause: "another_dynamic",
        effect: "outcome",
        crossSlice: true,
        source: "another_dynamic__p",
        target: "outcome",
      },
      {
        cause: "stable",
        effect: "outcome",
        crossSlice: false,
        source: "stable",
        target: "outcome",
      },
    ]);
  });
});
