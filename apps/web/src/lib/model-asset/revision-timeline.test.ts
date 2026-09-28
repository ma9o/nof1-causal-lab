import { describe, expect, it } from "vitest";
import { branchedRevisionRecords } from "@/components/__fixtures__/revision-timeline";
import { journalTicks } from "./journal";
import { revisionBranch, revisionTimeline } from "./revision-timeline";

describe("Git study timeline", () => {
  it("uses backend commit parents and branch membership independently of model input pins", () => {
    const records = structuredClone(branchedRevisionRecords.slice(0, 4));
    records[3].branch = "alternative";
    records[2].produced[0].derived_from = { model: "f".repeat(40) };
    records[3].parent_ids = [records[1].commit_id];
    const branches = { main: records[2].commit_id, alternative: records[3].commit_id };
    const timeline = revisionTimeline(journalTicks(records), branches);
    expect(timeline.links.map((link) => [link.from.tick.seq, link.to.tick.seq])).toEqual([
      [1, 2],
      [2, 3],
      [2, 4],
    ]);
    const branch = revisionBranch(timeline, branches.alternative);
    expect(branch.nodes.map((node) => node.tick.seq)).toEqual([1, 2, 4]);
    expect(branch.nodes.map((node) => node.column)).toEqual([0, 1, 2]);
    expect(
      branch.links.every(
        (link) => branch.nodes.includes(link.from) && branch.nodes.includes(link.to),
      ),
    ).toBe(true);
    expect(revisionTimeline([], {}).nodes).toEqual([]);
  });
});
