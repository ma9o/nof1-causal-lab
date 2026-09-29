import { describe, expect, it } from "vitest";
import { branchedRevisionRecords } from "@/components/__fixtures__/revision-timeline";
import { journalTicks } from "./journal";
import { timelineTickLabel } from "./timeline-presentation";
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

    const failures = ["main", "alternative"].map((branch, index) => ({
      ...records[3],
      seq: 5 + index,
      commit_id: String(index).repeat(40),
      branch,
      parent_ids: [records[1].commit_id],
      status: "raised" as const,
      action: "fit" as const,
      produced: [],
      error_type: "ValueError",
      error_message: 'INTERNAL: CpuCallback error: Traceback: File "/Users/example/engine.py"',
    }));
    const withFailures = revisionTimeline(journalTicks([...records, ...failures]), branches);
    expect(
      revisionBranch(withFailures, branches.alternative).nodes.map((node) => node.tick.seq),
    ).toEqual([1, 2, 4, 6]);
    const failed = withFailures.nodes.at(-1)!;
    expect(timelineTickLabel(failed.tick)).toBe("fit · 1111111");
    expect(failed.tick.error).toBe(failures[1].error_message);
    expect(failed.modelRevision).toBeNull();
  });
});
