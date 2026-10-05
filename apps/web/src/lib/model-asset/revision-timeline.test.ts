import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { TimelineRevision } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { latestSeq } from "./journal";
import { revisionTimeline } from "./revision-timeline";
import { timelineLayout, timelinePoint, timelineTickLabel } from "./timeline-presentation";

import { workbenchJournal } from "@/components/__fixtures__/workbench";

describe("argument timeline", () => {
  it("orders actions by execution, lanes them by what they produce and links served dependencies", () => {
    const records: TimelineRevision[] = [2, 5, 8, 6, 9, 13].map((seq, index) => {
      const fixture = fixtureValue(
        workbenchJournal.find((revision) => revision.record.seq === seq),
      );
      return {
        ...fixture,
        commit_id: String(index + 1).repeat(40),
        parent_ids: [String(index).repeat(40)],
        record: {
          ...fixture.record,
          seq: index + 1,
          ...(seq === 6
            ? {
                attempt: {
                  action: "fit",
                  request: null,
                  outcome: {
                    status: "raised",
                    error_type: "FitError",
                    error_message: "Fixture fit failed",
                    details: [],
                  },
                },
              }
            : {}),
        },
      };
    });
    const timeline = revisionTimeline(records, [
      { seq: 3, source_seq: 1, argument: "model", check: false },
      { seq: 3, source_seq: 2, argument: "panel", check: false },
      { seq: 4, source_seq: 1, argument: "model", check: false },
      { seq: 5, source_seq: 3, argument: "model", check: false },
      { seq: 6, source_seq: 2, argument: "left", check: false },
      { seq: 6, source_seq: 5, argument: "right", check: false },
      { seq: 9, source_seq: 2, argument: "panel", check: false },
    ]);
    expect(timeline.nodes.map((node) => [node.tick.record.seq, node.column, node.lane])).toEqual([
      [1, 0, 2],
      [2, 1, 1],
      [3, 2, 2],
      [4, 3, 2],
      [5, 4, 1],
      [6, 5, 0],
    ]);
    // Records outside the journal draw nothing; commit parents never become links.
    expect(
      timeline.links.map((link) => [link.from.tick.record.seq, link.to.tick.record.seq]),
    ).toEqual([
      [1, 3],
      [2, 3],
      [1, 4],
      [3, 5],
      [2, 6],
      [5, 6],
    ]);
    const layout = timelineLayout(timeline);
    const point = (index: number) => timelinePoint(layout, fixtureValue(timeline.nodes[index]));
    expect(point(5).y).toBeLessThan(point(4).y);
    expect(point(5).x).toBeGreaterThan(point(4).x);
    // The failed fit leaves the model track for a side track toward the data lane its inputs
    // come from, so the track later model actions use runs past it.
    expect(point(3).y).toBeLessThan(point(2).y);
    expect(point(3).y).toBeGreaterThan(point(4).y);
    expect(timelineTickLabel(fixtureValue(timeline.nodes[3]).tick)).toBe("fit · 4444444");
    expect(latestSeq(records)).toBe(5);
  });
});
