import type { StudyRevision } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { journalTicks, latestSeq } from "./journal";
import { revisionTimeline } from "./revision-timeline";
import { timelinePosition, timelineTickLabel } from "./timeline-presentation";

const record = (
  seq: number,
  action: StudyRevision["action"],
  status: StudyRevision["status"] = "applied",
): StudyRevision => ({
  seq,
  branch: "main",
  ts: "2026-09-30T12:00:00+00:00",
  action,
  inputs: {},
  status,
  diagnostics: {},
  messages: [],
  produced: [],
  retracted: [],
  trace_ids: [],
  commit_id: String(seq).repeat(40),
  parent_ids: [String(seq - 1).repeat(40)],
});

describe("argument timeline", () => {
  it("orders actions by execution, lanes them by what they produce and links served dependencies", () => {
    const records = [
      record(1, "edit_model"),
      record(2, "prepare_data"),
      record(3, "fit"),
      record(4, "fit", "raised"),
      record(5, "simulate"),
      record(6, "data_diff"),
    ];
    const timeline = revisionTimeline(journalTicks(records), [
      { seq: 3, source_seq: 1, argument: "model", check: false },
      { seq: 3, source_seq: 2, argument: "panel", check: false },
      { seq: 4, source_seq: 1, argument: "model", check: false },
      { seq: 5, source_seq: 3, argument: "model", check: false },
      { seq: 6, source_seq: 2, argument: "left", check: false },
      { seq: 6, source_seq: 5, argument: "right", check: false },
      { seq: 9, source_seq: 2, argument: "panel", check: false },
    ]);
    expect(timeline.nodes.map((node) => [node.tick.seq, node.column, node.lane])).toEqual([
      [1, 0, 2],
      [2, 1, 1],
      [3, 2, 2],
      [4, 3, 2],
      [5, 4, 1],
      [6, 5, 0],
    ]);
    // Records outside the journal draw nothing; commit parents never become links.
    expect(timeline.links.map((link) => [link.from.tick.seq, link.to.tick.seq])).toEqual([
      [1, 3],
      [2, 3],
      [1, 4],
      [3, 5],
      [2, 6],
      [5, 6],
    ]);
    const [comparison, simulation] = [timeline.nodes[5], timeline.nodes[4]];
    expect(timelinePosition(comparison).y).toBeLessThan(timelinePosition(simulation).y);
    expect(timelinePosition(comparison).x).toBeGreaterThan(timelinePosition(simulation).x);
    expect(timelineTickLabel(timeline.nodes[3].tick)).toBe("fit · 4444444");
    expect(latestSeq(records)).toBe(5);
  });
});
