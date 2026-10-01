import type { ActionId, RecordDependency } from "@nof1-causal-lab/api-types";
import type { JournalTick } from "./journal";

/** Lanes group actions by what they produce: comparisons read histories, data makes them. */
export const TIMELINE_LANES: ReadonlyArray<{ name: string; actions: readonly ActionId[] }> = [
  { name: "Comparisons", actions: ["data_diff"] },
  { name: "Data", actions: ["prepare_data", "simulate"] },
  { name: "Model", actions: ["edit_model", "fit"] },
];

export interface RevisionTimelineNode {
  tick: JournalTick;
  column: number;
  lane: number;
}

export interface RevisionTimelineLink {
  from: RevisionTimelineNode;
  to: RevisionTimelineNode;
  argument: string;
  check: boolean;
}

/** Presentation only: execution order gives columns; the served dependencies give links. */
export function revisionTimeline(
  ticks: readonly JournalTick[],
  dependencies: readonly RecordDependency[],
) {
  const nodes: RevisionTimelineNode[] = ticks.map((tick, column) => ({
    tick,
    column,
    lane: TIMELINE_LANES.findIndex((lane) => lane.actions.includes(tick.action)),
  }));
  const bySeq = new Map(nodes.map((node) => [node.tick.seq, node]));
  const links: RevisionTimelineLink[] = dependencies.flatMap((dependency) => {
    const from = bySeq.get(dependency.source_seq);
    const to = bySeq.get(dependency.seq);
    return from && to ? [{ from, to, argument: dependency.argument, check: dependency.check }] : [];
  });
  return { nodes, links };
}
