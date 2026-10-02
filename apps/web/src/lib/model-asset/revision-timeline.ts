import type { ActionId, RecordDependency } from "@nof1-causal-lab/api-types";
import type { StudyRevision } from "@nof1-causal-lab/api-types";

/** Lanes group actions by what they produce: comparisons read histories, data makes them. */
export const TIMELINE_LANES: ReadonlyArray<{ name: string; actions: readonly ActionId[] }> = [
  { name: "Comparisons", actions: ["data_diff"] },
  { name: "Data", actions: ["prepare_data", "simulate"] },
  { name: "Model", actions: ["edit_model", "fit"] },
];

export interface RevisionTimelineNode {
  tick: StudyRevision;
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
  ticks: readonly StudyRevision[],
  dependencies: readonly RecordDependency[],
) {
  const nodes: RevisionTimelineNode[] = ticks.map((tick, column) => ({
    tick,
    column,
    lane: TIMELINE_LANES.findIndex((lane) => lane.actions.includes(tick.record.attempt.action)),
  }));
  const bySeq = new Map(nodes.map((node) => [node.tick.record.seq, node]));
  const links: RevisionTimelineLink[] = dependencies.flatMap((dependency) => {
    const from = bySeq.get(dependency.source_seq);
    const to = bySeq.get(dependency.seq);
    return from && to ? [{ from, to, argument: dependency.argument, check: dependency.check }] : [];
  });
  return { nodes, links };
}
