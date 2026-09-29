import type { ScientificActionId } from "@nof1-causal-lab/api-types";
import type { JournalTick } from "./journal";
import type { RevisionTimelineNode } from "./revision-timeline";

/** Horizontal pitch between consecutive actions. */
export const COLUMN = 92;
/** Room on the left for the branch name at the start of each rail. */
const GUTTER = 72;
/** Height of the rail's centre line from the top of the strip. */
export const TOP = 18;
/** Space below a mark for the two-line action label. */
const LABEL_SPACE = 40;
/** A failed attempt changes nothing: it sits on a dead-end branch below its parent's rail. */
const FAILED_DROP = 30;
/** Vertical pitch between branch lanes when the full lineage is shown. */
export const ROW = FAILED_DROP + LABEL_SPACE + 18;

export type ActionGlyph = "dot" | "diamond" | "ring" | "arrow";

/** One mark and colour per scientific action; the tick's label repeats the name. */
export const ACTION_STYLE: Record<ScientificActionId, { color: string; glyph: ActionGlyph }> = {
  edit_model: { color: "#475569", glyph: "dot" },
  prepare_data: { color: "#0f766e", glyph: "diamond" },
  fit: { color: "#2563eb", glyph: "ring" },
  simulate: { color: "#7c3aed", glyph: "arrow" },
};
export const FAILED_COLOR = "#dc2626";

export function timelinePosition(node: RevisionTimelineNode, expanded: boolean) {
  return {
    x: GUTTER + node.column * COLUMN + COLUMN / 2,
    y: TOP + (expanded ? node.lane * ROW : 0) + (node.tick.status === "applied" ? 0 : FAILED_DROP),
  };
}

/**
 * Straight along a rail. A branch bends away right after its parent, so failed attempts from
 * one commit share a single dead-end branch while the rail continues to the next saved action.
 */
export function timelineLinkPath(from: { x: number; y: number }, to: { x: number; y: number }) {
  if (from.y === to.y) return `M ${from.x} ${from.y} H ${to.x}`;
  // Stay on the rail until clear of the parent's label, then bend within the gap.
  const start = from.x + Math.min(COLUMN * 0.4, (to.x - from.x) / 2);
  const end = Math.min(start + COLUMN * 0.45, to.x);
  const middle = (start + end) / 2;
  return `M ${from.x} ${from.y} H ${start} C ${middle} ${from.y}, ${middle} ${to.y}, ${end} ${to.y} H ${to.x}`;
}

export function timelineTickLabel(tick: JournalTick) {
  return `${tick.action} · ${tick.commitId.slice(0, 7)}`;
}

export function timelineSize(nodeCount: number, laneCount: number, hasFailures: boolean) {
  return {
    width: GUTTER + Math.max(1, nodeCount) * COLUMN + 12,
    height:
      TOP + (Math.max(1, laneCount) - 1) * ROW + (hasFailures ? FAILED_DROP : 0) + LABEL_SPACE,
  };
}
