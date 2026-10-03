import type { ActionId } from "@nof1-causal-lab/api-types";
import type { StudyRevision } from "@nof1-causal-lab/api-types";
import { TIMELINE_LANES, type RevisionTimelineNode } from "./revision-timeline";

/** Horizontal pitch between consecutive actions. */
export const COLUMN = 92;
/** Room on the left for the lane names. */
export const GUTTER = 96;
/** Height of the first lane's centre line from the top of the strip. */
const TOP = 18;
/** Vertical pitch between lanes; links run through the gaps. */
const LANE = 30;
/** One action per column, so every label shares one row under the last lane. */
export const LABEL_TOP = TOP + (TIMELINE_LANES.length - 1) * LANE + 18;

export type ActionGlyph = "dot" | "diamond" | "ring" | "arrow";

/** One mark and colour per scientific action; the tick's label repeats the name. */
export const ACTION_STYLE: Record<ActionId, { color: string; glyph: ActionGlyph }> = {
  set_question: { color: "#b45309", glyph: "diamond" },
  edit_model: { color: "#475569", glyph: "dot" },
  prepare_data: { color: "#0f766e", glyph: "diamond" },
  fit: { color: "#2563eb", glyph: "ring" },
  data_diff: { color: "#0369a1", glyph: "ring" },
  simulate: { color: "#7c3aed", glyph: "arrow" },
};
export const FAILED_COLOR = "#dc2626";

export const laneY = (lane: number) => TOP + lane * LANE;

export function timelinePosition(node: RevisionTimelineNode) {
  return { x: GUTTER + node.column * COLUMN + COLUMN / 2, y: laneY(node.lane) };
}

export function timelineSize(nodeCount: number) {
  return { width: GUTTER + Math.max(1, nodeCount) * COLUMN + 12, height: LABEL_TOP + 32 };
}

/**
 * Leaves the producer's lane vertically, runs along the gap between lanes, and enters the
 * consumer's lane vertically; consumers always sit to the right of their producers.
 */
export function timelineLinkPath(from: { x: number; y: number }, to: { x: number; y: number }) {
  if (from.y === to.y) return `M ${from.x} ${from.y} H ${to.x}`;
  const mid = (from.y + to.y) / 2;
  const bend = Math.min(18, (to.x - from.x) / 2);
  return `M ${from.x} ${from.y} C ${from.x} ${mid}, ${from.x} ${mid}, ${from.x + bend} ${mid} H ${to.x - bend} C ${to.x} ${mid}, ${to.x} ${mid}, ${to.x} ${to.y}`;
}

export function timelineTickLabel(tick: StudyRevision) {
  return `${tick.record.attempt.action} · ${tick.commit_id.slice(0, 7)}`;
}
