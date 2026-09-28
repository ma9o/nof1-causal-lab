import type { RevisionTimelineNode } from "./revision-timeline";
import { actionLabel } from "./selection";

export const COLUMN = 144;
export const ROW = 84;
const GUTTER = 80;
export const TOP = 18;
const COLORS = ["#475569", "#2563eb", "#0f766e", "#9333ea", "#b45309"];
export const colorFor = (lane: number) => COLORS[lane % COLORS.length];

export function timelinePosition(node: RevisionTimelineNode, expanded: boolean) {
  return {
    x: GUTTER + node.column * COLUMN + COLUMN / 2,
    y: TOP + (expanded ? node.lane * ROW : 0) + (node.modelRevision == null ? 22 : 0),
  };
}

export function timelineLinkPath(from: { x: number; y: number }, to: { x: number; y: number }) {
  const elbow = from.x + COLUMN / 2;
  const direction = Math.sign(to.y - from.y);
  const radius = Math.min(12, Math.abs(to.y - from.y) / 2);
  return from.y === to.y
    ? `M ${from.x} ${from.y} H ${to.x}`
    : `M ${from.x} ${from.y} H ${elbow - radius} Q ${elbow} ${from.y} ${elbow} ${from.y + direction * radius} V ${to.y - direction * radius} Q ${elbow} ${to.y} ${elbow + radius} ${to.y} H ${to.x}`;
}

export function timelineNodeLabel(node: RevisionTimelineNode) {
  return `${actionLabel(node.tick.action)} · version ${node.tick.seq}`;
}

export function timelineSize(nodeCount: number, laneCount: number) {
  return {
    width: GUTTER + Math.max(1, nodeCount) * COLUMN + 12,
    height: TOP + Math.max(1, laneCount) * ROW + 16,
  };
}
