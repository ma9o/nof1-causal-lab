"use client";

import { useMemo, useState } from "react";
import type { JournalTick } from "./journal";
import { revisionBranch, revisionTimeline, type RevisionTimelineNode } from "./revision-timeline";
import { timelinePosition, timelineSize } from "./timeline-presentation";

export function useRevisionTimeline(
  ticks: JournalTick[],
  branches: Record<string, string>,
  selectedBranch: string,
  playhead: number,
) {
  const timeline = useMemo(() => revisionTimeline(ticks, branches), [ticks, branches]);
  const [expanded, setExpanded] = useState(false);
  const branch = useMemo(
    () => revisionBranch(timeline, branches[selectedBranch]),
    [timeline, branches, selectedBranch],
  );
  const visible = expanded ? timeline : branch;
  const { width, height } = timelineSize(
    visible.nodes.length,
    expanded ? timeline.lanes.length : 1,
  );
  const selectedNode = visible.nodes.find((node) => node.tick.seq === playhead);
  const position = (node: RevisionTimelineNode) => timelinePosition(node, expanded);
  const hasComparisons = timeline.nodes.some((node) => node.tick.seq !== playhead);

  return {
    timeline,
    expanded,
    setExpanded,
    visible,
    width,
    height,
    selectedNode,
    position,
    hasComparisons,
  };
}
