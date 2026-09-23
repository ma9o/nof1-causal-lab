"use client";

import { useMemo, useState } from "react";
import type { ModelSpecAdmissionReplayState } from "@/lib/model-spec-admission-runtime";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import {
  buildTimeline,
  getFeaturedConstruct,
  progressCounts,
  DAG_NODE_W,
  DAG_NODE_H,
} from "./presentation";

/** Own report selection and following the live stream independently of rendering. */
export function useAdmissionView(state: ModelSpecAdmissionReplayState | null) {
  const [selectedConstructName, setSelectedConstructName] = useState<string | null>(null);
  const [reportSelection, setReportSelection] = useState<{
    selectedKey: string | null;
    latestKeyAtSelection: string | null;
  }>({ selectedKey: null, latestKeyAtSelection: null });
  const counts = state ? progressCounts(state) : { admitted: 0, revising: 0, blocked: 0, total: 0 };
  const liveFeaturedConstruct = state ? getFeaturedConstruct(state) : null;
  const featuredConstruct =
    state?.constructs.find((construct) => construct.name === selectedConstructName) ??
    liveFeaturedConstruct;
  const timeline = useMemo(
    () => (state && featuredConstruct ? buildTimeline(state, featuredConstruct) : []),
    [state, featuredConstruct],
  );
  const latestEntry = timeline[timeline.length - 1] ?? null;
  const explicitlySelectedEntry =
    timeline.find((entry) => entry.key === reportSelection.selectedKey) ?? null;
  const followsLatest =
    reportSelection.selectedKey === null ||
    reportSelection.selectedKey === reportSelection.latestKeyAtSelection;
  const selectedEntry = followsLatest ? latestEntry : (explicitlySelectedEntry ?? latestEntry);
  const progress = counts.total > 0 ? Math.round((counts.admitted / counts.total) * 100) : 0;
  const handleSelectConstruct = (constructName: string) => {
    setSelectedConstructName(constructName);
    setReportSelection({ selectedKey: null, latestKeyAtSelection: null });
  };
  const handleSelectEntry = (key: string) => {
    setReportSelection({ selectedKey: key, latestKeyAtSelection: latestEntry?.key ?? null });
  };

  return {
    counts,
    featuredConstruct,
    timeline,
    selectedEntry,
    progress,
    handleSelectConstruct,
    handleSelectEntry,
  };
}

export function useAdmissionGraph(state: ModelSpecAdmissionReplayState) {
  // Key the ELK layout on the topology only (construct names + edges) so status
  // ticks recolor in place instead of triggering a full re-layout. Construct and
  // edge endpoints are `[a-z0-9_]` identifiers, so `|`/`>` are safe delimiters.
  const nodeKey = state.constructs.map((construct) => construct.name).join("|");
  const edgeKey = (state.plan?.edges ?? []).map((edge) => `${edge.cause}>${edge.effect}`).join("|");
  const graph = useMemo<DagGraphInput>(() => {
    const names = nodeKey ? nodeKey.split("|") : [];
    const nameSet = new Set(names);
    return {
      direction: "RIGHT",
      nodes: names.map((id) => ({ id, width: DAG_NODE_W, height: DAG_NODE_H })),
      edges: (edgeKey ? edgeKey.split("|") : []).flatMap((pair, index) => {
        const [source, target] = pair.split(">");
        return nameSet.has(source) && nameSet.has(target)
          ? [{ id: `edge-${index}`, source, target }]
          : [];
      }),
    };
  }, [nodeKey, edgeKey]);

  const { nodes, edges, width, height, isLayouting } = useDagLayout(graph);
  const geoByName = new Map(nodes.map((node) => [node.id, node]));

  const constructByName = new Map(state.constructs.map((construct) => [construct.name, construct]));
  const orderByName = new Map(state.constructs.map((construct, index) => [construct.name, index]));
  return { edges, width, height, isLayouting, geoByName, constructByName, orderByName };
}
