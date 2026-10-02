"use client";

import type { ModelSnapshot, StudyRevision } from "@nof1-causal-lab/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { indexModel } from "./entities";
import { latestSeq } from "./journal";
import type { ScopeContext } from "./scope";
import type { EntitySelection } from "./selection";

export type SnapshotReader = (
  commitId: string | undefined,
  branch: string,
) => {
  data: ModelSnapshot | undefined;
  error: Error | null;
  isPlaceholderData?: boolean;
};

export function useWorkbenchSnapshots(
  attempts: readonly StudyRevision[],
  branches: Readonly<Partial<Record<string, string>>>,
  useSnapshot: SnapshotReader,
) {
  const [playheadOverride, viewAt] = useState<number | null>(null);
  const latest = latestSeq(attempts);
  const branch = attempts.find((record) => record.record.seq === latest)?.record.branch ?? "main";
  const playhead = playheadOverride ?? latest;
  const record = attempts.find((item) => item.record.seq === playhead);
  // Read-only leaves and failed attempts inspect their unchanged parent state.
  const commitId = record
    ? record.record.attempt.outcome.status === "applied" &&
      record.record.attempt.action !== "data_diff"
      ? record.commit_id
      : record.parent_ids[0]
    : branches[branch];
  const selected = useSnapshot(commitId, record?.record.branch ?? branch);
  const current = useSnapshot(branches[branch], branch);
  return { selected, current, viewAt, focusSeq: playhead };
}

interface WorkbenchOptions {
  workspaceId: string;
  question: string | undefined;
  attempts: readonly StudyRevision[];
  model: ModelSnapshot;
  currentModel: ModelSnapshot;
  viewAt: (seq: number | null) => void;
}

/** Coordinate scoped details, chat, version navigation and comparison previews. */
export function useWorkbench({
  workspaceId,
  question: initialQuestion,
  attempts,
  model,
  currentModel,
  viewAt,
}: WorkbenchOptions) {
  const [selection, select] = useState<EntitySelection | null>(null);
  const [comparison, setComparison] = useState<{
    before: number;
    after: number;
    pinned: boolean;
  } | null>(null);
  const previewTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (previewTimer.current) clearTimeout(previewTimer.current);
    },
    [],
  );
  const entities = useMemo(() => indexModel(model.model?.value), [model]);
  const ticks = attempts;
  const latest = currentModel.context.seq;
  const playhead = model.context.seq;
  const modelRevision = model.model?.source.ref.revision;
  const activeComparison = comparison?.before === playhead ? comparison : null;
  const compared = useModelDiff(
    workspaceId,
    model.context.commit_id,
    attempts.find((record) => record.record.seq === activeComparison?.after)?.commit_id ?? null,
  );
  const retainPreview = () => {
    if (previewTimer.current) clearTimeout(previewTimer.current);
  };
  const endPreview = () => {
    retainPreview();
    previewTimer.current = setTimeout(
      () => setComparison((value) => (value?.pinned ? value : null)),
      350,
    );
  };
  const previewComparison = (after: number, pinned = false) => {
    retainPreview();
    setComparison((value) =>
      !pinned && value?.pinned && value.before === playhead
        ? value
        : { before: playhead, after, pinned },
    );
  };
  const dismissComparison = () => {
    retainPreview();
    setComparison(null);
  };
  const selectVersion = (seq: number | null) => {
    dismissComparison();
    viewAt(seq === latest ? null : seq);
  };
  const question = model.model?.value.question ?? initialQuestion;
  const simulation = model.findings.simulation;
  // Node histories need a simulation of the viewed model revision, certified or not.
  const simulationResult =
    simulation?.source.validity === "fresh" && simulation.value.model.revision === modelRevision
      ? simulation.value
      : null;
  const context: ScopeContext = {
    model,
    entities,
    select,
    ticks,
    dataDiff: null,
  };

  const toggleComparison = () => {
    if (activeComparison) setComparison({ ...activeComparison, pinned: !activeComparison.pinned });
  };

  return {
    selection,
    ticks,
    latest,
    playhead,
    activeComparison,
    compared,
    retainPreview,
    endPreview,
    previewComparison,
    dismissComparison,
    toggleComparison,
    selectVersion,
    question,
    simulationResult,
    select,
    context,
  };
}
