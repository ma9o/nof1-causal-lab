"use client";

import type { ModelSnapshot, StudyRevision } from "@nof1-causal-lab/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { hasCausalEffects } from "@/lib/simulation-report";
import { indexModel } from "./entities";
import { journalTicks, latestSeq } from "./journal";
import type { ScopeContext } from "./scope";
import type { ModelSelection } from "./selection";
import { isModelOperation } from "./workbench";

export type SnapshotReader = (
  commitId: string,
  branch: string,
) => {
  data: ModelSnapshot | undefined;
  error: Error | null;
  isPlaceholderData?: boolean;
};

export function useWorkbenchSnapshots(
  transitions: StudyRevision[],
  branches: Record<string, string>,
  useSnapshot: SnapshotReader,
) {
  const [playheadOverride, viewAt] = useState<number | null>(null);
  const latest = latestSeq(transitions);
  const branch = transitions.find((record) => record.seq === latest)?.branch ?? "main";
  const playhead = playheadOverride ?? latest;
  const record = transitions.find((item) => item.seq === playhead);
  // Unsuccessful attempts record no version: inspect their unchanged parent state.
  const commitId = record
    ? record.status === "applied"
      ? record.commit_id
      : record.parent_ids[0]
    : branches[branch];
  const selected = useSnapshot(commitId, record?.branch ?? branch);
  const current = useSnapshot(branches[branch], branch);
  return { selected, current, viewAt };
}

interface WorkbenchOptions {
  workspaceId: string;
  question: string | undefined;
  transitions: StudyRevision[];
  progress: PipelineProgress;
  model: ModelSnapshot;
  currentModel: ModelSnapshot;
  viewAt: (seq: number | null) => void;
}

/** Coordinate scoped details, chat, version navigation and comparison previews. */
export function useWorkbench({
  workspaceId,
  question: initialQuestion,
  transitions,
  progress,
  model,
  currentModel,
  viewAt,
}: WorkbenchOptions) {
  const [selectionOverride, setSelection] = useState<ModelSelection | null>(null);
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
  const ticks = useMemo(() => journalTicks(transitions), [transitions]);
  const latest = currentModel.context.seq;
  const playhead = model.context.seq;
  // Version details follow the viewed snapshot, including new harness work at the live head.
  const selection: ModelSelection | null =
    selectionOverride?.kind === "revision" &&
    ticks.some((tick) => tick.seq === selectionOverride.seq && tick.status === "applied")
      ? { kind: "revision", seq: playhead }
      : selectionOverride;
  const modelRevision = model.context.state.current.model?.revision;
  const focusSeq = selection?.kind === "revision" ? selection.seq : playhead;
  const activeComparison = comparison?.before === playhead ? comparison : null;
  const compared = useModelDiff(
    workspaceId,
    model.context.commit_id,
    transitions.find((record) => record.seq === activeComparison?.after)?.commit_id ?? null,
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
    setSelection({ kind: "revision", seq: seq ?? latest });
    viewAt(seq === latest ? null : seq);
  };
  const question = model.model?.value.question ?? initialQuestion;
  const running = progress.runningTransitions.filter(isModelOperation);
  const simulation = model.findings.simulation;
  const causalResult =
    simulation?.source.validity === "fresh" &&
    hasCausalEffects(simulation.value) &&
    simulation.value.model.revision === modelRevision
      ? simulation.value
      : null;
  const select = (next: ModelSelection | null) => {
    if (next?.kind === "revision") selectVersion(next.seq);
    else setSelection(next);
  };
  const context: ScopeContext = {
    model,
    entities,
    ticks,
    select,
  };

  const toggleComparison = () => {
    if (activeComparison) setComparison({ ...activeComparison, pinned: !activeComparison.pinned });
  };

  return {
    selection,
    focusSeq,
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
    running,
    causalResult,
    select,
    context,
  };
}
