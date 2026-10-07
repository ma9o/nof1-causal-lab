"use client";

import type { ActionSuccess, ModelSnapshot, TimelineRevision } from "@nof1-causal-lab/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelComparison } from "@/lib/dag/comparison-overlay";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { indexModel } from "./entities";
import { latestSeq } from "./journal";
import type { ScopeContext } from "./scope";
import type { EntitySelection } from "./selection";

export type SnapshotReader = (commitId: string | undefined) => {
  data: ModelSnapshot | undefined;
  error: Error | null;
  isPlaceholderData?: boolean;
  result?: ActionSuccess | undefined;
};

export function useWorkbenchSnapshots(
  attempts: readonly TimelineRevision[],
  useSnapshot: SnapshotReader,
) {
  const [playheadOverride, viewAt] = useState<number | null>(null);
  const latest = latestSeq(attempts);
  const playhead = playheadOverride ?? latest;
  const record = attempts.find((item) => item.record.seq === playhead);
  // Applied leaves own their results; failed attempts inspect their unchanged parent state.
  const commitId = record
    ? record.record.attempt.outcome.status === "applied" && record.record.attempt.request !== null
      ? record.commit_id
      : record.parent_ids[0]
    : attempts.find((entry) => entry.record.seq === latest)?.commit_id;
  const selected = useSnapshot(commitId);
  const hasSelectedState = attempts.some(
    (entry) =>
      entry.commit_id === commitId &&
      entry.record.attempt.outcome.status === "applied" &&
      entry.record.attempt.request !== null,
  );
  const hasCurrentState = attempts.some(
    (entry) =>
      entry.record.seq === latest &&
      entry.record.attempt.outcome.status === "applied" &&
      entry.record.attempt.request !== null &&
      entry.record.attempt.action !== "data_diff" &&
      entry.record.attempt.action !== "model_diff",
  );
  return { selected, viewAt, focusSeq: playhead, hasSelectedState, hasCurrentState };
}

interface WorkbenchOptions {
  useSnapshot: SnapshotReader;
  workspaceId: string;
  question: string | undefined;
  attempts: readonly TimelineRevision[];
  model: ModelSnapshot;
  focusSeq: number;
  result: ActionSuccess | undefined;
  viewAt: (seq: number | null) => void;
}

/** Coordinate scoped details, chat, version navigation and comparison previews. */
export function useWorkbench({
  useSnapshot,
  workspaceId,
  question: initialQuestion,
  attempts,
  model,
  focusSeq,
  result,
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
  const entities = useMemo(() => indexModel(model.model), [model]);
  const ticks = attempts;
  const latest = latestSeq(attempts);
  const playhead = focusSeq;
  const modelRevision = model.state.current.model?.revision;
  const activeComparison = comparison?.before === playhead ? comparison : null;
  const comparedCall = attempts.find((record) => record.record.seq === activeComparison?.after);
  const comparedCommit =
    comparedCall?.record.attempt.outcome.status === "applied" &&
    comparedCall.record.attempt.request !== null
      ? comparedCall.commit_id
      : comparedCall?.parent_ids[0];
  const selectedCall = attempts.find((record) => record.record.seq === playhead);
  const selectedCommit =
    selectedCall?.record.attempt.outcome.status === "applied" &&
    selectedCall.record.attempt.request !== null
      ? selectedCall.commit_id
      : selectedCall?.parent_ids[0];
  const compared = useModelDiff(
    workspaceId,
    selectedCommit ?? null,
    comparedCommit ?? null,
    attempts,
  );
  const request = selectedCall?.record.attempt.request;
  const recorded =
    request?.action === "model_diff" && result?.action === "model_diff"
      ? { input: request.input, report: result.body }
      : null;
  const beforeView = useSnapshot(
    activeComparison ? selectedCommit : (recorded?.input.before_ref ?? model.commit_id),
  );
  const afterView = useSnapshot(
    activeComparison ? comparedCommit : (recorded?.input.after_ref ?? model.commit_id),
  );
  const report = activeComparison ? compared.data : recorded?.report;
  const modelComparison: ModelComparison | null =
    report && beforeView.data && afterView.data
      ? { ...report, beforeModel: beforeView.data.model, afterModel: afterView.data.model }
      : null;
  const comparisonError = beforeView.error ?? afterView.error;
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
  const question = model.question?.text ?? initialQuestion;
  const simulation = model.simulation;
  // Node histories need a simulation of the viewed model revision, certified or not.
  const simulationResult =
    simulation?.evidence.model.revision === modelRevision ? simulation : null;
  const context: ScopeContext = {
    model,
    entities,
    select,
    ticks,
    dataDiff: null,
    result,
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
    modelComparison,
    comparisonError,
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
