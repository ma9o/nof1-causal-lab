"use client";

import {
  type ActionMessageEvent,
  type ModelSnapshot,
  type StudyRevision,
} from "@nof1-causal-lab/api-types";
import Link from "next/link";
import { useEffect, useMemo, useRef } from "react";
import { LayeredCausalGraph } from "@/components/dag/layered/layered-causal-graph";
import { graphEntities } from "@/lib/dag/layered-model";
import { Button } from "@/components/ui/button";
import type { EpisodeProgressPayload } from "@/lib/api/analysis";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import { useLLMTraceForAction } from "@/lib/hooks/use-llm-trace";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { useActionMessages } from "@/lib/hooks/use-run-events";
import {
  useWorkbench,
  useWorkbenchSnapshots,
  type SnapshotReader,
} from "@/lib/model-asset/use-workbench";
import { ConversationPane, type ActionTraceState, type UseActionTrace } from "./conversation-pane";
import { DetailsPane } from "./details-pane";
import { VersionScrubber } from "./version-scrubber";

export interface CausalModelAssetViewProps {
  workspaceId: string;
  question: string | undefined;
  useSnapshot: SnapshotReader;
  transitions: StudyRevision[];
  branches: EpisodeProgressPayload["branches"];
  progress: PipelineProgress;
  useActionTrace: UseActionTrace;
  actionMessages?: ActionMessageEvent[];
}

export function CausalModelAssetView(props: CausalModelAssetViewProps) {
  const { selected, current, viewAt } = useWorkbenchSnapshots(
    props.transitions,
    props.branches,
    props.useSnapshot,
  );
  if (selected.error || current.error)
    return (
      <div role="alert" className="p-6">
        {(selected.error ?? current.error)?.message}
      </div>
    );
  if (!selected.data || !current.data)
    return (
      <div role="status" className="p-6">
        Loading model version…
      </div>
    );
  return (
    <ModelRevision
      {...props}
      model={selected.data}
      currentModel={current.data}
      loadingRevision={selected.isPlaceholderData === true || current.isPlaceholderData === true}
      viewAt={viewAt}
    />
  );
}

function ModelRevision({
  workspaceId,
  question: initialQuestion,
  transitions,
  branches,
  progress,
  useActionTrace,
  model,
  currentModel,
  loadingRevision,
  viewAt,
  actionMessages = [],
}: CausalModelAssetViewProps & {
  model: ModelSnapshot;
  currentModel: ModelSnapshot;
  loadingRevision: boolean;
  viewAt: (seq: number | null) => void;
}) {
  const {
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
  } = useWorkbench({
    workspaceId,
    question: initialQuestion,
    transitions,
    progress,
    model,
    currentModel,
    viewAt,
  });
  const graph = useMemo(() => graphEntities(model), [model]);
  const comparisonPane = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (activeComparison?.pinned) {
      comparisonPane.current?.focus({ preventScroll: true });
      comparisonPane.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [activeComparison?.pinned, activeComparison?.before, activeComparison?.after]);

  return (
    <div
      className="flex min-h-screen flex-col bg-muted/20 md:h-dvh md:overflow-hidden"
      onKeyDown={(event) => {
        if (event.key === "Escape" && activeComparison) {
          event.stopPropagation();
          dismissComparison();
        }
      }}
    >
      <header className="flex flex-none flex-wrap items-center gap-x-4 gap-y-2 border-b bg-card px-4 py-3 sm:px-5">
        <Link href="/" className="text-sm font-semibold tracking-tight">
          N-of-1 Causal Lab
        </Link>
        <span className="rounded-md border px-2 py-1 font-mono text-[10px] tracking-widest text-muted-foreground">
          {workspaceId}
        </span>
        <div className="w-full text-sm leading-relaxed">{question ?? "Untitled causal model"}</div>
      </header>
      <VersionScrubber
        ticks={ticks}
        playhead={playhead}
        latest={latest}
        branches={branches}
        branch={model.context.branch}
        comparedSeq={activeComparison?.after ?? null}
        onPlayhead={selectVersion}
        onPreviewComparison={(seq) => previewComparison(seq)}
        onEndPreview={endPreview}
        onKeepComparison={(seq) => previewComparison(seq, true)}
      />
      <main
        aria-busy={loadingRevision}
        className="relative grid min-h-0 flex-1 gap-3 p-3 md:grid-cols-[minmax(0,1fr)_320px] lg:px-5 lg:pb-4 xl:grid-cols-[minmax(0,1fr)_400px]"
      >
        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          <section
            aria-label="Causal graph"
            className="relative flex h-[480px] flex-none flex-col overflow-hidden rounded-2xl border bg-card md:h-auto md:min-h-0 md:flex-1"
          >
            <div className="flex h-11 flex-none items-center justify-between gap-2 border-b px-4">
              <h1 className="text-sm font-semibold">Causal model</h1>
              {activeComparison ? (
                <div
                  ref={comparisonPane}
                  id="model-comparison-preview"
                  role="region"
                  aria-label="Model comparison preview"
                  tabIndex={-1}
                  className="flex min-w-0 items-center gap-2 text-[11px]"
                  onPointerEnter={retainPreview}
                  onPointerLeave={endPreview}
                  onFocus={retainPreview}
                >
                  <span className="whitespace-nowrap font-medium text-amber-700">
                    version {playhead} → {activeComparison.after}
                  </span>
                  <span className="hidden items-center gap-2 text-[10px] text-muted-foreground sm:flex">
                    <span className="text-amber-700">~ changed</span>
                    <span className="text-emerald-700">+ added</span>
                    <span className="text-rose-700">− removed</span>
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    variant={activeComparison.pinned ? "secondary" : "ghost"}
                    aria-label="Keep comparison"
                    aria-pressed={activeComparison.pinned}
                    onClick={toggleComparison}
                  >
                    Keep
                  </Button>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    aria-label="Close comparison"
                    onClick={dismissComparison}
                  >
                    ×
                  </Button>
                </div>
              ) : null}
            </div>
            <div
              className="relative min-h-0 flex-1"
              onPointerEnter={retainPreview}
              onPointerLeave={endPreview}
            >
              {activeComparison && (compared.isLoading || compared.error) && (
                <p
                  role={compared.error ? "alert" : "status"}
                  className="absolute top-3 left-3 z-20 rounded border bg-card px-3 py-2 text-xs"
                >
                  {compared.error ? compared.error.message : "Reading differences…"}
                </p>
              )}
              {graph.constructs.length > 0 || activeComparison ? (
                <LayeredCausalGraph
                  model={model}
                  simulation={causalResult}
                  selectedNode={selection?.kind === "construct" ? selection.id : null}
                  onSelectNode={(id) => select(id ? { kind: "construct", id } : null)}
                  comparison={activeComparison ? (compared.data ?? null) : null}
                  variant="asset"
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
                  <p className="text-sm font-medium">No structure defined yet</p>
                  <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">
                    The graph will appear when the agent harness records a model definition.
                  </p>
                </div>
              )}
            </div>
          </section>
          {selection && (
            <DetailsPane
              selection={selection}
              context={context}
              loading={loadingRevision}
              onClose={() => select(null)}
            />
          )}
        </div>
        <aside
          aria-label="Action log"
          className="relative flex h-[480px] min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border bg-card md:h-auto"
        >
          <ConversationPane
            ticks={ticks}
            branches={branches}
            branch={model.context.branch}
            playhead={playhead}
            latest={latest}
            running={running}
            focusSeq={focusSeq}
            useActionTrace={useActionTrace}
            onSelectTick={selectVersion}
            actionMessages={actionMessages}
          />
        </aside>
      </main>
    </div>
  );
}

/** The workbench reads semantic snapshots, committed history and recorded traces. */
export function CausalModelAsset({
  workspaceId,
  question,
  progress,
  episode,
}: {
  workspaceId: string;
  question: string | undefined;
  progress: PipelineProgress;
  episode: EpisodeProgressPayload;
}) {
  const actionMessages = useActionMessages(workspaceId);
  const useSnapshot = useMemo(
    () =>
      function useWorkspaceSnapshot(commitId: string, branch: string) {
        return useModelSnapshot(workspaceId, commitId, branch);
      },
    [workspaceId],
  );
  const useActionTrace = useMemo<UseActionTrace>(
    () =>
      function useWorkspaceActionTrace(seq, enabled): ActionTraceState {
        const query = useLLMTraceForAction(
          workspaceId,
          episode.transitions.find((record) => record.seq === seq)?.commit_id ?? null,
          enabled,
        );
        if (!enabled || query.isError) return { status: "absent" };
        if (query.data) return { status: "ready", trace: query.data };
        return { status: "loading" };
      },
    [workspaceId, episode.transitions],
  );
  return (
    <CausalModelAssetView
      workspaceId={workspaceId}
      question={question}
      useSnapshot={useSnapshot}
      transitions={episode.transitions}
      branches={episode.branches}
      progress={progress}
      useActionTrace={useActionTrace}
      actionMessages={actionMessages}
    />
  );
}
