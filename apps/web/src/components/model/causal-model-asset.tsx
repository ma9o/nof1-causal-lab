"use client";

import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";

import { DagCanvasFrame } from "@/components/dag/core/dag-canvas";
import { LayeredCausalGraph } from "@/components/dag/layered/layered-causal-graph";
import { Button } from "@/components/ui/button";
import { type EpisodeProgressPayload, recomputeStaleArtifacts } from "@/lib/api/analysis";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import { useLLMTraceForMove } from "@/lib/hooks/use-llm-trace";
import { journalTicks, latestSeq, modelPosition } from "@/lib/model-asset/journal";
import { cn } from "@/lib/utils";
import type {
  ArtifactFreshness,
  ArtifactId,
  LLMTrace,
  ModelSnapshot,
  Move,
  TransitionRecord,
} from "@nof1-causal-lab/api-types";
import { useMutation } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import { indexModel } from "./asset-data";
import { ConversationPane, type MoveTraceState, type UseMoveTrace } from "./conversation-pane";
import { DetailsPane } from "./details-pane";
import { ARTIFACT_LABEL, ASSET_SELECTION, type ModelSelection } from "./model-selection";
import { buildModelQueries } from "./queries";
import type { ScopeContext } from "./scopes/scope-context";
import { VersionScrubber } from "./version-scrubber";

export interface CausalModelAssetViewProps {
  workspaceId: string;
  question: string | undefined;
  readOnly: boolean;
  useSnapshot: (atSeq: number) => { data: ModelSnapshot | undefined; error: Error | null };
  transitions: TransitionRecord[];
  artifacts: ArtifactFreshness[];
  legal: Move[];
  progress: PipelineProgress;
  analysisTrace: LLMTrace | undefined;
  useMoveTrace: UseMoveTrace;
  /** Starts the machine's auto-run: the next legal moves in dependency order. */
  onRun: (() => void) | null;
}

/** The persistent model view: header, full-width scrubber, graph over details, conversation. */
export function CausalModelAssetView(props: CausalModelAssetViewProps) {
  const [selection, setSelection] = useState<ModelSelection>(ASSET_SELECTION);
  const [focusSeq, setFocusSeq] = useState<number | null>(null);

  const [playheadOverride, setPlayheadOverride] = useState<number | null>(null);
  const latest = latestSeq(props.transitions);
  const requested = playheadOverride ?? latest;
  const playhead = latestSeq(props.transitions.filter((record) => record.seq <= requested));
  const selected = props.useSnapshot(playhead);
  const current = props.useSnapshot(latest);
  if (selected.error || current.error)
    return (
      <div role="alert" className="p-6">
        {(selected.error ?? current.error)?.message}
      </div>
    );
  if (!selected.data || !current.data)
    return (
      <div role="status" className="p-6">
        Loading model revision…
      </div>
    );
  return (
    <ModelRevision
      {...props}
      model={selected.data}
      currentModel={current.data}
      viewAt={setPlayheadOverride}
      selection={selection}
      setSelection={setSelection}
      focusSeq={focusSeq}
      setFocusSeq={setFocusSeq}
    />
  );
}

function ModelRevision({
  workspaceId,
  readOnly,
  transitions,
  artifacts: liveArtifacts,
  legal,
  progress,
  analysisTrace,
  useMoveTrace,
  onRun,
  model,
  currentModel,
  viewAt,
  selection,
  setSelection,
  focusSeq,
  setFocusSeq,
}: CausalModelAssetViewProps & {
  model: ModelSnapshot;
  currentModel: ModelSnapshot;
  viewAt: (seq: number | null) => void;
  selection: ModelSelection;
  setSelection: (selection: ModelSelection) => void;
  focusSeq: number | null;
  setFocusSeq: (seq: number | null) => void;
}) {
  const entities = useMemo(() => indexModel(model), [model]);
  const question = model.question?.value.text;
  const artifacts = model.artifacts;

  const ticks = useMemo(() => journalTicks(transitions), [transitions]);
  const latest = useMemo(() => latestSeq(transitions), [transitions]);
  const playhead = model.seq;
  const snapshot = useMemo(() => modelPosition(model), [model]);
  const current = useMemo(() => modelPosition(currentModel), [currentModel]);
  const outcome =
    entities.constructs.find((construct) => construct.id === model.latent_structure?.value.default_outcome?.id)
      ?.name ?? null;
  const queries = useMemo(() => buildModelQueries(model), [model]);
  const selectedQuery =
    selection.kind === "query" ? queries.find((query) => query.key === selection.key) : undefined;
  const simulation =
    selectedQuery?.simulation?.evaluation.model.id === model.model.id &&
    selectedQuery.simulation.evaluation.posterior.version === model.state.current.posterior?.version
      ? selectedQuery.simulation
      : null;
  const staleArtifacts = useMemo(
    () =>
      new Set<ArtifactId>(
        artifacts.filter((artifact) => artifact.stale).map((artifact) => artifact.artifact_id),
      ),
    [artifacts],
  );
  const running = progress.runningTransitions;
  const isNow = playhead >= latest && running.length === 0;
  const nextRun = useMemo(() => {
    const statusById = new Map(liveArtifacts.map((artifact) => [artifact.artifact_id, artifact]));
    const candidates = legal
      .filter((move) => move.kind === "run")
      .map((move) => move.artifact_id)
      .filter((artifactId) => {
        const status = statusById.get(artifactId);
        return !status || !status.exists || status.stale;
      });
    const ordered = [...progress.transitionOrder].filter((artifactId) =>
      candidates.includes(artifactId),
    );
    return ordered[0] ?? candidates[0] ?? null;
  }, [liveArtifacts, legal, progress.transitionOrder]);

  const select = useCallback((next: ModelSelection) => setSelection(next), [setSelection]);
  const selectTick = useCallback(
    (seq: number) => {
      setSelection({ kind: "version", seq });
      setFocusSeq(seq);
    },
    [setSelection, setFocusSeq],
  );

  const context: ScopeContext = useMemo(
    () => ({
      model,
      entities,
      snapshot,
      current,
      ticks,
      artifacts,
      question,
      queries,
      outcome,
      analysisTrace,
      select,
      viewAt,
      focusConversation: setFocusSeq,
    }),
    [
      model,
      entities,
      snapshot,
      current,
      ticks,
      artifacts,
      question,
      queries,
      outcome,
      analysisTrace,
      select,
      viewAt,
      setFocusSeq,
    ],
  );

  const selectedNode = selection.kind === "construct" ? selection.id : null;
  const status =
    running.length > 0
      ? `auto-run · ${running.map((id) => ARTIFACT_LABEL[id]).join(", ")} running`
      : !isNow
        ? `viewing v${playhead} of v${latest} · read-only`
        : "live · idle";

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="flex h-12 flex-none items-center gap-3 border-b bg-background/80 px-6">
        <div className="whitespace-nowrap text-base font-semibold tracking-tight">
          N-of-1 Causal Lab
        </div>
        <span className="rounded border bg-secondary/50 px-2 py-0.5 font-mono text-xs tracking-widest text-muted-foreground">
          {workspaceId}
        </span>
        <div className="min-w-0 flex-1 truncate text-[13px] text-muted-foreground">{question}</div>
        <div className="flex flex-none items-center gap-2">
          <span
            className={cn(
              "inline-flex h-5 items-center rounded-full border px-2 text-xs font-medium",
              running.length > 0 && "border-[#2f6bf0] text-[#2f6bf0]",
              !isNow &&
                running.length === 0 &&
                "border-transparent bg-warning/15 text-warning-foreground",
            )}
          >
            {status}
          </span>
          {!isNow && running.length === 0 ? (
            <Button type="button" variant="outline" size="sm" onClick={() => viewAt(null)}>
              Return to now
            </Button>
          ) : null}
          {isNow && nextRun && onRun && !readOnly ? (
            <Button type="button" size="sm" onClick={onRun}>
              ▶ run {ARTIFACT_LABEL[nextRun]}
            </Button>
          ) : null}
        </div>
      </header>
      <VersionScrubber
        ticks={ticks}
        playhead={playhead}
        latest={latest}
        timings={progress.timings}
        running={running}
        staleArtifacts={staleArtifacts}
        nextRun={nextRun}
        selectedSeq={selection.kind === "version" ? selection.seq : null}
        onSelectTick={selectTick}
        onPlayhead={(seq) => viewAt(seq >= latest ? null : seq)}
      />
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_400px] gap-3 px-6 pt-3 pb-4">
        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          <div className="min-h-0 flex-1">
            {entities.constructs.length > 0 ? (
              <LayeredCausalGraph
                model={model}
                simulation={simulation}
                selectedNode={selectedNode}
                onSelectNode={(id) => select(id ? { kind: "construct", id } : ASSET_SELECTION)}
                variant="asset"
              />
            ) : (
              <DagCanvasFrame fill>{null}</DagCanvasFrame>
            )}
          </div>
          <DetailsPane
            selection={selection}
            context={context}
            posteriorStale={staleArtifacts.has("posterior")}
          />
        </div>
        <ConversationPane
          ticks={ticks}
          playhead={playhead}
          latest={latest}
          running={running}
          question={question}
          focusSeq={focusSeq}
          useMoveTrace={useMoveTrace}
          onSelectTick={selectTick}
        />
      </div>
    </div>
  );
}

/** The asset view fed by the selected semantic snapshot, journal and traces. */
export function CausalModelAsset({
  workspaceId,
  question,
  readOnly,
  progress,
  episode,
}: {
  workspaceId: string;
  question: string | undefined;
  readOnly: boolean;
  progress: PipelineProgress;
  episode: EpisodeProgressPayload;
}) {
  const useSnapshot = useMemo(
    () =>
      function useWorkspaceSnapshot(atSeq: number) {
        return useModelSnapshot(workspaceId, atSeq);
      },
    [workspaceId],
  );
  const useMoveTrace = useMemo<UseMoveTrace>(
    () =>
      function useWorkspaceMoveTrace(seq, enabled): MoveTraceState {
        const query = useLLMTraceForMove(workspaceId, seq, enabled);
        if (!enabled || query.isError) return { status: "absent" };
        if (query.data) return { status: "ready", trace: query.data };
        return { status: "loading" };
      },
    [workspaceId],
  );
  const run = useMutation({ mutationFn: () => recomputeStaleArtifacts(workspaceId) });

  return (
    <CausalModelAssetView
      workspaceId={workspaceId}
      question={question}
      readOnly={readOnly}
      useSnapshot={useSnapshot}
      transitions={episode.transitions}
      artifacts={episode.artifacts}
      legal={episode.legal}
      progress={progress}
      analysisTrace={undefined}
      useMoveTrace={useMoveTrace}
      onRun={readOnly ? null : () => run.mutate()}
    />
  );
}
