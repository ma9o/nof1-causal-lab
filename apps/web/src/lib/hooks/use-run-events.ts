"use client";

import {
  getEpisodeProgress,
  type EpisodeProgressPayload,
  type RuntimeEvent,
  type StudyRevision,
} from "@/lib/api/analysis";
import { groupStaleArtifactsByProducer } from "@/lib/artifact-staleness";
import {
  applyExtractionEvent,
  getExtractionStateQueryKey,
  parseExtractionEvent,
  type ExtractionReplayState,
} from "@/lib/extraction-runtime";
import {
  applyModelSpecAdmissionEvent,
  getModelSpecAdmissionStateQueryKey,
  parseModelSpecAdmissionEvent,
  type ModelSpecAdmissionReplayState,
} from "@/lib/model-spec-admission-runtime";
import { type TransitionProgressStatus } from "@/lib/transition-runtime";
import type { PipelineSectionId } from "@nof1-causal-lab/api-types";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { isMockMode, simulatePipelineEvents } from "../api/mock-provider";
import {
  applyTransitionUpdate,
  initialProgress,
  restartTransitionAttempt,
  type PipelineProgress,
  type TransitionRunStatus,
} from "./pipeline-progress";

export type { PipelineProgress, TransitionRunStatus, TransitionTiming } from "./pipeline-progress";

const PROGRESS_POLL_INTERVAL_MS = 2_000;
const IDLE_PROGRESS_POLL_INTERVAL_MS = 10_000;

export function progressPollIntervalMs(active: boolean): number {
  return active ? PROGRESS_POLL_INTERVAL_MS : IDLE_PROGRESS_POLL_INTERVAL_MS;
}

function getPipelineStatusQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "status"] as const;
}

export function getEpisodeProgressQueryKey(workspaceId: string) {
  return ["episode", workspaceId, "progress"] as const;
}

/** Event cursors are `{time_ns:020d}-{uuid}.json` filenames — time-ordered by construction. */
export function cursorTimestampMs(cursor: string): number | undefined {
  const nanos = cursor.split("-", 1)[0];
  if (!/^\d+$/.test(nanos)) {
    return undefined;
  }
  return Math.floor(Number(nanos) / 1_000_000);
}

function isPipelineSectionId(value: unknown): value is PipelineSectionId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

export interface TransitionProgressEvent {
  artifactId: PipelineSectionId;
  status: TransitionProgressStatus;
  eventTime?: number;
  error?: { type: string; message: string };
}

export function parseTransitionProgressEvent(record: RuntimeEvent): TransitionProgressEvent | null {
  if (
    record.event !== "nof1-causal-lab.transition.running" &&
    record.event !== "nof1-causal-lab.transition.completed" &&
    record.event !== "nof1-causal-lab.transition.failed"
  ) {
    return null;
  }

  const artifactId = record.transition_id;
  if (!isPipelineSectionId(artifactId)) return null;
  const statuses = {
    "nof1-causal-lab.transition.running": "running",
    "nof1-causal-lab.transition.completed": "completed",
    "nof1-causal-lab.transition.failed": "failed",
  } as const;
  return {
    artifactId,
    status: statuses[record.event],
    eventTime: cursorTimestampMs(record.cursor),
    error: record.error ?? undefined,
  };
}

/** Attach display time to the canonical event. */
function toRuntimeEventRecord(record: RuntimeEvent) {
  const timestampMs = cursorTimestampMs(record.cursor);
  return {
    ...record,
    occurred: timestampMs === undefined ? null : new Date(timestampMs).toISOString(),
  };
}

function invalidateArtifactView(
  queryClient: ReturnType<typeof useQueryClient>,
  workspaceId: string,
) {
  queryClient.invalidateQueries({ queryKey: ["pipeline", workspaceId, "artifact"] });
  queryClient.invalidateQueries({ queryKey: ["model-snapshot", workspaceId, "latest"] });
}

/**
 * The durable journal is authoritative for a transition's terminal state: an applied
 * run transition means it completed, a raised one means it failed. Telemetry
 * `completed` events also drive completion, but they are ephemeral — the
 * transition keeps the display correct even when the event log has been pruned.
 */
function applyRunTransition(
  progress: PipelineProgress | undefined,
  transition: StudyRevision,
  transitionOrder: readonly PipelineSectionId[],
): PipelineProgress | undefined {
  if (transition.operation_id === null) {
    return progress;
  }
  const artifactId = transition.operation_id;
  if (!isPipelineSectionId(artifactId)) {
    return progress;
  }
  const eventTime = Date.parse(transition.ts);
  const ts = Number.isFinite(eventTime) ? eventTime : undefined;

  if (transition.status === "applied") {
    return applyTransitionUpdate(progress, artifactId, "completed", ts, undefined, transitionOrder);
  }
  if (transition.status === "raised") {
    return applyTransitionUpdate(
      progress,
      artifactId,
      "failed",
      ts,
      transition.error_message ?? transition.error_type ?? undefined,
      transitionOrder,
    );
  }
  return progress; // rejected attempts never executed — leave status untouched
}

function hasRunningTransition(progress: PipelineProgress | undefined): boolean {
  return (
    !!progress &&
    progress.transitionOrder.some((transitionId) => progress.artifacts[transitionId] === "running")
  );
}

function applyExistingArtifactView(
  progress: PipelineProgress | undefined,
  artifactId: PipelineSectionId,
  transitionOrder: readonly PipelineSectionId[],
): PipelineProgress {
  const current = progress ?? initialProgress(transitionOrder);
  if (current.artifacts[artifactId] !== "pending") {
    return current;
  }
  return applyTransitionUpdate(
    current,
    artifactId,
    "completed",
    undefined,
    undefined,
    transitionOrder,
  );
}

export function useRunEvents(
  workspaceId: string | null,
  transitionOrder: readonly PipelineSectionId[] | undefined,
) {
  const queryClient = useQueryClient();
  const cursorRef = useRef<string | null>(null);
  const lastSeqRef = useRef(0);
  const hydratedWorkspaceRef = useRef<string | null>(null);

  const updateTransition = useCallback(
    (
      artifactId: PipelineSectionId,
      status: TransitionRunStatus,
      eventTime?: number,
      errorMessage?: string,
    ) => {
      if (!transitionOrder) {
        return;
      }
      queryClient.setQueryData<PipelineProgress>(["pipeline", workspaceId, "status"], (old) =>
        applyTransitionUpdate(old, artifactId, status, eventTime, errorMessage, transitionOrder),
      );
    },
    [queryClient, transitionOrder, workspaceId],
  );

  const applyProgressPayload = useCallback(
    (payload: EpisodeProgressPayload) => {
      if (!workspaceId || !transitionOrder) {
        return;
      }

      for (const record of payload.events) {
        const runtimeRecord = toRuntimeEventRecord(record);

        const admissionEvent = parseModelSpecAdmissionEvent(runtimeRecord);
        if (admissionEvent) {
          queryClient.setQueryData<ModelSpecAdmissionReplayState>(
            getModelSpecAdmissionStateQueryKey(workspaceId),
            (old) => applyModelSpecAdmissionEvent(old, admissionEvent),
          );
          continue;
        }

        const extractionEvent = parseExtractionEvent(runtimeRecord);
        if (extractionEvent) {
          queryClient.setQueryData<ExtractionReplayState>(
            getExtractionStateQueryKey(workspaceId),
            (old) => applyExtractionEvent(old, extractionEvent),
          );
          continue;
        }

        const transitionEvent = parseTransitionProgressEvent(record);
        if (!transitionEvent) {
          continue;
        }

        if (transitionEvent.status === "running") {
          // The event stream is totally ordered, so a running event after a
          // terminal state is a genuine re-run (stale inputs recomputed).
          queryClient.setQueryData<PipelineProgress>(
            getPipelineStatusQueryKey(workspaceId),
            (old) =>
              restartTransitionAttempt(
                old,
                transitionEvent.artifactId,
                transitionEvent.eventTime,
                transitionOrder,
              ),
          );
          continue;
        }

        updateTransition(
          transitionEvent.artifactId,
          transitionEvent.status,
          transitionEvent.eventTime,
          transitionEvent.error?.message,
        );
        if (transitionEvent.status === "completed") {
          invalidateArtifactView(queryClient, workspaceId);
        }
      }
      if (payload.events.length > 0) {
        cursorRef.current = payload.events[payload.events.length - 1].cursor;
      }

      for (const artifact of payload.artifacts) {
        const artifactId = artifact.artifact_id;
        if (!artifact.exists || !isPipelineSectionId(artifactId)) {
          continue;
        }
        queryClient.setQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId), (old) =>
          applyExistingArtifactView(old, artifactId, transitionOrder),
        );
      }

      for (const transition of payload.transitions) {
        if (transition.seq <= lastSeqRef.current) {
          continue;
        }
        queryClient.setQueryData<PipelineProgress>(
          getPipelineStatusQueryKey(workspaceId),
          (old) => applyRunTransition(old, transition, transitionOrder) ?? old,
        );
        lastSeqRef.current = Math.max(lastSeqRef.current, transition.seq);
      }

      queryClient.setQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId), (old) => ({
        ...(old ?? initialProgress(transitionOrder)),
        staleArtifactsByProducer: groupStaleArtifactsByProducer(payload.artifacts),
      }));
    },
    [queryClient, transitionOrder, updateTransition, workspaceId],
  );

  // Reset the reduced caches when the workspace changes.
  useEffect(() => {
    if (!workspaceId || !transitionOrder || hydratedWorkspaceRef.current === workspaceId) {
      return;
    }
    hydratedWorkspaceRef.current = workspaceId;
    cursorRef.current = null;
    lastSeqRef.current = 0;

    queryClient.setQueryData(
      getPipelineStatusQueryKey(workspaceId),
      initialProgress(transitionOrder),
    );
    queryClient.removeQueries({ queryKey: getExtractionStateQueryKey(workspaceId) });
    queryClient.removeQueries({ queryKey: getModelSpecAdmissionStateQueryKey(workspaceId) });
  }, [queryClient, transitionOrder, workspaceId]);

  useEffect(() => {
    if (!workspaceId || !transitionOrder) return;

    if (isMockMode()) {
      const cleanup = simulatePipelineEvents(
        {
          onTransitionStart: (id) => updateTransition(id, "running"),
          onTransitionComplete: (id) => {
            updateTransition(id, "completed");
            invalidateArtifactView(queryClient, workspaceId);
          },
        },
        transitionOrder,
      );
      return () => {
        cleanup();
      };
    }
  }, [queryClient, transitionOrder, updateTransition, workspaceId]);

  return useQuery({
    queryKey: getEpisodeProgressQueryKey(workspaceId ?? "__none__"),
    queryFn: async () => {
      const held = queryClient.getQueryData<EpisodeProgressPayload>(
        getEpisodeProgressQueryKey(workspaceId as string),
      );
      const response = await getEpisodeProgress(
        workspaceId as string,
        cursorRef.current,
        held?.seq,
      );
      // The journal is only resent when its seq moved; otherwise keep the one already held.
      let payload: EpisodeProgressPayload;
      if (response.transitions !== null && response.branches !== null) {
        payload = { ...response, transitions: response.transitions, branches: response.branches };
      } else if (held) {
        payload = { ...response, transitions: held.transitions, branches: held.branches };
      } else {
        throw new Error("Progress omitted a journal this client does not hold");
      }
      applyProgressPayload(payload);
      return payload;
    },
    // Read-only viewers poll too: published workspaces carry a real journal,
    // and a live local run publishing to the hosted store tails through here.
    enabled: !isMockMode() && !!workspaceId && !!transitionOrder,
    refetchInterval: (query) => {
      const payload = query.state.data;
      if (!payload) {
        return PROGRESS_POLL_INTERVAL_MS;
      }
      const progress = workspaceId
        ? queryClient.getQueryData<PipelineProgress>(getPipelineStatusQueryKey(workspaceId))
        : undefined;
      // Poll fast only while an attempt is in flight: a running transition, or an action the
      // episode workflow is executing. Stale artifacts are a resting state.
      return progressPollIntervalMs(hasRunningTransition(progress) || payload.running !== null);
    },
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
