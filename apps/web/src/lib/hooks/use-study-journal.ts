"use client";

import {
  createModelClient,
  type RecordDependency,
  type RunningAction,
  type StudyRevision,
} from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { isMockMode } from "../api/mock-provider";

const client = createModelClient();

const ACTIVE_POLL_INTERVAL_MS = 2_000;
const IDLE_POLL_INTERVAL_MS = 10_000;

export function journalPollIntervalMs(active: boolean): number {
  return active ? ACTIVE_POLL_INTERVAL_MS : IDLE_POLL_INTERVAL_MS;
}

/** The attempt journal with its branch heads, and the attempt the study is still executing. */
export interface StudyJournal {
  seq: number;
  running: RunningAction | null;
  attempts: StudyRevision[];
  branches: Record<string, string>;
  /** Which earlier records each record's request named, served with the journal. */
  dependencies: RecordDependency[];
}

export function getStudyJournalQueryKey(workspaceId: string) {
  return ["study", workspaceId, "journal"] as const;
}

/** Every attempt, failed or not, advances `seq`, so a held journal at that seq is current. */
export async function readStudyJournal(
  workspaceId: string,
  held?: StudyJournal,
  signal?: AbortSignal,
): Promise<StudyJournal> {
  const status = await client.GET("/api/studies/{workspace_id}", {
    params: { path: { workspace_id: workspaceId } },
    signal,
  });
  if (status.error) {
    throw new Error(
      `Cannot read the study (${status.response.status}): ${JSON.stringify(status.error)}`,
    );
  }
  if (held?.seq === status.data.seq) return { ...held, running: status.data.running };
  const timeline = await client.GET("/api/studies/{workspace_id}/timeline", {
    params: { path: { workspace_id: workspaceId } },
    signal,
  });
  if (timeline.error) {
    throw new Error(
      `Cannot read the journal (${timeline.response.status}): ${JSON.stringify(timeline.error)}`,
    );
  }
  return {
    seq: status.data.seq,
    running: status.data.running,
    attempts: timeline.data.attempts,
    branches: timeline.data.branches,
    dependencies: timeline.data.dependencies,
  };
}

export function useStudyJournal(workspaceId: string) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: getStudyJournalQueryKey(workspaceId),
    queryFn: ({ signal }) =>
      readStudyJournal(
        workspaceId,
        queryClient.getQueryData<StudyJournal>(getStudyJournalQueryKey(workspaceId)),
        signal,
      ),
    // Read-only viewers poll too: published workspaces carry a real journal,
    // and a live local run publishing to the hosted store tails through here.
    enabled: !isMockMode(),
    // Poll fast until the journal arrives and while the study executes an action.
    refetchInterval: (query) =>
      journalPollIntervalMs(!query.state.data || query.state.data.running !== null),
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
