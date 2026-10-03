"use client";

import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  applyProgressEvents,
  EMPTY_ATTEMPT_PROGRESS,
  type AttemptProgressView,
} from "@/lib/attempt-progress";
import { isMockMode } from "../api/mock-provider";

const client = createModelClient();

const PROGRESS_POLL_INTERVAL_MS = 2_000;

/**
 * Tail one running attempt's progress. Each attempt has its own view and cursor, so a new attempt
 * starts empty and a late response for an earlier one never reaches it. Nothing is kept after the
 * reader unmounts: reopening re-reads the attempt's retained events from the beginning.
 */
export function useAttemptProgress(workspaceId: string, attemptId: string) {
  const queryClient = useQueryClient();
  const queryKey = ["study", workspaceId, "attempt-progress", attemptId] as const;
  return useQuery({
    queryKey,
    queryFn: async ({ signal }) => {
      const view =
        queryClient.getQueryData<AttemptProgressView>(queryKey) ?? EMPTY_ATTEMPT_PROGRESS;
      const { data, error, response } = await client.GET("/api/studies/{workspace_id}/events", {
        params: {
          path: { workspace_id: workspaceId },
          query: { attempt_id: attemptId, after: view.cursor },
        },
        signal,
      });
      if (error) {
        throw new Error(`Cannot read progress (${response.status}): ${JSON.stringify(error)}`);
      }
      return applyProgressEvents(view, data);
    },
    enabled: !isMockMode(),
    refetchInterval: PROGRESS_POLL_INTERVAL_MS,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
