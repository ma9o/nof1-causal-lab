"use client";

import { useQuery } from "@tanstack/react-query";
import type { EpisodeProgressPayload } from "@/lib/api/analysis";
import { getEpisodeProgressQueryKey } from "./use-run-events";

/**
 * Read-only subscriber to the latest episode progress payload — the journal, the
 * per-artifact freshness report and the legal moves — which `useRunEvents` keeps polling.
 */
export function useEpisodeProgress(workspaceId: string | null): EpisodeProgressPayload | undefined {
  return useQuery<EpisodeProgressPayload>({
    queryKey: getEpisodeProgressQueryKey(workspaceId ?? "__none__"),
    queryFn: () => undefined as never,
    enabled: false,
  }).data;
}
