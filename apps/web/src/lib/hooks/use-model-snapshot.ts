"use client";

import { skipToken, useQuery } from "@tanstack/react-query";
import { readActionResult } from "@/lib/api/endpoints";
import { useStudyJournal } from "./use-study-journal";

/** A selected call or explicitly named input revision resolves to its applied producer. */
export function useActionResult(workspaceId: string, identity: string | undefined, enabled = true) {
  const journal = useStudyJournal(workspaceId).data;
  const call = journal?.attempts.find((entry) => entry.record.attempt.outcome.status === "applied" && entry.record.attempt.request !== null && (
    entry.commit_id === identity || entry.record.attempt.outcome.effects.produced.some((artifact) => artifact.revision === identity)
  ));
  return useQuery({
    queryKey: ["action-result", workspaceId, call?.commit_id ?? identity],
    queryFn: call ? ({ signal }) => readActionResult(workspaceId, call, signal) : skipToken,
    enabled: enabled && call !== undefined,
    staleTime: Infinity, retry: false,
  });
}

export function useModelSnapshot(workspaceId: string, identity?: string, enabled = true) {
  const query = useActionResult(workspaceId, identity, enabled);
  return { ...query, data: query.data?.snapshot ?? undefined, result: query.data };
}
