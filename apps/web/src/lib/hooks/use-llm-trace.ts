"use client";

import { useQuery } from "@tanstack/react-query";
import { getLLMTrace, getLLMTraceForAction } from "../api/endpoints";

const LLM_TRACE_QUERY_VERSION = 2;

export function getLLMTraceQueryKey(workspaceId: string | null, artifactId: string | null) {
  return ["pipeline", workspaceId, "llm-trace", artifactId, `v${LLM_TRACE_QUERY_VERSION}`] as const;
}

/** Merged LLM trace of the applied transition that produced an artifact's current revision. */
export function useLLMTrace(
  workspaceId: string | null,
  artifactId: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: getLLMTraceQueryKey(workspaceId, artifactId),
    queryFn: () => getLLMTrace(workspaceId as string, artifactId as string),
    enabled: !!workspaceId && !!artifactId && enabled,
    staleTime: Number.POSITIVE_INFINITY,
    // A 404 means the producing transition promoted no traces — not transient.
    retry: false,
  });
}

export function getLLMTraceForActionQueryKey(workspaceId: string | null, commitId: string | null) {
  return [
    "pipeline",
    workspaceId,
    "llm-trace",
    "action",
    commitId,
    `v${LLM_TRACE_QUERY_VERSION}`,
  ] as const;
}

/** Merged LLM trace of one recorded action: the conversation turn behind a scrubber tick. */
export function useLLMTraceForAction(
  workspaceId: string | null,
  commitId: string | null,
  traceIds: readonly string[],
  enabled: boolean,
) {
  return useQuery({
    queryKey: getLLMTraceForActionQueryKey(workspaceId, commitId),
    queryFn: () => getLLMTraceForAction(workspaceId as string, commitId as string, traceIds),
    // An action without traces (e.g. submitted directly through the API) has nothing to fetch.
    enabled: !!workspaceId && commitId != null && traceIds.length > 0 && enabled,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}
