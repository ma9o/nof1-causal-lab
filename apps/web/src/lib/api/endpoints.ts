import type { LLMTrace, UploadResponse } from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";

export async function uploadFile(file: File, workspaceId: string): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("workspaceId", workspaceId);
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

/** The merged traces of one recorded action; the caller holds its trace IDs from the journal. */
export async function getLLMTraceForAction(
  workspaceId: string,
  commitId: string,
  traceIds: readonly string[],
): Promise<LLMTrace> {
  const search = new URLSearchParams([
    ["commitId", commitId],
    ...traceIds.map((traceId) => ["trace", traceId]),
  ]).toString();
  return apiFetch<LLMTrace>(`/api/traces/${workspaceId}?${search}`);
}
