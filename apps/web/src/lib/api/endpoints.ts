import type { LLMTrace, UploadResponse } from "@nof1-causal-lab/api-types";
import { apiClient } from "./client";

export async function uploadFile(file: File, workspaceId: string): Promise<UploadResponse> {
  const { data, response } = await apiClient.POST("/api/upload", {
    body: { file, workspaceId },
    bodySerializer: (body) => {
      const form = new FormData();
      form.set("file", body.file, file.name);
      form.set("workspaceId", body.workspaceId);
      return form;
    },
  });
  if (data === undefined) throw new Error(`Upload failed: ${response.status}`);
  return data;
}

/** The merged traces of one recorded action; the caller holds its trace IDs from the journal. */
export async function getLLMTraceForAction(
  workspaceId: string,
  commitId: string,
  traceIds: readonly string[],
): Promise<LLMTrace> {
  const traces = await Promise.all(
    traceIds.map(async (subroutineId) => {
      const { data, response } = await apiClient.GET(
        "/api/studies/{workspace_id}/traces/{commit_id}/{subroutine_id}",
        {
          params: {
            path: { workspace_id: workspaceId, commit_id: commitId, subroutine_id: subroutineId },
          },
        },
      );
      if (data === undefined) throw new Error(`Trace API error ${response.status}`);
      return data;
    }),
  );
  return traces.reduce<LLMTrace>(
    (merged, trace) => ({
      messages: [...merged.messages, ...trace.messages],
      model: trace.model || merged.model,
      total_time_seconds: merged.total_time_seconds + trace.total_time_seconds,
      usage: {
        input_tokens: merged.usage.input_tokens + trace.usage.input_tokens,
        output_tokens: merged.usage.output_tokens + trace.usage.output_tokens,
        reasoning_tokens:
          merged.usage.reasoning_tokens === null && trace.usage.reasoning_tokens === null
            ? null
            : (merged.usage.reasoning_tokens ?? 0) + (trace.usage.reasoning_tokens ?? 0),
      },
    }),
    {
      messages: [],
      model: "",
      total_time_seconds: 0,
      usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: null },
    },
  );
}
