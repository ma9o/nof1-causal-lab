import type { ActionPoll, DataDiffRequest, LLMTrace, ScientificActionRequest, TimelineRevision } from "@nof1-causal-lab/api-types";
import { apiClient } from "./client";

export async function uploadFile(file: File, workspaceId: string): Promise<string> {
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

/** Replay only a call whose timeline entry proves that it has an applied result. */
export async function callAction(workspaceId: string, request: ScientificActionRequest | DataDiffRequest, signal?: AbortSignal): Promise<ActionPoll> {
  const options = { params: { path: { workspace_id: workspaceId } }, ...(signal === undefined ? {} : { signal }) };
  const response = await (() => {
    switch (request.action) {
      case "set_question": return apiClient.POST("/api/studies/{workspace_id}/set_question", { ...options, body: request });
      case "edit_model": return apiClient.POST("/api/studies/{workspace_id}/edit_model", { ...options, body: request });
      case "prepare_data": return apiClient.POST("/api/studies/{workspace_id}/prepare_data", { ...options, body: request });
      case "fit": return apiClient.POST("/api/studies/{workspace_id}/fit", { ...options, body: request });
      case "simulate": return apiClient.POST("/api/studies/{workspace_id}/simulate", { ...options, body: request });
      case "data_diff": return apiClient.POST("/api/studies/{workspace_id}/data_diff", { ...options, body: request });
    }
  })();
  if (response.data === undefined) throw new Error(`Cannot read ${request.action} (${response.response.status}): ${JSON.stringify(response.error)}`);
  return response.data;
}

export async function readActionResult(workspaceId: string, call: TimelineRevision, signal?: AbortSignal) {
  if (call.record.attempt.outcome.status !== "applied" || call.record.attempt.request === null) throw new Error("The viewer cannot replay unsuccessful calls");
  const result = await callAction(workspaceId, call.record.attempt.request, signal);
  if (result.kind !== "completed" || result.attempt.outcome.status !== "applied") throw new Error("The saved applied call is unavailable");
  return result;
}

/** The result owns all traces; merging conversation envelopes is presentation only. */
export async function getLLMTraceForAction(workspaceId: string, call: TimelineRevision, traceIds: readonly string[]): Promise<LLMTrace> {
  const result = await readActionResult(workspaceId, call);
  const traces = traceIds.map((identity) => {
    const trace = result.traces[identity];
    if (!trace) throw new Error(`Saved trace is unavailable: ${identity}`);
    return trace;
  });
  return traces.reduce<LLMTrace>((merged, trace) => ({
    messages: [...merged.messages, ...trace.messages], model: trace.model || merged.model,
    total_time_seconds: merged.total_time_seconds + trace.total_time_seconds,
    usage: { input_tokens: merged.usage.input_tokens + trace.usage.input_tokens,
      output_tokens: merged.usage.output_tokens + trace.usage.output_tokens,
      reasoning_tokens: merged.usage.reasoning_tokens === null && trace.usage.reasoning_tokens === null ? null : (merged.usage.reasoning_tokens ?? 0) + (trace.usage.reasoning_tokens ?? 0) },
  }), { messages: [], model: "", total_time_seconds: 0, usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: null } });
}
