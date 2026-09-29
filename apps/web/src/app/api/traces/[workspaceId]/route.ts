import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { NextResponse } from "next/server";
import {
  EpisodeRunError,
  getOperationTraceIndex,
  getEpisodeTrace,
} from "@/lib/server/episode-runs";
import { normalizeWorkspaceId } from "@/lib/workspace-id";

export const dynamic = "force-dynamic";

/** Concatenate a transition's subroutine traces into one panel-renderable trace. */
function mergeTraces(traces: LLMTrace[]): LLMTrace {
  const merged: LLMTrace = {
    messages: [],
    model: "",
    total_time_seconds: 0,
    usage: { input_tokens: 0, output_tokens: 0, reasoning_tokens: null },
  };
  let hasReasoningTokens = false;
  for (const trace of traces) {
    merged.messages.push(...trace.messages);
    merged.model = trace.model || merged.model;
    merged.total_time_seconds += trace.total_time_seconds;
    merged.usage.input_tokens += trace.usage.input_tokens;
    merged.usage.output_tokens += trace.usage.output_tokens;
    if (trace.usage.reasoning_tokens != null) {
      hasReasoningTokens = true;
      merged.usage.reasoning_tokens =
        (merged.usage.reasoning_tokens ?? 0) + trace.usage.reasoning_tokens;
    }
  }
  if (!hasReasoningTokens) {
    merged.usage.reasoning_tokens = null;
  }
  return merged;
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ workspaceId: string }> },
) {
  const { workspaceId } = await params;
  const safeWorkspaceId = normalizeWorkspaceId(workspaceId);
  if (!safeWorkspaceId) {
    return NextResponse.json({ error: "Invalid workspaceId format" }, { status: 400 });
  }

  const search = new URL(request.url).searchParams;
  const artifactId = search.get("artifact")?.trim();
  const commitId = search.get("commitId")?.trim();
  if (!artifactId && !commitId) {
    return NextResponse.json({ error: "Missing artifact id or action commit" }, { status: 400 });
  }

  try {
    if (commitId) {
      if (!/^[0-9a-f]{40}$/.test(commitId)) {
        return NextResponse.json({ error: "Invalid action commit" }, { status: 400 });
      }
      // The caller names the action's trace IDs from the journal it already holds,
      // so this read never re-downloads the whole journal.
      const traceIds = search.getAll("trace").filter((traceId) => traceId.length > 0);
      if (traceIds.length === 0) {
        return NextResponse.json({ error: "No traces for this action" }, { status: 404 });
      }
      const traces = await Promise.all(
        traceIds.map((traceId) => getEpisodeTrace(safeWorkspaceId, commitId, traceId)),
      );
      return NextResponse.json(mergeTraces(traces));
    }
    const index = await getOperationTraceIndex(safeWorkspaceId, artifactId as string);
    if (index.trace_ids.length === 0) {
      return NextResponse.json({ error: "No traces for this artifact" }, { status: 404 });
    }
    const traces = await Promise.all(
      index.trace_ids.map((traceId) => getEpisodeTrace(safeWorkspaceId, index.commit_id, traceId)),
    );
    return NextResponse.json(mergeTraces(traces));
  } catch (error) {
    const status = error instanceof EpisodeRunError && error.status === 404 ? 404 : 502;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status },
    );
  }
}
