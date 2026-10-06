import type { ModelDiffOutput, ModelSpec, TimelineRevision } from "@nof1-causal-lab/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";
import { readActionResult } from "@/lib/api/endpoints";

export type ResolvedModelDiff = ModelDiffOutput & {
  beforeModel: ModelSpec | null;
  afterModel: ModelSpec | null;
};

export function useModelDiff(
  workspaceId: string,
  before: string | null,
  after: string | null,
  attempts: readonly TimelineRevision[],
) {
  const call = attempts.find((entry) => {
    const { request, outcome } = entry.record.attempt;
    return (
      outcome.status === "applied" &&
      request?.action === "model_diff" &&
      request.input.before_ref === before &&
      request.input.after_ref === after
    );
  });
  return useQuery({
    queryKey: ["action-result", workspaceId, call?.call_id],
    queryFn: call ? ({ signal }) => readActionResult(workspaceId, call, signal) : skipToken,
    select: (result): ResolvedModelDiff | null =>
      result.action === "model_diff"
        ? {
            ...result.body,
            beforeModel: result.body.before_model,
            afterModel: result.body.after_model,
          }
        : null,
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}
