import {
  createModelClient,
  type ModelDiffReport,
  type ModelSpec,
} from "@nof1-causal-lab/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";

export type ResolvedModelDiff = ModelDiffReport & {
  beforeModel: ModelSpec | null;
  afterModel: ModelSpec | null;
};

const client = createModelClient();

export function useModelDiff(workspaceId: string, before: string | null, after: string | null) {
  return useQuery({
    queryKey: ["model-diff", workspaceId, before, after],
    enabled: before != null && after != null,
    queryFn:
      after !== null && before !== null
        ? async ({ signal }) => {
            const response = await client.GET("/api/studies/{workspace_id}/model-comparison", {
              params: { path: { workspace_id: workspaceId }, query: { before, after } },
              signal,
            });
            if (response.error) throw new Error(JSON.stringify(response.error));
            return {
              ...response.data,
              beforeModel: response.data.before_model,
              afterModel: response.data.after_model,
            } satisfies ResolvedModelDiff;
          }
        : skipToken,
    // Both sides are immutable commits or revisions, so a diff never changes.
    staleTime: Number.POSITIVE_INFINITY,
  });
}
