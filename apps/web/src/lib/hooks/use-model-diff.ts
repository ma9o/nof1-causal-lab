import { createModelClient } from "@nof1-causal-lab/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";

const client = createModelClient();

export function useModelDiff(workspaceId: string, before: string, after: string | null) {
  return useQuery({
    queryKey: ["model-diff", workspaceId, before, after],
    enabled: after != null,
    queryFn:
      after !== null
        ? async ({ signal }) => {
            const response = await client.GET("/api/studies/{workspace_id}/model-diff", {
              params: {
                path: { workspace_id: workspaceId },
                query: { before, after },
              },
              signal,
            });
            if (response.error) throw new Error(JSON.stringify(response.error));
            return response.data;
          }
        : skipToken,
    // Both sides are immutable commits or revisions, so a diff never changes.
    staleTime: Number.POSITIVE_INFINITY,
  });
}
