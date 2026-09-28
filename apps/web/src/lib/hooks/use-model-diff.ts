import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const client = createModelClient();

export function useModelDiff(workspaceId: string, before: string, after: string | null) {
  return useQuery({
    queryKey: ["model-diff", workspaceId, before, after],
    enabled: after != null,
    queryFn: async ({ signal }) => {
      const response = await client.GET("/api/episodes/{workspace_id}/model-diff", {
        params: {
          path: { workspace_id: workspaceId },
          query: { before, after: after! },
        },
        signal,
      });
      if (response.error || !response.data) throw new Error(JSON.stringify(response.error));
      return response.data;
    },
    staleTime: 60_000,
  });
}
