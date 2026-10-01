import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const client = createModelClient();

export function useDataDiff(workspaceId: string, commitId: string | null) {
  return useQuery({
    queryKey: ["data-diff", workspaceId, commitId],
    enabled: commitId !== null,
    queryFn: async ({ signal }) => {
      const response = await client.GET("/api/studies/{workspace_id}/data-diff/{commit_id}", {
        params: { path: { workspace_id: workspaceId, commit_id: commitId! } },
        signal,
      });
      if (response.error) throw new Error(JSON.stringify(response.error));
      return response.data;
    },
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useModelDiff(workspaceId: string, before: string, after: string | null) {
  return useQuery({
    queryKey: ["model-diff", workspaceId, before, after],
    enabled: after != null,
    queryFn: async ({ signal }) => {
      const response = await client.GET("/api/studies/{workspace_id}/model-diff", {
        params: {
          path: { workspace_id: workspaceId },
          query: { before, after: after! },
        },
        signal,
      });
      if (response.error) throw new Error(JSON.stringify(response.error));
      return response.data;
    },
    // Both sides are immutable commits or revisions, so a diff never changes.
    staleTime: Number.POSITIVE_INFINITY,
  });
}
