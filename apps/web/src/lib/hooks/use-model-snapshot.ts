"use client";

import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const modelClient = createModelClient();

export function useModelSnapshot(
  workspaceId: string,
  commitId?: string,
  branch = "main",
  enabled = true,
) {
  return useQuery({
    queryKey: ["model-snapshot", workspaceId, commitId ?? "latest", branch],
    enabled,
    queryFn: async ({ signal }) => {
      const { data, error, response } = await modelClient.GET("/api/studies/{workspace_id}/model", {
        params: { path: { workspace_id: workspaceId }, query: { at: commitId, branch } },
        signal,
      });
      if (error) {
        throw new Error(
          `Cannot read model revision (${response.status}): ${JSON.stringify(error)}`,
        );
      }
      return data;
    },
    // Keep the workbench mounted while another version loads.
    placeholderData: (previous, query) =>
      query?.queryKey[1] === workspaceId ? previous : undefined,
    staleTime: commitId === undefined ? 0 : Infinity,
  });
}
