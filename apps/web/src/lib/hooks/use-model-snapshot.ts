"use client";

import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const modelClient = createModelClient();

export function useModelSnapshot(workspaceId: string, atSeq?: number) {
  return useQuery({
    queryKey: ["model-snapshot", workspaceId, atSeq ?? "latest"],
    queryFn: async ({ signal }) => {
      const { data, error, response } = await modelClient.GET(
        "/api/episodes/{workspace_id}/model",
        {
          params: { path: { workspace_id: workspaceId }, query: { at_seq: atSeq } },
          signal,
        },
      );
      if (error || !data) {
        throw new Error(
          `Cannot read model revision (${response.status}): ${JSON.stringify(error)}`,
        );
      }
      return data;
    },
    staleTime: atSeq === undefined ? 0 : Infinity,
  });
}
