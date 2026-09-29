import { createModelClient } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const client = createModelClient();

export function useSimulationTrajectories(workspaceId: string, at: string, enabled: boolean) {
  return useQuery({
    queryKey: ["simulation-trajectories", workspaceId, at],
    enabled,
    queryFn: async ({ signal }) => {
      const response = await client.GET(
        "/api/episodes/{workspace_id}/model/simulation-trajectories",
        {
          params: { path: { workspace_id: workspaceId }, query: { at } },
          signal,
        },
      );
      if (response.error) throw new Error(JSON.stringify(response.error));
      return response.data ?? null;
    },
    staleTime: Number.POSITIVE_INFINITY,
  });
}
