import {
  createModelClient,
  type ModelDiffReport,
  type ModelSpec,
} from "@nof1-causal-lab/api-types";
import { skipToken, useQuery } from "@tanstack/react-query";

export type ResolvedModelDiff = ModelDiffReport & { beforeModel: ModelSpec; afterModel: ModelSpec };

const client = createModelClient();

export function useModelDiff(workspaceId: string, before: string | null, after: string | null) {
  return useQuery({
    queryKey: ["model-diff", workspaceId, before, after],
    enabled: before != null && after != null,
    queryFn:
      after !== null && before !== null
        ? async ({ signal }) => {
            const response = await client.GET("/api/studies/{workspace_id}/model-diff", {
              params: {
                path: { workspace_id: workspaceId },
                query: { before, after },
              },
              signal,
            });
            if (response.error) throw new Error(JSON.stringify(response.error));
            const snapshots = await Promise.all(
              [before, after].map(async (at) => {
                const snapshot = await client.GET("/api/studies/{workspace_id}/model", {
                  params: { path: { workspace_id: workspaceId }, query: { at } },
                  signal,
                });
                if (snapshot.error) throw new Error(JSON.stringify(snapshot.error));
                if (!snapshot.data.model) throw new Error("Compared revision has no model");
                return snapshot.data.model.value;
              }),
            );
            const beforeModel = snapshots[0],
              afterModel = snapshots[1];
            if (!beforeModel || !afterModel) throw new Error("Compared model read is incomplete");
            return { ...response.data, beforeModel, afterModel } satisfies ResolvedModelDiff;
          }
        : skipToken,
    // Both sides are immutable commits or revisions, so a diff never changes.
    staleTime: Number.POSITIVE_INFINITY,
  });
}
