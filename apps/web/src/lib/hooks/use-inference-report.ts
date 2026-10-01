"use client";

import { createModelClient, type ModelSnapshot } from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const modelClient = createModelClient();

/** The full inference report at a pinned commit, including per-draw diagnostics. */
export function useInferenceReport(model: ModelSnapshot) {
  const { workspace_id: workspaceId, commit_id: commitId, branch } = model.context;
  return useQuery({
    enabled: model.findings.fit != null,
    queryKey: ["inference-report", workspaceId, commitId, branch],
    queryFn: async ({ signal }) => {
      const { data, error, response } = await modelClient.GET(
        "/api/studies/{workspace_id}/model/inference-report",
        {
          params: { path: { workspace_id: workspaceId }, query: { at: commitId, branch } },
          signal,
        },
      );
      if (error) {
        throw new Error(
          `Cannot read inference report (${response.status}): ${JSON.stringify(error)}`,
        );
      }
      return data ?? null;
    },
    // A pinned commit's report never changes.
    staleTime: Number.POSITIVE_INFINITY,
  });
}
