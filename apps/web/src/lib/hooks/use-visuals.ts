"use client";

import {
  createModelClient,
  type IndicatorId,
  type MechanismViewRequest,
  type ModelSnapshot,
} from "@nof1-causal-lab/api-types";
import { useQuery } from "@tanstack/react-query";

const client = createModelClient();
const pinned = (model: ModelSnapshot) => ({
  path: { workspace_id: model.context.workspace_id },
  query: { at: model.context.commit_id, branch: model.context.branch },
});
const key = (model: ModelSnapshot) => [
  model.context.workspace_id,
  model.context.commit_id,
  model.context.branch,
];

async function read<T>(request: Promise<{ data?: T; error?: unknown; response: Response }>) {
  const { data, error, response } = await request;
  if (error) throw new Error(`Cannot read plot (${response.status}): ${JSON.stringify(error)}`);
  return data ?? null;
}

export function useObservationHistory(model: ModelSnapshot, id: IndicatorId) {
  return useQuery({
    queryKey: ["observation-history", ...key(model), id],
    queryFn: ({ signal }) =>
      read(
        client.GET("/api/episodes/{workspace_id}/model/visuals/observations/{indicator_id}", {
          params: { ...pinned(model), path: { ...pinned(model).path, indicator_id: id } },
          signal,
        }),
      ),
    staleTime: Infinity,
  });
}

export function usePredictiveHistory(model: ModelSnapshot, id: IndicatorId) {
  return useQuery({
    queryKey: ["predictive-history", ...key(model), id],
    queryFn: ({ signal }) =>
      read(
        client.GET("/api/episodes/{workspace_id}/model/visuals/predictive/{indicator_id}", {
          params: { ...pinned(model), path: { ...pinned(model).path, indicator_id: id } },
          signal,
        }),
      ),
    staleTime: Infinity,
  });
}

export function useSimulationPaths(model: ModelSnapshot, start = 0, count = 24) {
  return useQuery({
    queryKey: ["simulation-paths", ...key(model), start, count],
    queryFn: ({ signal }) =>
      read(
        client.GET("/api/episodes/{workspace_id}/model/visuals/simulation", {
          params: { ...pinned(model), query: { ...pinned(model).query, start, count } },
          signal,
        }),
      ),
    enabled: model.findings.simulation != null,
    staleTime: Infinity,
  });
}

export function useParameterDraws(model: ModelSnapshot) {
  return useQuery({
    queryKey: ["parameter-draws", ...key(model)],
    queryFn: ({ signal }) =>
      read(
        client.GET("/api/episodes/{workspace_id}/model/visuals/parameters", {
          params: pinned(model),
          signal,
        }),
      ),
    enabled: model.findings.fit != null,
    staleTime: Infinity,
  });
}

export function useMechanismCurves(model: ModelSnapshot, request: MechanismViewRequest) {
  return useQuery({
    queryKey: ["mechanism-curves", ...key(model), request],
    queryFn: ({ signal }) =>
      read(
        client.POST("/api/episodes/{workspace_id}/model/visuals/mechanism", {
          params: pinned(model),
          body: request,
          signal,
        }),
      ),
    staleTime: Infinity,
    retry: false,
  });
}
