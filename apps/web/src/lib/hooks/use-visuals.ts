"use client";

import { presentEntries } from "@/lib/model-accessors";

import {
  createModelClient,
  type IndicatorId,
  type ModelSnapshot,
} from "@nof1-causal-lab/api-types";
import type { paths } from "@nof1-causal-lab/api-types/src/generated/model-api";
import { useQuery } from "@tanstack/react-query";

export type MechanismViewport = Required<
  paths["/api/studies/{workspace_id}/model/visuals/mechanism"]["post"]["requestBody"]["content"]["application/json"]
>;

const client = createModelClient();
const pinned = (model: ModelSnapshot) => ({
  path: { workspace_id: model.workspace_id },
  query: { at: model.commit_id, branch: model.branch },
});
const key = (model: ModelSnapshot) => [model.workspace_id, model.commit_id, model.branch];

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
        client.GET("/api/studies/{workspace_id}/model/visuals/observations/{indicator_id}", {
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
        client.GET("/api/studies/{workspace_id}/model/visuals/predictive/{indicator_id}", {
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
        client.GET("/api/studies/{workspace_id}/model/visuals/simulation", {
          params: { ...pinned(model), query: { ...pinned(model).query, start, count } },
          signal,
        }),
      ),
    enabled: model.simulation != null,
    staleTime: Infinity,
  });
}

export function useParameterDraws(model: ModelSnapshot) {
  return useQuery({
    queryKey: ["parameter-draws", ...key(model)],
    queryFn: ({ signal }) =>
      read(
        client.GET("/api/studies/{workspace_id}/model/visuals/parameters", {
          params: pinned(model),
          signal,
        }),
      ),
    enabled: model.fit != null,
    staleTime: Infinity,
  });
}

export function useMechanismCurves(model: ModelSnapshot, request: MechanismViewport) {
  return useQuery({
    queryKey: ["mechanism-curves", ...key(model), request],
    queryFn: ({ signal }) =>
      read(
        client.POST("/api/studies/{workspace_id}/model/visuals/mechanism", {
          params: pinned(model),
          body: { ...request, held: Object.fromEntries(presentEntries(request.held)) },
          signal,
        }),
      ),
    staleTime: Infinity,
    retry: false,
  });
}
