"use client";

import type { IndicatorId, ModelSnapshot, PathSeries, SimulationPaths } from "@nof1-causal-lab/api-types";
import { presentEntries } from "@/lib/model-accessors";
import { useActionResult } from "./use-model-snapshot";

export function useObservationHistory(model: ModelSnapshot, id: IndicatorId) {
  const query = useActionResult(model.workspace_id, model.metadata?.source.ref.revision ?? model.commit_id);
  return { ...query, data: query.data?.observation_histories[id] ?? null };
}

export function usePredictiveHistory(model: ModelSnapshot, id: IndicatorId) {
  const query = useActionResult(model.workspace_id, model.predictive?.source.ref.revision ?? model.commit_id);
  return { ...query, data: query.data?.predictive_overlays[id] ?? null };
}

/** The server returns every draw; the existing draw pager selects which ones are displayed. */
export function useSimulationPaths(model: ModelSnapshot, start = 0, count = 24) {
  const query = useActionResult(model.workspace_id, model.simulation?.source.ref.revision ?? model.commit_id, model.simulation != null);
  const paths = query.data?.simulation_paths;
  const page = (series: PathSeries): PathSeries => ({ ...series, action: series.action.slice(start, start + count), reference: series.reference.slice(start, start + count) });
  const data: SimulationPaths | null = paths ? {
    ...paths, start, count: Math.min(count, paths.total_draws - start),
    states: Object.fromEntries(presentEntries(paths.states).map(([id, series]) => [id, page(series)])),
    indicators: Object.fromEntries(presentEntries(paths.indicators).map(([id, series]) => [id, page(series)])),
    effect: paths.effect ? page(paths.effect) : null,
  } : null;
  return { ...query, data };
}

export function useParameterDraws(model: ModelSnapshot) {
  const query = useActionResult(model.workspace_id, model.model?.source.ref.revision ?? model.commit_id, model.fit != null);
  return { ...query, data: query.data?.parameter_draws ?? null };
}
