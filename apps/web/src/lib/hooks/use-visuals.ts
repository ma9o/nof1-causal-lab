"use client";

import type { IndicatorId, ModelSnapshot } from "@nof1-causal-lab/api-types";
import {
  historyView,
  pathsView,
  drawsView,
  type PathSeriesView,
  type SimulationPathsView,
} from "@/lib/model-asset/result-values";
import { presentEntries } from "@/lib/model-accessors";
import { viewedSimulation } from "@/lib/simulation-report";
import { useActionResult, useViewedSimulationResult } from "./use-model-snapshot";

export function useObservationHistory(model: ModelSnapshot, id: IndicatorId) {
  const query = useActionResult(model.workspace_id, model.state.data?.revision ?? model.commit_id);
  const result = query.data;
  const history =
    result?.action === "simulate"
      ? model.state.data && result.body.data[model.state.data.replicate_index]
      : result?.action === "prepare_data"
        ? result.body.data
        : null;
  const selected = history?.[id];
  return {
    ...query,
    data: selected
      ? historyView(selected, result?.action === "simulate" ? result.body.arrays : {})
      : null,
  };
}

/** Which saved draws a chart shows: a contiguous page, or every draw. */
export interface DrawSelection {
  readonly start: number;
  readonly count: number;
}

export const ALL_DRAWS: DrawSelection = { start: 0, count: Number.POSITIVE_INFINITY };

/**
 * The server returns every draw of the viewed model's simulation; a selection pages them.
 * Another model revision's simulation is never fetched.
 */
export function useSimulationPaths(model: ModelSnapshot, selection: DrawSelection = ALL_DRAWS) {
  const simulation = viewedSimulation(model);
  const query = useViewedSimulationResult(model, simulation !== null);
  const paths =
    query.data?.action === "simulate"
      ? query.data.body.paths
        ? pathsView(query.data.body.paths, query.data.body.arrays)
        : null
      : null;
  const { start, count } = selection;
  const page = (series: PathSeriesView): PathSeriesView => ({
    ...series,
    action: series.action.slice(start, start + count),
    reference: series.reference.slice(start, start + count),
  });
  const data: SimulationPathsView | null =
    paths && simulation
      ? {
          ...paths,
          start,
          count: Math.min(count, paths.total_draws - start),
          states: Object.fromEntries(
            presentEntries(paths.states).map(([id, series]) => [id, page(series)]),
          ),
          indicators: Object.fromEntries(
            presentEntries(paths.indicators).map(([id, series]) => [id, page(series)]),
          ),
          effect: paths.effect ? page(paths.effect) : null,
        }
      : null;
  return { ...query, data };
}

export function useParameterDraws(model: ModelSnapshot) {
  const query = useActionResult(
    model.workspace_id,
    model.state.current.model?.revision,
    model.fit != null,
  );
  return { ...query, data: query.data?.action === "fit" ? drawsView(query.data.body) : null };
}
