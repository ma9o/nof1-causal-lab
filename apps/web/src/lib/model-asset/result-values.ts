import type {
  ArrayVector,
  FitOutput,
  NumericalArray,
  InferenceReport,
  InferenceReportDetail,
  ObservationHistory,
  ParameterDrawColumn,
  PathSeries,
  RecordedPath,
  ScalarValues,
  SimulationPaths,
} from "@nof1-causal-lab/api-types";

export type ResultArrays = Readonly<Record<string, NumericalArray | undefined>>;
export type HistoryView = Omit<ObservationHistory, "values" | "support_start" | "support_end"> & {
  readonly values: readonly (number | null)[];
  readonly support_start: readonly (number | null)[];
  readonly support_end: readonly (number | null)[];
};
export type PathView = Omit<RecordedPath, "values"> & {
  readonly values: readonly (number | null)[];
};
export type PathSeriesView = Omit<PathSeries, "action" | "reference"> & {
  readonly action: readonly PathView[];
  readonly reference: readonly PathView[];
};
export type SimulationPathsView = Omit<SimulationPaths, "states" | "indicators" | "effect"> & {
  readonly states: Readonly<Record<string, PathSeriesView | undefined>>;
  readonly indicators: Readonly<Record<string, PathSeriesView | undefined>>;
  readonly effect: PathSeriesView | null;
};
export type DrawColumnView = Omit<ParameterDrawColumn, "values"> & {
  readonly values: readonly number[];
};

/** Decode a recorded vector selection; all statistical summaries already belong to the result. */
export function scalarValues(
  value: ScalarValues,
  arrays: ResultArrays,
): readonly (number | null)[] {
  if (!("array_ref" in value)) return value;
  const selection: ArrayVector = value;
  const array = required(arrays[selection.array_ref]);
  const axis = selection.indices.indexOf(null);
  const strides = array.shape.map((_, index) =>
    array.shape.slice(index + 1).reduce((a, b) => a * b, 1),
  );
  const offset = selection.indices.reduce<number>(
    (sum, index, dimension) => sum + (index ?? 0) * required(strides[dimension]),
    0,
  );
  const mask = selection.mask ? scalarValues(selection.mask, arrays) : null;
  const stop = selection.stop ?? required(array.shape[axis]);
  return Array.from({ length: stop - selection.start }, (_, index) => {
    if (mask && !mask[index]) return null;
    const scalar = required(
      array.values[offset + (index + selection.start) * required(strides[axis])],
    );
    return typeof scalar === "string" ? null : Number(scalar);
  });
}

export function historyView(history: ObservationHistory, arrays: ResultArrays): HistoryView {
  return {
    ...history,
    values: scalarValues(history.values, arrays),
    support_start: scalarValues(history.support_start, arrays),
    support_end: scalarValues(history.support_end, arrays),
  };
}

export function pathsView(paths: SimulationPaths, arrays: ResultArrays): SimulationPathsView {
  const path = (value: RecordedPath): PathView => ({
    ...value,
    values: scalarValues(value.values, arrays),
  });
  const series = (value: PathSeries): PathSeriesView => ({
    ...value,
    action: value.action.map(path),
    reference: value.reference.map(path),
  });
  return {
    ...paths,
    states: Object.fromEntries(
      Object.entries(paths.states).flatMap(([id, value]) => (value ? [[id, series(value)]] : [])),
    ),
    indicators: Object.fromEntries(
      Object.entries(paths.indicators).flatMap(([id, value]) =>
        value ? [[id, series(value)]] : [],
      ),
    ),
    effect: paths.effect ? series(paths.effect) : null,
  };
}

export function drawsView(
  result: FitOutput,
):
  | { readonly kind: "available"; readonly value: readonly DrawColumnView[] }
  | Extract<FitOutput["parameter_draws"], { kind: "unavailable" }> {
  const draws = result.parameter_draws;
  return draws.kind === "available"
    ? {
        ...draws,
        value: draws.value.map((column) => ({
          ...column,
          values: scalarValues(column.values, result.arrays).map(required),
        })),
      }
    : draws;
}

/** Missing coordinates are a corrupt result, never an alternate scientific value. */
function required<T>(value: T | null | undefined): T {
  if (value === undefined || value === null)
    throw new Error("Saved numerical result has an invalid coordinate");
  return value;
}

export type InferenceDetailView = Omit<
  InferenceReportDetail,
  "trace_data" | "divergent" | "initial_latent_delta" | "final_latent_delta"
> & {
  readonly trace_data: readonly {
    readonly subject: ParameterDrawColumn["subject"];
    readonly chains: readonly (readonly number[])[];
  }[];
  readonly divergent: readonly boolean[] | null;
  readonly initial_latent_delta: readonly (readonly number[])[] | null;
  readonly final_latent_delta: readonly (readonly number[])[] | null;
};
export type InferenceView = Omit<InferenceReport, "detail"> & {
  readonly detail: InferenceDetailView;
};

/** Resolve stored chart coordinates without calculating any diagnostic. */
export function inferenceView(result: FitOutput): InferenceView | null {
  const report = result.inference_report;
  if (!report) return null;
  const detail = report.detail;
  const vector = (value: ScalarValues) => scalarValues(value, result.arrays).map(required);
  return {
    ...report,
    detail: {
      ...detail,
      trace_data: detail.trace_data.map((trace) => ({
        ...trace,
        chains: trace.chains.map(vector),
      })),
      divergent:
        typeof detail.divergent === "string"
          ? required(result.arrays[detail.divergent]).values.map(Boolean)
          : detail.divergent,
      initial_latent_delta: detail.initial_latent_delta?.map(vector) ?? null,
      final_latent_delta: detail.final_latent_delta?.map(vector) ?? null,
    },
  };
}
