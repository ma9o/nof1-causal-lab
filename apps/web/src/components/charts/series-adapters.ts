import type { DataVariableDiff, PathSeries, PPCOverlay } from "@nof1-causal-lab/api-types";
import { CHART_COLORS, chainColor } from "./chart-tokens";
import type { DrawLayer, DrawRow, DrawsChartProps } from "./draws-chart";
import { DAY_MS } from "./plot-geometry";

/** Saved timestamps aligned for plotting: no resampling, imputation or pooled statistics. */

export type ArmChoice = "both" | "reference" | "intervened";

const pathRows = (paths: PathSeries["action"], arm: string): DrawRow[] =>
  paths.map((path) => ({
    key: `${arm}-${path.draw}`,
    label: `draw ${path.draw + 1}`,
    values: path.values,
  }));

/** A saved series by arm: colour names the arm, and matching draw numbers are paired. */
export function pathLayers(series: PathSeries, arms: ArmChoice, points: boolean): DrawLayer[] {
  const paired = series.reference.length > 0;
  const reference: DrawLayer[] =
    paired && arms !== "intervened"
      ? [
          {
            key: "reference",
            label: "Reference",
            color: CHART_COLORS.reference,
            rows: pathRows(series.reference, "reference"),
            points,
          },
        ]
      : [];
  const action: DrawLayer[] =
    !paired || arms !== "reference"
      ? [
          {
            key: "intervened",
            label: paired ? "Intervened" : "Simulation",
            color: paired ? CHART_COLORS.intervened : CHART_COLORS.reference,
            rows: pathRows(series.action, "action"),
            points,
          },
        ]
      : [];
  return [...reference, ...action];
}

/** Paired effects: each draw's intervened path minus the same draw's reference path. */
export function effectLayers(series: PathSeries): DrawLayer[] {
  return [
    {
      key: "effect",
      label: "Paired effect",
      color: CHART_COLORS.intervened,
      rows: pathRows(series.action, "effect"),
    },
  ];
}

/** Replicates exist only where the indicator was measured; sparse ones are dots, not lines. */
export function overlayChart(overlay: PPCOverlay, label: string): Omit<DrawsChartProps, "height"> {
  const measured = overlay.observed.filter((value) => value != null).length;
  return {
    label,
    times: overlay.times,
    timeOrigin: overlay.time_origin,
    layers: [
      {
        key: "replicates",
        label: "Replicate",
        color: CHART_COLORS.replicate,
        rows: overlay.spaghetti_draws.map((values, index) => ({
          key: `replicate-${index}`,
          label: `replicate ${index + 1}`,
          values,
        })),
        points: measured * 2 < overlay.observed.length,
      },
    ],
    observed: { label: "Observed", values: overlay.observed },
    frame: overlay.frame,
  };
}

const CHANGE_COLORS = {
  added: CHART_COLORS.added,
  removed: CHART_COLORS.removed,
  revised: CHART_COLORS.revised,
} as const;

/** Every saved history of one variable on the union of their anchors, with point changes. */
export function dataComparisonChart(variable: DataVariableDiff): Omit<DrawsChartProps, "height"> {
  const referenceSide =
    variable.predictive.kind === "comparison" ? variable.predictive.reference_side : null;
  const histories = [...variable.left, ...variable.right];
  const definition = histories.flatMap((history) => history.variable ?? []).at(0);
  const anchors = [
    ...new Set(histories.flatMap((history) => history.points.map((point) => point.anchor_time))),
  ].sort();
  const origin = anchors.at(0) ?? null;
  const align = (points: readonly { anchor_time: string; value: number | null }[]) => {
    const byAnchor = new Map(points.map((point) => [point.anchor_time, point.value]));
    return anchors.map((anchor) => byAnchor.get(anchor) ?? null);
  };
  const sides: DrawLayer[] = (["left", "right"] as const).flatMap((side): DrawLayer[] =>
    side === referenceSide || variable[side].length === 0
      ? []
      : [
          {
            key: side,
            label: side === "left" ? "Left" : "Right",
            color: referenceSide ? CHART_COLORS.replicate : chainColor(side === "left" ? 0 : 1),
            rows: variable[side].map((history, index) => ({
              key: `${side}-${index}`,
              label: `history ${index + 1}`,
              values: align(history.points),
            })),
            points: referenceSide === null,
          },
        ],
  );
  const reference = referenceSide ? variable[referenceSide] : [];
  const [single] = reference.length === 1 ? reference : [];
  const references: DrawLayer[] =
    reference.length > 1
      ? [
          {
            key: "observed",
            label: "Observed",
            color: CHART_COLORS.observed,
            rows: reference.map((history, index) => ({
              key: `observed-${index}`,
              label: `history ${index + 1}`,
              values: align(history.points),
            })),
            strong: true,
          },
        ]
      : [];
  const changes = (["added", "removed", "revised"] as const).flatMap((kind) =>
    (["left", "right"] as const).flatMap((side): DrawLayer[] => {
      const points = variable.changes.filter((change) => change.kind === kind);
      if (points.length === 0) return [];
      const values = new Map(
        points.map((change) => [
          change.kind === "removed" ? change.before.anchor_time : change.after.anchor_time,
          side === "left"
            ? change.kind === "added"
              ? null
              : change.before.value
            : change.kind === "removed"
              ? null
              : change.after.value,
        ]),
      );
      return [
        {
          key: `${kind}-${side}`,
          label: `${kind} · ${side}`,
          color: CHANGE_COLORS[kind],
          rows: [
            {
              key: `${kind}-${side}`,
              label: `${kind} points`,
              values: anchors.map((anchor) => values.get(anchor) ?? null),
            },
          ],
          points: true,
          strong: true,
        },
      ];
    }),
  );
  return {
    label: `${definition?.name ?? variable.indicator_id}: data comparison`,
    times:
      origin === null
        ? []
        : anchors.map((anchor) => (Date.parse(anchor) - Date.parse(origin)) / DAY_MS),
    timeOrigin: histories.every(
      (history) => history.variable === null || history.time_origin !== null,
    )
      ? origin
      : null,
    layers: [...sides, ...references, ...changes],
    ...(single ? { observed: { label: "Observed", values: align(single.points) } } : {}),
    levels: definition?.ordinal_levels ?? definition?.categorical_levels ?? null,
    xLabel: "Days from first anchor",
  };
}
