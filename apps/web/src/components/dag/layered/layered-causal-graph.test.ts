import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { SimulationPaths, SimulationReport } from "@nof1-causal-lab/api-types";
import { treatment as dose, outcome as symptoms } from "@/lib/__fixtures__/model";
import { fittedSnapshot, modelRef } from "@/lib/__fixtures__/snapshot";
import { indexModel } from "@/lib/model-asset/entities";
import { graphEntities } from "@/lib/dag/layered-model";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { formatModelDate } from "@/lib/utils/format";
import { LayeredCausalGraph, type LayeredCausalGraphVariant } from "./layered-causal-graph";

// Only the asynchronous layout is stubbed; render the real graph and its overlays.
vi.mock("@/lib/hooks/use-dag-layout", () => ({
  useDagLayout: (graph: DagGraphInput) => ({
    width: 800,
    height: 400,
    isLayouting: false,
    nodes: graph.nodes.map((node, index) => ({ ...node, x: index * 100, y: 40 })),
    edges: graph.edges.map((edge) => ({
      ...edge,
      points: [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
      ],
    })),
  }),
}));

const base = fittedSnapshot;
const indexed = indexModel(base.model);
const entities = graphEntities(base, indexed);
const edge = fixtureValue(
  entities.edges.find((item) => item.cause.id === dose.id && item.effect.id === symptoms.id),
);
const model = {
  ...base,
  graph: {
    ...base.graph,
    construct_ids: [dose.id, symptoms.id],
    dynamic_construct_ids: [dose.id, symptoms.id],
    edge_ids: [edge.id],
  },
};

const report: SimulationReport = {
  evidence: {
    model: modelRef,
    draws: 1,
    seed: 0,
    origin_data: null,
    state_ids: [dose.id, symptoms.id],
    parameter_draws: {},
    latent_paths: "a".repeat(64),
    observations: "b".repeat(64),
    observation_layout: {
      variables: [],
      support_start_times: "c".repeat(64),
      support_end_times: "d".repeat(64),
      mask: "e".repeat(64),
    },
    reference_latent_paths: null,
    reference_observations: null,
    time_origin: "2026-01-01T00:00:00Z",
    times: [0, 1, 2, 3, 4, 5, 6, 7],
    design: {
      start: "2026-01-01",
      horizon: "7d",
      interventions: [
        { target: dose.id, after: "1d", value: 10 },
        { target: dose.id, after: "3d", value: 5 },
        { target: dose.id, after: "6d", value: 0 },
      ],
    },
    assignments: [
      { target: dose.id, time: 1, value: 10 },
      { target: dose.id, time: 3, value: 5 },
      { target: dose.id, time: 6, value: 0 },
    ],
  },
  law: { kind: "authored", interpretation: "prior_predictive" },
  findings: [],
  fit_reliability: "not_fitted",
  causal: { kind: "unavailable", reason: "This fixture covers intervention rendering." },
};
const paths: SimulationPaths = {
  effect_summary: null,
  reference_mean: null,
  manifest_effects: {},
  action_category_probabilities: {},
  reference_category_probabilities: {},
  times: report.evidence.times,
  time_origin: null,
  total_draws: 1,
  start: 0,
  count: 1,
  states: {
    [dose.id]: {
      label: dose.name,
      levels: null,
      action: [{ draw: 0, values: [9, 10, 9, 5, 6, 4, 0, 1] }],
      reference: [],
      frame: [0, 10],
    },
  },
  indicators: {},
  effect: null,
};

function render(variant: LayeredCausalGraphVariant, simulation: SimulationReport | null) {
  return renderToStaticMarkup(
    createElement(LayeredCausalGraph, {
      model,
      entities: indexed,
      simulation,
      simulationPaths: simulation ? paths : null,
      selection: null,
      onSelect: () => {},
      variant,
    }),
  );
}

describe("dated intervention overlay", () => {
  it.each([
    "asset",
    "workbench",
  ] as const)("preserves the model's arrows and their styling in the %s view", (variant) => {
    const arrows = (markup: string) => markup.match(/<path\b[^>]*marker-end=[^>]*>/g) ?? [];
    const baseline = arrows(render(variant, null));
    expect(baseline).toHaveLength(3);
    expect(arrows(render(variant, report))).toEqual(baseline);
  });

  it("shows every dated assignment without implying a continuously fixed value", () => {
    const markup = render("asset", report);
    expect(markup.match(/>3 assignments</g)).toHaveLength(1);
    for (const event of report.evidence.assignments) {
      expect(markup).toContain(
        `aria-label="${formatModelDate(event.time, report.evidence.time_origin)} (day ${event.time}): set to ${event.value}"`,
      );
    }
    expect(markup).not.toContain("do(");
    expect(markup).not.toContain("Cut by intervention");
  });
});
