import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { SimulationPaths, SimulationReport } from "@nof1-causal-lab/api-types";
import { demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { demoSimulationTrace } from "@/components/dag/__fixtures__/simulation-fixture";
import { indexModel } from "@/lib/model-asset/entities";
import { graphEntities } from "@/lib/dag/layered-model";
import { buildSimulationScenarios } from "@/lib/dag/simulation-results";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
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

const model = structuredClone(demoSnapshotAt(8));
const indexed = indexModel(model.model?.value);
const entities = graphEntities(model, indexed);
const dose = entities.constructs.find((item) => item.name === "escitalopram_dose_taken")!;
const symptoms = entities.constructs.find((item) => item.name === "internalizing_symptom_burden")!;
const edge = entities.edges.find(
  (item) => item.cause.id === dose.id && item.effect.id === symptoms.id,
)!;
model.findings.graph.construct_ids = [dose.id, symptoms.id];
model.findings.graph.dynamic_construct_ids = [dose.id, symptoms.id];
model.findings.graph.edge_ids = [edge.id];

const report: SimulationReport = {
  ...buildSimulationScenarios({ trace: demoSimulationTrace })[0].result,
  time_origin: null,
  times: [0, 1, 2, 3, 4, 5, 6, 7],
  design: {
    start: 0,
    end: 7,
    interventions: [
      { target: dose.id, time: 1, value: 10 },
      { target: dose.id, time: 3, value: 5 },
      { target: dose.id, time: 6, value: 0 },
    ],
  },
};
const paths: SimulationPaths = {
  times: report.times,
  time_origin: null,
  total_draws: 1,
  start: 0,
  count: 1,
  states: {
    [dose.id]: {
      label: dose.name,
      action: [{ draw: 0, values: [9, 10, 9, 5, 6, 4, 0, 1] }],
      reference: [],
    },
  },
  indicators: {},
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
    for (const event of report.design.interventions) {
      expect(markup).toContain(`aria-label="Day ${event.time}: set to ${event.value}"`);
    }
    expect(markup).not.toContain("do(");
    expect(markup).not.toContain("Cut by intervention");
  });
});
