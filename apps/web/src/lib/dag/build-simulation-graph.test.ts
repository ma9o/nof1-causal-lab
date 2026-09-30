import { describe, expect, it } from "vitest";
import { constructs, edges } from "@/components/dag/__fixtures__/dag-base-fixtures";
import { buildSimulationGraph } from "@/lib/dag/build-simulation-graph";

describe("buildSimulationGraph", () => {
  it("routes DEMO dynamic and static causes and fitted persistence", () => {
    const built = buildSimulationGraph(constructs, edges, {
      dir: "RIGHT",
      showIndicators: false,
      showUnroll: true,
      indicators: [],
      persistenceNodes: ["internalizing_symptom_burden"],
    });
    const metadata = [...built.edgeMeta.values()];

    expect(metadata).toContainEqual({
      a: "internalizing_symptom_burden__p",
      b: "patient_taper_preference_beliefs",
      isSelf: false,
      crossSlice: true,
    });
    expect(metadata).toContainEqual({
      a: "external_stressful_events__p",
      b: "perceived_stress_burden",
      isSelf: false,
      crossSlice: true,
    });
    expect(metadata).toContainEqual({
      a: "natural_recovery_propensity",
      b: "internalizing_symptom_burden",
      isSelf: false,
      crossSlice: false,
    });
    expect(metadata).toContainEqual({
      a: "internalizing_symptom_burden__p",
      b: "internalizing_symptom_burden",
      isSelf: true,
      crossSlice: true,
    });
    expect(built.graph.nodes.some(({ id }) => id === "internalizing_symptom_burden__p")).toBe(true);
    expect(built.graph.nodes.some(({ id }) => id === "external_stressful_events__p")).toBe(true);
    expect(built.graph.nodes.some(({ id }) => id === "natural_recovery_propensity__p")).toBe(false);
  });
});
