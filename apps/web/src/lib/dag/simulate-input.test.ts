import { describe, expect, it } from "vitest";
import { demoSimulationTrace } from "@/components/dag/__fixtures__/simulation-fixture";
import { buildSimulateInput } from "./simulate-input";
import { buildSimulationScenarios } from "./simulation-results";

const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

describe("dated simulation inputs", () => {
  it("pins the model and resolved historical start when editing interventions", () => {
    const result = scenarios.find((scenario) => scenario.key === "sim-4")!.result;
    const event = {
      target: result.design.interventions[0].target,
      value: 0.5,
      time: 110,
    };
    const input = buildSimulateInput(result, [event], 160);
    expect(input).toEqual({
      action: "simulate",
      model_revision: result.model.revision,
      start: 100,
      end: 160,
      interventions: [event],
    });
  });
  it("accepts an empty intervention list", () => {
    const result = scenarios[0].result;
    expect(buildSimulateInput(result, [], 60).interventions).toEqual([]);
  });
});
