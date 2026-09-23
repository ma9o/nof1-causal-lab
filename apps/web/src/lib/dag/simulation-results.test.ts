import { modelConstructs } from "@/lib/model-accessors";
import type { SimulationResult } from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import { demoModel, demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { demoSimulationTrace } from "@/components/dag/__fixtures__/simulation-fixture";
import { buildSimulationScenarios, buildEdgePosteriors } from "@/lib/dag/simulation-results";

const fixtureScenarios = buildSimulationScenarios({ trace: demoSimulationTrace });
const interventionResult = fixtureScenarios.find((scenario) => scenario.key === "sim-5")?.result;
const counterfactualResult = fixtureScenarios.find((scenario) => scenario.key === "sim-4")?.result;

if (!interventionResult || !counterfactualResult) {
  throw new Error("The canonical DEMO trace is missing its materialized test scenarios.");
}

/** A refinement assistant turn carrying a live (object-valued) simulation result. */
function refinementSimMessage(
  toolCallId: string,
  result: SimulationResult,
  blurb = "Done.",
  input: unknown = {},
): UIMessage {
  return {
    id: `${toolCallId}-message`,
    role: "assistant",
    parts: [
      { type: "text", text: blurb },
      {
        type: "dynamic-tool",
        toolCallId,
        toolName: "simulate",
        state: "output-available",
        input,
        output: result,
      },
    ],
  };
}

describe("buildSimulationScenarios — interventions from a persisted trace", () => {
  it("recovers interventions from the trace where tool_result is a JSON string (reload path)", () => {
    const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

    expect(scenarios).toHaveLength(5);
    expect(scenarios.every((scenario) => scenario.provenance === "intervention")).toBe(true);

    const newest = scenarios[0];
    expect(newest.key).toBe("sim-5");
    expect(newest.result.request.start.kind).toBe("baseline");
    expect(newest.title).toBe("do(taper_speed_dose_reduction set 0.9)");
    expect(newest.requestedHorizonDays).toBe(60);
    expect(newest.userQuery).toContain("taper speed is raised sharply");
    // The assistant text beside the tool call becomes the scenario blurb.
    expect(newest.blurb).toContain("Rapid taper");
    // String-coerced result round-trips to the structured object.
    expect(newest.result.summary.mean).toBe(interventionResult.summary.mean);
    expect(newest.result.trajectories[newest.result.request.outcome]).toBeDefined();
  });

  it("captures abducted counterfactual fields and manifest projection", () => {
    const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

    const counterfactual = scenarios.find(
      (scenario) => scenario.result.request.start.kind === "abducted",
    );
    expect(counterfactual?.key).toBe("sim-4");
    expect(counterfactual?.result.summary.mean).toBe(counterfactualResult.summary.mean);

    // Manifest projection carried through on the set-mode simulation.
    const setMode = scenarios.find((scenario) => scenario.key === "sim-5");
    expect(setMode?.manifestEffects).toHaveProperty("state_of_mind_valence");
  });
});

describe("buildSimulationScenarios — trace ∪ extra messages", () => {
  it("dedupes by tool-call id with the extra-message copy winning and ranked newest", () => {
    const edited: SimulationResult = {
      ...interventionResult,
      summary: { ...interventionResult.summary, mean: 0.99 },
    };

    const scenarios = buildSimulationScenarios({
      trace: demoSimulationTrace,
      extraMessages: [refinementSimMessage("sim-5", edited)],
    });

    // sim-5 is not duplicated…
    expect(scenarios).toHaveLength(5);
    expect(scenarios.filter((scenario) => scenario.key === "sim-5")).toHaveLength(1);
    // …the refinement copy wins and leads the interventions.
    expect(scenarios[0].key).toBe("sim-5");
    expect(scenarios[0].result.summary.mean).toBe(0.99);
    expect(scenarios[0].requestedHorizonDays).toBe(edited.request.readout.horizon_days);
  });

  it("orders production-valid interventions newest-first", () => {
    const scenarios = buildSimulationScenarios({ trace: demoSimulationTrace });

    expect(scenarios.map((scenario) => scenario.key)).toEqual([
      "sim-5",
      "sim-4",
      "sim-3",
      "sim-2",
      "sim-1",
    ]);
  });
});

describe("owned graph findings", () => {
  it("uses persistent owners even when the display name changes", () => {
    const structure = structuredClone(demoModel);
    const edge = structure.edges.find(
      (item) => demoModelSnapshot.findings.fit!.value.edge_estimates[item.id],
    )!;
    modelConstructs(structure).find((item) => item.id === edge.cause.id)!.name = "renamed";
    expect(
      buildEdgePosteriors({
        latentStructure: structure,
        estimates: demoModelSnapshot.findings.fit!.value.edge_estimates,
      }),
    ).toHaveProperty(
      `renamed→${modelConstructs(demoModel).find((item) => item.id === edge.effect.id)!.name}`,
    );
  });
  it("does not parse parameter descriptions or invent absent findings", () => {
    expect(buildEdgePosteriors({ latentStructure: demoModel, estimates: {} })).toEqual({});
  });
});
