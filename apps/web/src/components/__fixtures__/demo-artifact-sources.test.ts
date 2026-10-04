import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { modelConstructs } from "@/lib/model-accessors";
import { hasCausalEffects } from "@/lib/simulation-report";
import simulationReports from "@/components/dag/__fixtures__/simulation-reports.json";
import { ownLawUses } from "@/lib/model-asset/laws";
import { demoModel, demoModelSnapshot, demoPosterior } from "./demo-artifacts";
import { predictiveChecks } from "./inference-data";

const repoRoot = fileURLToPath(new URL("../../../../../", import.meta.url));

describe("promoted DEMO fixture", () => {
  it("has no second component-local demo fixture root", () => {
    expect(existsSync(join(repoRoot, "apps/web/src/components/__fixtures__/demo-run"))).toBe(false);
  });

  it("keeps one scientific catalog with all references resolved", () => {
    const { edges, parameters } = demoModel;
    const constructs = modelConstructs(demoModel);
    const indicators = constructs.flatMap((c) => c.indicators);
    const mechanisms = [
      ...constructs.flatMap((c) => c.dynamics),
      ...edges.flatMap((e) => e.mechanisms),
    ];
    const ids = new Set(
      [...constructs, ...edges, ...indicators, ...mechanisms].map((e) =>
        "observation" in e ? e.observation.id : e.id,
      ),
    );
    expect(constructs).toHaveLength(17);
    expect(edges).toHaveLength(32);
    expect(indicators).toHaveLength(19);
    expect(parameters).toHaveLength(45);
    for (const edge of edges) {
      expect(ids.has(edge.cause.id)).toBe(true);
      expect(ids.has(edge.effect.id)).toBe(true);
    }
    const parameterIds = new Set(parameters.map((parameter) => parameter.id));
    for (const { parameterId: id } of [...constructs, ...edges].flatMap(ownLawUses))
      expect(parameterIds.has(id)).toBe(true);
    for (const parameter of parameters) {
      expect("owners" in parameter).toBe(false);
      expect("quantity" in parameter).toBe(false);
    }
    expect(indicators.every((i) => !("construct_id" in i))).toBe(true);
  });

  it("shows absent posterior evidence when the archive retained no atoms", () => {
    expect(demoModelSnapshot).not.toHaveProperty("execution");
    expect(demoModelSnapshot.fit).toBeNull();
    expect(demoPosterior).toBeNull();
    expect(predictiveChecks).toBeNull();
  });

  it("materializes comprehensive DAG layers only where their process semantics exist", () => {
    const simulations = simulationReports.filter(hasCausalEffects);

    // Retained illustrative traces cover the constructs with authored dynamics.
    const stateIds = modelConstructs(demoModel)
      .filter((construct) => construct.dynamics.length > 0)
      .map((construct) => construct.id)
      .sort();

    expect(simulations).toHaveLength(5);
    for (const result of simulations) {
      expect(
        result.causal.value.warnings.some((warning) =>
          warning.includes("Artificial Storybook simulation"),
        ),
      ).toBe(true);
      expect(result.evidence.model.revision).toBe("a".repeat(40));
      expect(result.causal.value.labels[result.causal.value.outcome]).toBe(
        "internalizing_symptom_burden",
      );
      expect(result.evidence.design.interventions).toHaveLength(1);
      expect([...result.evidence.state_ids].sort()).toEqual(stateIds);
      expect(result.evidence.times).toHaveLength(61);
      expect(result.evidence.latent_paths).toBeTruthy();
      expect(result.evidence.reference_latent_paths).toBeTruthy();
      expect(result).not.toHaveProperty("predictive");
      expect(result.causal.value).not.toHaveProperty("effect_trajectory");
    }
  });
});
