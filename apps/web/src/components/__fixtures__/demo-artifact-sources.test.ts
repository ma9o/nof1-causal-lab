import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { presentEntries } from "@/lib/model-accessors";
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

  it("keeps retained numerical findings on scientific IDs without compiler coordinates", () => {
    expect(demoModelSnapshot.findings).not.toHaveProperty("execution");
    const posterior = fixtureValue(demoModelSnapshot.findings.fit).value.report;
    const parameters = new Set(demoModel.parameters.map((p) => p.id));
    expect(
      fixtureValue(posterior.posterior_marginals).every((m) =>
        parameters.has(m.subject.parameter_id),
      ),
    ).toBe(true);
    expect(posterior.posterior_marginals).toHaveLength(92);
    expect(posterior.engine.kind).toBe("not_evaluated");
    expect(posterior.inference_diagnostics).toEqual(demoPosterior.inference_diagnostics);
    const indicators = new Set(
      modelConstructs(demoModel).flatMap((construct) =>
        construct.indicators.map((i) => i.observation.id),
      ),
    );
    expect(predictiveChecks.overlays.every((o) => indicators.has(o.indicator_id))).toBe(true);
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
      const trajectories = result.predictive.states;
      const trajectory = fixtureValue(result.causal_result.effect_trajectory);
      expect(
        result.causal_result.warnings.some((warning) =>
          warning.includes("Artificial Storybook simulation"),
        ),
      ).toBe(true);
      expect(result.model.revision).toBe("a".repeat(40));
      expect(result.causal_result.labels[result.causal_result.outcome]).toBe(
        "internalizing_symptom_burden",
      );
      expect(result.design.interventions).toHaveLength(1);
      expect(Object.keys(trajectories).sort()).toEqual(stateIds);
      expect(trajectory).toHaveLength(61);
      for (const series of presentEntries(trajectories).map(([, series]) => series)) {
        expect(series.reference?.kind).toBe("numeric");
        expect(series.action.kind).toBe("numeric");
        if (series.reference?.kind === "numeric" && series.action.kind === "numeric") {
          expect(series.reference.mean).toHaveLength(result.times.length);
          expect(series.action.mean).toHaveLength(result.times.length);
        }
      }
    }
  });
});
