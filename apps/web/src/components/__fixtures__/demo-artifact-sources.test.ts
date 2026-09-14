import { modelConstructs } from "@/lib/model-accessors";
import { referencedParameterIds } from "@/lib/model-accessors";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { SimulationResult } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { demoModelSnapshot, demoModel, demoPosterior } from "./demo-artifacts";

const repoRoot = fileURLToPath(new URL("../../../../../", import.meta.url));

describe("promoted DEMO fixture", () => {
  it("has no second component-local demo fixture root", () => {
    expect(existsSync(join(repoRoot, "apps/web/src/components/__fixtures__/demo-run"))).toBe(false);
  });

  it("keeps one scientific catalog with all references resolved", () => {
    const { edges, parameters } = demoModel;
    const constructs = modelConstructs(demoModel);
    const indicators = constructs.flatMap((c) => c.indicators ?? []);
    const mechanisms = [
      ...constructs.flatMap((c) => c.dynamics ?? []),
      ...edges.flatMap((e) => e.mechanisms ?? []),
    ];
    const ids = new Set([...constructs, ...edges, ...indicators, ...mechanisms].map((e) => e.id));
    expect(constructs).toHaveLength(17);
    expect(edges).toHaveLength(32);
    expect(indicators).toHaveLength(19);
    expect(parameters).toHaveLength(45);
    for (const edge of edges) {
      expect(ids.has(edge.cause.id)).toBe(true);
      expect(ids.has(edge.effect.id)).toBe(true);
    }
    const parameterIds = new Set(parameters.map((parameter) => parameter.id));
    for (const id of referencedParameterIds([constructs, edges]))
      expect(parameterIds.has(id as (typeof parameters)[number]["id"])).toBe(true);
    for (const parameter of parameters) {
      expect("owners" in parameter).toBe(false);
      expect("quantity" in parameter).toBe(false);
    }
    expect(indicators.every((i) => !("construct_id" in i))).toBe(true);
  });

  it("keeps retained numerical findings on scientific IDs without compiler coordinates", () => {
    const execution = demoModelSnapshot.findings.execution!;
    expect(Object.keys(execution.value).sort()).toEqual([
      "anchor_certificates",
      "unmet_requirements",
    ]);
    expect(execution.value.unmet_requirements).toEqual([]);
    expect(execution.source).toEqual(demoModelSnapshot.model!.source);
    const posterior = demoModelSnapshot.findings.fit!.value.report;
    const retained = JSON.parse(
      readFileSync(join(repoRoot, "data/DEMO/fixture/inference.json"), "utf8"),
    );
    const coordinates = Object.keys(retained.retained_draw_axes.parameter_shapes).sort();
    const parameters = new Set(demoModel.parameters.map((p) => p.id));
    expect(
      posterior.posterior_marginals!.every((m) => parameters.has(m.subject.parameter_id)),
    ).toBe(true);
    expect(posterior.posterior_marginals!.map((m) => m.subject.element_id).sort()).toEqual(
      coordinates,
    );
    expect(posterior.inference_diagnostics).toEqual(demoPosterior.inference_diagnostics);
    const observed = demoModelSnapshot.findings.admission_report!.value.prior_predictive_samples!;
    expect(posterior.assessment.ppc!.overlays.map((o) => o.indicator_id).sort()).toEqual(
      Object.keys(observed).sort(),
    );
  });

  it("materializes comprehensive DAG layers only where their process semantics exist", () => {
    const trace = JSON.parse(
      readFileSync(join(repoRoot, "data/DEMO/fixture/traces/baseline_report.json"), "utf8"),
    ) as {
      messages: Array<{ tool_name: string | null; tool_result: string | null }>;
    };
    const simulations = trace.messages
      .filter(
        (message): message is { tool_name: string; tool_result: string } =>
          message.tool_name === "simulate" && message.tool_result != null,
      )
      .map((message) => JSON.parse(message.tool_result)) as SimulationResult[];

    const stateIds = demoModelSnapshot.findings
      .dispositions!.value.filter((item) => item.disposition === "retained_state")
      .map((item) => item.source_id)
      .sort();

    expect(simulations).toHaveLength(5);
    for (const result of simulations) {
      const visualization = result.visualization!;
      const trajectory = result.effect_trajectory!;
      expect(result.provenance.model.version).toBe(3);
      expect(result.labels[result.request.outcome.id]).toBe("internalizing_symptom_burden");
      expect(result.request.clamps).toHaveLength(1);
      expect(Object.keys(visualization).sort()).toEqual([
        "action_node_trajectories",
        "node_effect_trajectories",
        "reference_node_trajectories",
        "start_state",
      ]);
      expect(Object.keys(visualization.reference_node_trajectories!).sort()).toEqual(stateIds);
      expect(Object.keys(visualization.action_node_trajectories!).sort()).toEqual(stateIds);
      expect(Object.keys(visualization.node_effect_trajectories!).sort()).toEqual(stateIds);
      expect(Object.keys(visualization.start_state!).sort()).toEqual(stateIds);
      expect(trajectory).toHaveLength(61);
      expect(
        Object.values(visualization.reference_node_trajectories!).every(
          (series) => series.length === trajectory.length,
        ),
      ).toBe(true);
    }
  });
});
