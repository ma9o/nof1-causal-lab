import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { SimulateScenarioResult } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "./demo-artifacts";

const repoRoot = fileURLToPath(new URL("../../../../../", import.meta.url));

// The current DEMO episode stops at validation. Raw data and extracted measurements
// remain byte-identical; the deterministic presentation projection is generated in
// the same canonical fixture root from that durable source.
const copiedStoreArtifacts = {
  raw_data: ["store/raw_data/v1/profile.json", "fixture/artifacts/raw_data.json"],
  measurements: ["store/measurements/v1/measurements.json", "fixture/artifacts/measurements.json"],
} as const;

describe("promoted DEMO fixture", () => {
  it("keeps materialized artifact projections byte-identical to the store", () => {
    const drifted = Object.entries(copiedStoreArtifacts)
      .filter(([, [storePath, fixturePath]]) => {
        const store = readFileSync(join(repoRoot, "data/DEMO", storePath), "utf8");
        const fixture = readFileSync(join(repoRoot, "data/DEMO", fixturePath), "utf8");
        return store !== fixture;
      })
      .map(([artifactId]) => artifactId);

    expect(drifted, "Run `bun run fixture:promote --from <workspace-id>`.").toEqual([]);
  });

  it("has no second component-local demo fixture root", () => {
    expect(existsSync(join(repoRoot, "apps/web/src/components/__fixtures__/demo-run"))).toBe(false);
  });

  it("keeps the compact scientific DAG and backend structural dispositions aligned", () => {
    const read = (path: string) =>
      JSON.parse(readFileSync(join(repoRoot, "data/DEMO", path), "utf8")) as Record<
        string,
        unknown
      >;
    const fixtureLatent = read("fixture/artifacts/latent_structure.json") as {
      latent_structure: {
        constructs: Array<{ id: string; name: string; role: string }>;
        edges: Array<{ cause_id: string; effect_id: string }>;
      };
    };
    const storedLatent = read("store/latent_structure/v1/latent-structure.json") as {
      latent_structure: {
        edges: Array<{ cause_id: string; effect_id: string }>;
      };
    };
    const fixtureMeasurement = read("fixture/artifacts/measurement_structure.json") as {
      measurement_structure: unknown;
      known_inputs: Array<{ construct_id: string }>;
      scientific_only_constructs: Array<{ construct_id: string }>;
    };
    const fixtureCausal = read("fixture/artifacts/causal_design.json").causal_design as {
      latent: unknown;
      measurement: unknown;
      known_inputs: Array<{ construct_id: string }>;
      scientific_only_constructs: Array<{ construct_id: string }>;
      estimation?: unknown;
    };
    const fixturePlan = read("fixture/artifacts/structural_plan.json").structural_plan as {
      semantics: { constructs: Record<string, { name: string }> };
      dispositions: Array<{
        source_id: string;
        source_kind: string;
        disposition: string;
      }>;
    };

    expect(fixtureCausal.latent).toEqual(fixtureLatent.latent_structure);
    expect(fixtureCausal.measurement).toEqual(fixtureMeasurement.measurement_structure);
    expect(fixtureCausal.known_inputs).toEqual(fixtureMeasurement.known_inputs);
    expect(fixtureCausal.scientific_only_constructs).toEqual(
      fixtureMeasurement.scientific_only_constructs,
    );
    expect(fixtureCausal.known_inputs).toHaveLength(2);
    expect(fixtureCausal.scientific_only_constructs).toHaveLength(6);
    expect(fixtureLatent.latent_structure.constructs).toHaveLength(17);
    expect(fixtureLatent.latent_structure.edges).toHaveLength(32);
    expect(
      (fixtureMeasurement.measurement_structure as { indicators: unknown[] }).indicators,
    ).toHaveLength(19);

    const constructsWithIncomingEdges = new Set(
      fixtureLatent.latent_structure.edges.map(({ effect_id }) => effect_id),
    );
    expect(
      fixtureLatent.latent_structure.constructs
        .filter(({ id, role }) => role === "endogenous" && !constructsWithIncomingEdges.has(id))
        .map(({ name }) => name),
      "The reduced story must not silently turn endogenous constructs into unexplained roots.",
    ).toEqual([]);

    const selectedConstructIds = new Set(
      fixtureLatent.latent_structure.constructs.map(({ id }) => id),
    );
    const labels = new Map(
      fixtureLatent.latent_structure.constructs.map(({ id, name }) => [id, name]),
    );
    const storedInducedEdgeKeys = storedLatent.latent_structure.edges
      .filter(
        ({ cause_id, effect_id }) =>
          selectedConstructIds.has(cause_id) && selectedConstructIds.has(effect_id),
      )
      .map(({ cause_id, effect_id }) => `${labels.get(cause_id)}→${labels.get(effect_id)}`);
    const contractedEdgeKeys = [
      "natural_recovery_propensity→internalizing_symptom_burden",
      "taper_speed_dose_reduction→withdrawal_symptom_burden",
      "escitalopram_dose_taken→internalizing_symptom_burden",
    ];
    expect(
      fixtureLatent.latent_structure.edges
        .map(({ cause_id, effect_id }) => `${labels.get(cause_id)}→${labels.get(effect_id)}`)
        .sort(),
    ).toEqual([...storedInducedEdgeKeys, ...contractedEdgeKeys].sort());

    const dispositions = fixturePlan.dispositions
      .filter(({ source_kind }) => source_kind === "construct")
      .map((item) => ({
        name: fixturePlan.semantics.constructs[item.source_id]?.name,
        disposition: item.disposition,
      }));
    expect(dispositions).toEqual(
      expect.arrayContaining([
        { name: "withdrawal_symptom_burden", disposition: "identification_only" },
        { name: "neuroadaptation_dependence_state", disposition: "identification_only" },
        { name: "natural_recovery_propensity", disposition: "identification_only" },
        {
          name: "past_escitalopram_response_tolerability",
          disposition: "identification_only",
        },
        { name: "clinical_monitoring_rescue_care", disposition: "identification_only" },
        { name: "stable_withdrawal_susceptibility", disposition: "marginalized" },
        { name: "external_stressful_events", disposition: "known_input" },
        { name: "internalizing_symptom_burden", disposition: "retained_state" },
      ]),
    );
    expect(fixtureCausal).not.toHaveProperty("estimation");
  });

  it("keeps every artificial downstream projection on the current DEMO ontology", () => {
    const readFixture = (name: string) =>
      JSON.parse(readFileSync(join(repoRoot, "data/DEMO/fixture", name), "utf8")) as Record<
        string,
        unknown
      >;

    const causal = readFixture("artifacts/causal_design.json").causal_design as {
      latent: {
        constructs: Array<{ id: string; name: string; temporal_status: string }>;
        edges: Array<{ cause_id: string; effect_id: string }>;
      };
      measurement: { indicators: Array<{ name: string; construct_id: string }> };
      identifiability: { identifiable_treatments: Record<string, unknown> };
      known_inputs: Array<{ construct_id: string }>;
      scientific_only_constructs: Array<{ construct_id: string }>;
    };
    const plan = readFixture("artifacts/structural_plan.json").structural_plan as {
      semantics: {
        constructs: Record<string, { name: string }>;
        edges: Record<string, { cause_id: string; effect_id: string }>;
        indicators: Record<string, { name: string }>;
      };
      state_order: string[];
      edges: Array<{ source_id: string }>;
      manifest_indicator_order: string[];
    };
    const model = readFixture("artifacts/statistical_model_spec.json") as {
      statistical_model_spec: {
        likelihoods: Array<{ indicator_id: string }>;
        parameters: Array<{ id: string; name: string; role: string }>;
      };
      authored_priors: Record<string, unknown>;
      resolved_priors: Array<{ parameter_id: string }>;
      prior_predictive_samples: Record<string, number[]>;
    };
    const posterior = readFixture("artifacts/posterior.json") as {
      assessment: {
        ppc: { overlays: Array<{ indicator_id: string }> };
        mcmc_diagnostics: { per_parameter: Array<{ parameter: string }> };
      };
      posterior_marginals: Array<{ parameter: string }>;
    };
    const report = readFixture("artifacts/baseline_report.json") as {
      intervention_results: Array<{ treatment: string }>;
    };

    const stateNames = plan.state_order.map((sourceId) => plan.semantics.constructs[sourceId].name);
    const stateNameSet = new Set(stateNames);
    const indicatorNames = causal.measurement.indicators.map(({ name }) => name);
    const manifestIndicatorNames = plan.manifest_indicator_order.map(
      (sourceId) => plan.semantics.indicators[sourceId].name,
    );
    const executableEdges = plan.edges.map(({ source_id }) => plan.semantics.edges[source_id]);
    const edgeParameterNames = executableEdges.map(
      ({ cause_id, effect_id }) =>
        `beta_${plan.semantics.constructs[cause_id].name}_${plan.semantics.constructs[effect_id].name}`,
    );
    const parameterNames = model.statistical_model_spec.parameters.map(({ name }) => name);
    const parameterIds = model.statistical_model_spec.parameters.map(({ id }) => id);

    expect(Object.keys(model).sort()).toEqual([
      "authored_priors",
      "prior_predictive_diagnostics",
      "prior_predictive_samples",
      "resolved_priors",
      "search_queries",
      "statistical_model_spec",
      "validation_warnings",
    ]);
    expect(Object.keys(posterior).sort()).toEqual([
      "assessment",
      "draws",
      "inference_metadata",
      "posterior_marginals",
      "posterior_pairs",
      "provenance",
    ]);
    expect(Object.keys(report).sort()).toEqual([
      "final_summary",
      "intervention_results",
      "saved_scenarios",
    ]);
    expect(
      model.statistical_model_spec.likelihoods.map(({ indicator_id }) => indicator_id),
    ).toEqual(plan.manifest_indicator_order);
    expect(Object.keys(model.prior_predictive_samples).sort()).toEqual(
      [...plan.manifest_indicator_order].sort(),
    );
    expect(Object.keys(model.authored_priors).sort()).toEqual([...parameterIds].sort());
    expect(model.resolved_priors.map(({ parameter_id }) => parameter_id)).toEqual(parameterIds);
    expect(posterior.assessment.ppc.overlays.map(({ indicator_id }) => indicator_id)).toEqual(
      plan.manifest_indicator_order,
    );
    const coordinates = demoModelSnapshot.fit!.value.posterior.posterior_marginals!.map((marginal) => marginal.parameter).sort();
    expect(
      posterior.assessment.mcmc_diagnostics.per_parameter.map(({ parameter }) => parameter).sort(),
    ).toEqual(coordinates);
    expect(posterior.posterior_marginals.map(({ parameter }) => parameter).sort()).toEqual(
      coordinates,
    );
    expect(
      model.statistical_model_spec.parameters
        .filter(({ role }) => role === "fixed_effect")
        .map(({ name }) => name),
    ).toEqual(edgeParameterNames);
    expect(report.intervention_results.map(({ treatment }) => treatment).sort()).toEqual(
      causal.latent.constructs
        .filter((construct) => construct.id in causal.identifiability.identifiable_treatments)
        .map((construct) => construct.name)
        .filter((treatment) => stateNameSet.has(treatment))
        .sort(),
    );
    expect(stateNames).toHaveLength(7);
    expect(manifestIndicatorNames).toHaveLength(10);
    expect(executableEdges).toHaveLength(13);
    expect(parameterNames).toHaveLength(45);
    expect(indicatorNames).toHaveLength(19);
    expect(stateNames).toContain("internalizing_symptom_burden");
  });

  it("materializes comprehensive DAG layers only where their process semantics exist", () => {
    const plan = JSON.parse(
      readFileSync(join(repoRoot, "data/DEMO/fixture/artifacts/structural_plan.json"), "utf8"),
    ).structural_plan as {
      semantics: {
        constructs: Record<string, { name: string }>;
      };
      state_order: string[];
    };
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
      .map((message) => JSON.parse(message.tool_result)) as SimulateScenarioResult[];

    const stateIds = [...plan.state_order].sort();

    expect(simulations).toHaveLength(5);
    for (const { query, evaluation, result } of simulations) {
      const visualization = result.visualization!;
      const trajectory = result.effect_trajectory!;
      expect(evaluation.query_id).toBe(query.id);
      expect(result.evaluation_id).toBe(evaluation.id);
      expect(result.outcome_label).toBe("internalizing_symptom_burden");
      expect(query.clamps).toHaveLength(1);
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
