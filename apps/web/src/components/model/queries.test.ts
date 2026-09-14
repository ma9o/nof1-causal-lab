import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { demoSimulationTrace } from "@/components/dag/__fixtures__/simulation-fixture";
import { buildSimulationScenarios } from "@/components/dag/simulation-results";
import { buildModelQueries } from "./queries";

describe("runtime queries", () => {
  it("offers identified intervention requests without manufacturing results", () => {
    const queries = buildModelQueries(demoModelSnapshot);
    expect(queries.map((query) => query.treatmentId)).toEqual(
      Object.entries(demoModelSnapshot.findings.identification!.value.treatments)
        .filter(([, finding]) => finding.status === "identified")
        .map(([id]) => id),
    );
    expect(queries.length).toBeGreaterThan(0);
    for (const query of queries) {
      expect(query.simulation).toBeNull();
      expect(query.posterior).toBeNull();
      expect(query.request.clamps[0].target).toBe(query.treatmentId);
      expect(query.request.outcome).toBe(demoModelSnapshot.findings.identification!.value.outcome);
    }
    const unfitted = structuredClone(demoModelSnapshot);
    unfitted.findings.fit = null;
    expect(buildModelQueries(unfitted)).toEqual([]);
    const unidentified = structuredClone(demoModelSnapshot);
    const treatmentId = queries[0].treatmentId;
    unidentified.findings.identification!.value.treatments[treatmentId] = {
      status: "not_identified",
      confounders: [],
      notes: "No identifying estimand",
    };
    expect(buildModelQueries(unidentified).map((query) => query.treatmentId)).not.toContain(
      treatmentId,
    );
  });

  it("uses session responses with their original fit without adding them to the model", () => {
    const model = structuredClone(demoModelSnapshot);
    const before = structuredClone(model);
    const query = buildModelQueries(model)[0];
    const result = structuredClone(
      buildSimulationScenarios({ trace: demoSimulationTrace })[0].result,
    );
    result.request = query.request;
    result.model.version = 42;
    const entry = buildModelQueries(model, { [query.key]: result })[0];
    expect(entry.simulation).toEqual(result);
    expect(entry.modelVersion).toBe(42);
    expect(entry.posterior).toEqual(result.summary);
    expect(model).toEqual(before);
    expect(buildModelQueries(model)[0].simulation).toBeNull();
    result.model.workspace_id = "OTHER";
    expect(buildModelQueries(model, { [query.key]: result })[0].simulation).toBeNull();
  });
});
