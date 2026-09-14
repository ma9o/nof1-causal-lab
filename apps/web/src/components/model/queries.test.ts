import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { buildModelQueries } from "./queries";

describe("report queries", () => {
  it("retains rerunnable requests with their original responses and posterior versions", () => {
    const model = structuredClone(demoModelSnapshot);
    const result = model.findings.baseline_report!.value.simulation_results[0];
    result.provenance.model.version = 42;
    const entry = buildModelQueries(model).find((query) => query.key === "report:0")!;
    expect(entry.simulation).toEqual(result);
    expect(entry.request).toEqual(result.request);
    expect(entry.modelVersion).toBe(42);
    expect(entry.posterior).toEqual(result.summary);
  });
});
