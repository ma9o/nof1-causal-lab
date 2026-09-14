import { describe, expect, it } from "vitest";
import { demoBaselineTrace } from "../__fixtures__/baseline_report-materialized-fixture";
import { buildBaselineReportScenarios } from "../../pipeline/output-views/baseline-report-scenarios";
import { buildSimulateInput } from "./simulate-input";

const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });

describe("buildSimulateInput", () => {
  it("round-trips an abducted result with exactly one backend-valid start selector", () => {
    const result = scenarios.find(
      (scenario) => scenario.result.request.start.kind === "abducted",
    )?.result;
    if (!result) throw new Error("DEMO fixture must include an abducted simulation result");

    const input = buildSimulateInput(
      result,
      [{ target: result.request.clamps[0].target, mode: "set", value: 0.5, from_day: 10 }],
      60,
    );

    expect(input.start).toEqual(result.request.start);
    expect(input.start.time).toBeNull();
  });

  it("does not copy result-only start metadata into a baseline input", () => {
    const result = scenarios.find(
      (scenario) => scenario.result.request.start.kind === "baseline",
    )?.result;
    if (!result) throw new Error("DEMO fixture must include a baseline simulation result");

    const input = buildSimulateInput(
      result,
      [{ target: result.request.clamps[0].target, mode: "set", value: 0.5, from_day: 0 }],
      60,
    );

    expect(input.start).toEqual(result.request.start);
    expect("state_source" in input.start).toBe(false);
  });

  it("preserves a latest-observation start rule across evaluations with different data", () => {
    const result = structuredClone(scenarios[0].result);
    result.request.start = { kind: "abducted" };
    result.provenance.start_time_index = 99;
    result.provenance.start_time = "2026-01-01";
    const input = buildSimulateInput(
      result,
      [{ target: result.request.clamps[0].target, mode: "set", value: 0.5, from_day: 0 }],
      60,
    );
    expect(input.start).toEqual({ kind: "abducted" });
  });
});
