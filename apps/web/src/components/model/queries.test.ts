import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { buildModelQueries } from "./queries";

describe("saved model queries", () => {
  it("pairs the scientific query with its pinned evaluation and outputs", () => {
    const saved = demoModelSnapshot.saved_scenarios!.value.scenarios[0];
    const item = saved.evaluations[0];
    const entry = buildModelQueries(demoModelSnapshot).find(
      (query) => query.key === saved.query.id,
    )!;
    expect(entry.simulation).toEqual({ query: saved.query, ...item });
    expect(entry.simulation?.evaluation.query_id).toBe(saved.query.id);
    expect(entry.simulation?.result.evaluation_id).toBe(item.evaluation.id);
    expect(entry.posteriorVersion).toBe(item.evaluation.posterior.version);
    expect(entry.posterior).toEqual(item.result.summary);
  });

  it("keeps all evaluations under one question and selects the current fit", () => {
    const model = structuredClone(demoModelSnapshot);
    const saved = model.saved_scenarios!.value.scenarios[0];
    const historical = structuredClone(saved.evaluations[0]);
    historical.evaluation.id = `evaluation:${"a".repeat(64)}`;
    historical.result.evaluation_id = historical.evaluation.id;
    historical.evaluation.model.id = "ALTERNATIVE";
    historical.result.summary.mean = 0.9;
    saved.evaluations.push(historical);
    const entries = buildModelQueries(model).filter((query) => query.key === saved.query.id);
    expect(entries).toHaveLength(1);
    expect(entries[0].evaluations).toHaveLength(2);
    expect(entries[0].simulation?.evaluation).toEqual(saved.evaluations[0].evaluation);
    expect(entries[0].evaluations[1].result.summary.mean).toBe(0.9);
  });

  it("keeps an unevaluated saved query selectable without inventing results", () => {
    const model = structuredClone(demoModelSnapshot);
    const saved = model.saved_scenarios!.value.scenarios[0];
    saved.evaluations = [];
    const entry = buildModelQueries(model).find((query) => query.key === saved.query.id)!;
    expect(entry.savedQuery).toEqual(saved.query);
    expect(entry.simulation).toBeNull();
    expect(entry.posterior).toBeNull();
    expect(entry.horizonDays).toBe(saved.query.readout.horizon_days);
  });
});
