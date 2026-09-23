import { describe, expect, it } from "vitest";
import {
  formatScenarioActionDescription,
  getSimulationDays,
  getNodeActionSeries,
  getNodeReferenceSeries,
} from "@/lib/dag/simulation";
import type { AnalysisSimulationResult } from "@/lib/dag/simulation-types";

describe("intervention DAG semantics", () => {
  it("plots end-state responses on their simulation grid and formats clamp labels", () => {
    const result = {
      request: {
        readout: { estimand: "end_state" },
        clamps: [
          {
            target: "construct:lipid",
            mode: "shift",
            amount: 1,
            from_day: 0,
          },
        ],
      },
      labels: { "construct:lipid": "lipid_burden" },
      time_grid_days: [0, 0.5, 2],
      effect_trajectory: null,
      trajectories: {
        "construct:lipid": {
          reference_mean: [0.85, 0.9, 0.95],
          action_mean: [1.85, 1.8, 1.75],
        },
      },
    } as unknown as AnalysisSimulationResult;

    expect(formatScenarioActionDescription(result)).toBe("do(lipid_burden shift +1.0)");
    expect(getSimulationDays(result)).toEqual([0, 0.5, 2]);
    expect(getNodeReferenceSeries(result, "construct:lipid")).toEqual([0.85, 0.9, 0.95]);
    expect(getNodeActionSeries(result, "construct:lipid")).toEqual([1.85, 1.8, 1.75]);
    expect(getNodeReferenceSeries(result, "construct:unprojected")).toBeNull();
    expect(getNodeActionSeries(result, "construct:unprojected")).toBeNull();
  });
});
