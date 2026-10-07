import type { SimulationPaths } from "@nof1-causal-lab/api-types";
import { expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import visuals from "@/components/__fixtures__/workbench-visuals.json";
import { pathsView } from "./result-values";

it("renders selected saved replicates without exposing masked or non-finite observations", () => {
  const observations = "a".repeat(64);
  const mask = "b".repeat(64);
  const identity = "indicator:recorded";
  const saved: SimulationPaths = {
    ...visuals.simulation,
    states: {},
    effect: null,
    indicators: {
      [identity]: {
        label: "Recorded observations",
        levels: null,
        frame: [5, 9],
        action: [
          {
            draw: 1,
            values: {
              array_ref: observations,
              indices: [1, null, 1],
              start: 1,
              stop: 4,
              mask: { array_ref: mask, indices: [1, null, 1], start: 1, stop: 4, mask: null },
            },
          },
        ],
        reference: [],
      },
    },
  };
  const displayed = pathsView(saved, {
    [observations]: {
      dtype: "float64",
      shape: [2, 4, 2],
      values: [90, 91, 92, 93, 94, 95, 96, 97, 0, 1, 2, 5, 3, 7, 4, "+inf"],
    },
    [mask]: {
      dtype: "bool",
      shape: [2, 4, 2],
      values: [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
      ],
    },
  });
  const series = fixtureValue(displayed.indicators[identity]);
  expect(series.action).toEqual([{ draw: 1, values: [5, null, null] }]);
  expect(series.frame).toEqual([5, 9]);
  expect(saved.indicators[identity]?.action[0]?.values).toHaveProperty("array_ref", observations);
});
