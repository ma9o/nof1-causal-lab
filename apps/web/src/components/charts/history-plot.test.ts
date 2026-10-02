import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HistoryPlot } from "./history-plot";

describe("recorded history geometry", () => {
  it("places observations by elapsed time rather than row order", () => {
    const html = renderToStaticMarkup(
      createElement(HistoryPlot, {
        times: [0, 1, 10],
        series: [{ id: "observed", label: "Observed", values: [1, 2, 3] }],
        label: "Irregular observations",
        pointsOnly: true,
      }),
    );
    const x = [...html.matchAll(/<circle cx="([\d.]+)"/g)].map((match) => Number(match[1]));
    expect(x).toHaveLength(3);
    expect(
      (fixtureValue(x[1]) - fixtureValue(x[0])) / (fixtureValue(x[2]) - fixtureValue(x[0])),
    ).toBeCloseTo(0.1);
  });

  it("breaks trajectories at missing values without fabricating a bridge", () => {
    const html = renderToStaticMarkup(
      createElement(HistoryPlot, {
        times: [0, 1, 2, 10, 11],
        series: [{ id: "draw", label: "Draw", values: [1, 2, null, 4, 3] }],
        label: "Missing interval",
      }),
    );
    const path = html.match(/<path d="([^"]+)"/)?.[1];
    expect(path?.match(/M/g)).toHaveLength(2);
    expect(path?.match(/L/g)).toHaveLength(2);
    expect(path).not.toMatch(/[CQ]/);
  });
});
