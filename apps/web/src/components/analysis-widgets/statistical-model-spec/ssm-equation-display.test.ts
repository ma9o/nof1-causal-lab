import type { Indicator } from "@nof1-causal-lab/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  observationEquations,
  equations,
  indicators,
  constructs,
  parameters,
  model,
  confounderEquations,
} from "./__fixtures__/statistical-model-spec-fixtures";
import { SSMEquationDisplay } from "./ssm-equation-display";

// Supply a visible viewport for table rows during server rendering.
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 48,
    getVirtualItems: () =>
      Array.from({ length: count }, (_, index) => ({
        index,
        key: index,
        start: index * 48,
        end: (index + 1) * 48,
        size: 48,
      })),
    measureElement: () => {},
  }),
}));

describe("SSMEquationDisplay", () => {
  it("displays the observation equation supplied by the backend", () => {
    const source = indicators[0];
    const owner = constructs.find((construct) =>
      construct.indicators.some((indicator) => indicator.id === source.id),
    )!;
    const indicator: Indicator = {
      ...source,
      likelihood: {
        law: {
          distribution: "Normal",
          arguments: {
            loc: {
              kind: "binary",
              operator: "add",
              left: {
                kind: "coefficient",
                role: "observation_intercept",
                coefficient: { kind: "fixed", value: 0 },
              },
              right: {
                kind: "binary",
                operator: "multiply",
                left: {
                  kind: "coefficient",
                  role: "loading",
                  coefficient: { kind: "fixed", value: 1 },
                },
                right: { kind: "state", construct_id: owner.id },
              },
            },
            scale: {
              kind: "coefficient",
              role: "observation_scale",
              coefficient: { kind: "fixed", value: 0.6 },
            },
          },
        },
        standardized: false,
        reasoning: "Test conditional law",
        sources: [],
      },
    };
    const supplied = String.raw`y(t) \sim \operatorname{Normal}(\theta_{\text{authored baseline}} + 17.25, 0.6)`;
    const markup = renderToStaticMarkup(
      createElement(SSMEquationDisplay, {
        indicators: [indicator],
        observationEquations: { [indicator.id]: supplied },
        equations: [],
        model,
        confounderEquations: [],
        parameters,
      }),
    );
    expect(markup).toContain("authored baseline");
    expect(markup).toContain("17.25");
  });

  it("renders backend equations as continuous-time dynamics", () => {
    const markup = renderToStaticMarkup(
      createElement(SSMEquationDisplay, {
        indicators: [],
        observationEquations,
        equations,
        model,
        confounderEquations,
        parameters,
      }),
    );

    expect(markup).toContain("Continuous-time dynamics");
    expect(markup).toContain("Brownian motion");
    expect(markup).not.toContain("AR(1)");
  });
});
