import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  equations,
  parameters,
  structuralPlan,
} from "./__fixtures__/statistical-model-spec-fixtures";
import { SSMEquationDisplay } from "./ssm-equation-display";

describe("SSMEquationDisplay", () => {
  it("renders backend equations as continuous-time dynamics", () => {
    const markup = renderToStaticMarkup(
      createElement(SSMEquationDisplay, {
        likelihoods: [],
        equations,
        structuralPlan,
        parameters,
        priors: [],
      }),
    );

    expect(markup).toContain("Continuous-time dynamics");
    expect(markup).toContain("Brownian motion");
    expect(markup).not.toContain("AR(1)");
  });
});
