import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { parameters } from "./__fixtures__/statistical-model-spec-fixtures";

import { ObsPriorList } from "./obs-model-table";

describe("ObsPriorList", () => {
  it("marks missing authored observation priors as not authored", () => {
    const markup = renderToStaticMarkup(createElement(ObsPriorList, { terms: [] }));

    expect(markup).toContain("Not authored");
  });

  it("marks missing expected observation terms as not authored", () => {
    const markup = renderToStaticMarkup(
      createElement(ObsPriorList, {
        terms: [{ ...parameters[0], name: "obs_sd_sleep", distribution: null }],
      }),
    );

    expect(markup).toContain("Not authored");
    expect(markup).toContain("\\theta_{\\text{obs sd sleep}}");
  });

  it("renders the scientific parameter label used in the conditional equation", () => {
    const markup = renderToStaticMarkup(
      createElement(ObsPriorList, {
        terms: [
          {
            ...parameters[0],
            name: "obs_concentration",
            distribution: {
              distribution: "Gamma",
              params: { concentration: 5, rate: 0.5 },
            },
          },
        ],
      }),
    );

    expect(markup).toContain("\\theta_{\\text{obs concentration}} \\sim \\text{Gamma}(5,\\; 0.5)");
  });
});
