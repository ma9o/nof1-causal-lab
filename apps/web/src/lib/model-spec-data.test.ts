import {
  demoModelSnapshot,
  demoStatisticalModelSpec,
} from "@/components/__fixtures__/demo-artifacts";
import { describe, expect, it } from "vitest";
import { collectModelSpecObservationPriorTerms, collectModelSpecUiPriors } from "./model-spec-data";

describe("scientific prior references", () => {
  it("selects authored priors by parameter ID through renames", () => {
    const data = structuredClone(demoStatisticalModelSpec);
    const definition = data.statistical_model_spec.parameters[0];
    const prior = data.authored_priors[definition.id];
    data.statistical_model_spec.parameters = [{ ...definition, name: "new label" }];
    expect(collectModelSpecUiPriors(data)).toEqual([prior]);
  });

  it("uses explicit owners and distinguishes reuse of an indicator name", () => {
    const base = demoModelSnapshot.compiled_parameters!.value.find((p) =>
      p.owners.some((o) => o.kind === "indicator"),
    )!;
    const owner = base.owners.find((o) => o.kind === "indicator")!;
    const parameter = { ...base, name: "arbitrary parameter label" };
    const likelihood = {
      indicator_id: owner.id as `indicator:${string}`,
      distribution: "gaussian" as const,
      link: "identity" as const,
      reasoning: "",
      standardized: false,
      sources: [],
    };
    const args = { likelihood, priors: [], parameters: [parameter] };
    expect(collectModelSpecObservationPriorTerms(args)).toEqual([
      { parameterName: parameter.name, prior: undefined },
    ]);
    expect(
      collectModelSpecObservationPriorTerms({
        ...args,
        likelihood: { ...likelihood, indicator_id: "indicator:another" },
      }),
    ).toEqual([]);
  });
});
