import { modelConstructs } from "@/lib/model-accessors";
import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { referencedParameterIds } from "./model-accessors";
import { collectModelSpecObservationPriorTerms } from "./model-spec-data";

describe("scientific prior references", () => {
  it("follows coefficient references independently of display names", () => {
    const model = demoModelSnapshot.model!.value;
    const indicator = modelConstructs(model)
      .flatMap((c) => c.indicators)
      .find((i) => referencedParameterIds(i.likelihood).size > 0)!;
    const id = [...referencedParameterIds(indicator.likelihood)][0];
    const parameter = {
      ...model.parameters.find((p) => p.id === id)!,
      name: "arbitrary parameter label",
    };
    expect(
      collectModelSpecObservationPriorTerms({
        indicator: { ...indicator, name: "renamed indicator" },
        parameters: [parameter],
      }),
    ).toEqual([parameter]);
    expect(
      collectModelSpecObservationPriorTerms({
        indicator: { ...indicator, likelihood: null },
        parameters: [parameter],
      }),
    ).toEqual([]);
  });
});
