import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { describe, expect, it } from "vitest";
import { collectModelSpecObservationPriorTerms } from "./model-spec-data";

describe("scientific prior references", () => {
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
    const args = { likelihood, parameters: [parameter] };
    expect(collectModelSpecObservationPriorTerms(args)).toEqual([parameter]);
    expect(
      collectModelSpecObservationPriorTerms({
        ...args,
        likelihood: { ...likelihood, indicator_id: "indicator:another" },
      }),
    ).toEqual([]);
  });
});
