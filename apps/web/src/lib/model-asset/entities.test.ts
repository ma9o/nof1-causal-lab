import { describe, expect, it } from "vitest";
import { demoModel } from "@/components/__fixtures__/demo-artifacts";
import { modelConstructs } from "@/lib/model-accessors";
import { indexModel, resolveEntity } from "./entities";
import type { EntitySelection } from "./selection";

/** Put shared definitions at the other end of the same serialized graph. */
function reversedGraph() {
  const constructs = new Map(modelConstructs(demoModel).map((item) => [item.id, item]));
  const seen = new Set<string>();
  const endpoint = (id: (typeof demoModel.edges)[number]["cause"]["id"]) => {
    if (seen.has(id)) return { kind: "construct" as const, id };
    seen.add(id);
    return constructs.get(id)!;
  };
  return {
    ...demoModel,
    edges: [...demoModel.edges].reverse().map((edge) => ({
      ...edge,
      cause: endpoint(edge.cause.id),
      effect: endpoint(edge.effect.id),
    })),
  };
}

describe("scoped model inspection", () => {
  it("reads the same entities when shared endpoint definitions move", () => {
    const reversed = reversedGraph();
    const constructs = modelConstructs(demoModel);
    const selections: EntitySelection[] = [
      ...demoModel.edges.map((item) => ({ kind: "edge" as const, id: item.id })),
      ...constructs.map((item) => ({ kind: "construct" as const, id: item.id })),
      ...constructs.flatMap((item) =>
        item.indicators.map((indicator) => ({ kind: "indicator" as const, id: indicator.id })),
      ),
    ];
    for (const selection of selections) {
      expect(resolveEntity(indexModel(reversed), selection)).toEqual(
        resolveEntity(indexModel(demoModel), selection),
      );
    }
  });
});
