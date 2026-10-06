import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { describe, expect, it } from "vitest";
import { modelFixture } from "@/lib/__fixtures__/model";
import { modelConstructs } from "@/lib/model-accessors";
import { indexModel, resolveEntity } from "./entities";
import type { EntitySelection } from "./selection";

/** Put shared definitions at the other end of the same serialized graph. */
function reversedGraph() {
  const constructs = new Map(modelConstructs(modelFixture).map((item) => [item.id, item]));
  const seen = new Set<string>();
  const endpoint = (id: (typeof modelFixture.edges)[number]["cause"]["id"]) => {
    if (seen.has(id)) return { kind: "construct" as const, id };
    seen.add(id);
    return fixtureValue(constructs.get(id));
  };
  return {
    ...modelFixture,
    edges: [...modelFixture.edges].reverse().map((edge) => ({
      ...edge,
      cause: endpoint(edge.cause.id),
      effect: endpoint(edge.effect.id),
    })),
  };
}

describe("scoped model inspection", () => {
  it("reads the same entities when shared endpoint definitions move", () => {
    const reversed = reversedGraph();
    const constructs = modelConstructs(modelFixture);
    const selections: EntitySelection[] = [
      ...modelFixture.edges.map((item) => ({ kind: "edge" as const, id: item.id })),
      ...constructs.map((item) => ({ kind: "construct" as const, id: item.id })),
      ...constructs.flatMap((item) =>
        item.indicators.map((indicator) => ({
          kind: "indicator" as const,
          id: indicator.observation.id,
        })),
      ),
    ];
    for (const selection of selections) {
      expect(resolveEntity(indexModel(reversed), selection)).toEqual(
        resolveEntity(indexModel(modelFixture), selection),
      );
    }
  });
});
