import { describe, expect, it } from "vitest";
import { constructs, edges } from "@/components/dag/__fixtures__/dag-base-fixtures";
import { demoModel } from "@/components/__fixtures__/demo-artifacts";
import { buildLayeredCausalGraph } from "@/lib/dag/build-layered-causal-graph";

describe("buildLayeredCausalGraph", () => {
  it("keeps authored edges and persistence in distinct temporal slots", () => {
    const built = buildLayeredCausalGraph(constructs, edges);
    const varying = new Set(
      constructs.filter((item) => item.temporal_status === "time_varying").map((item) => item.id),
    );
    for (const edge of edges) {
      expect(built.edgeMeta.get(edge.id)).toMatchObject({
        source: edge.lagged && varying.has(edge.cause.id) ? `${edge.cause.id}__p` : edge.cause.id,
        target: edge.effect.id,
        isSelf: false,
      });
    }
    const outcome = constructs.find((item) => item.id === demoModel.default_outcome)!;
    expect(built.edgeMeta.get(`self:${outcome.id}`)).toMatchObject({
      source: `${outcome.id}__p`,
      target: outcome.id,
      lagged: true,
      isSelf: true,
    });
  });

  it("preserves topology and graph identity when every display name changes", () => {
    const before = buildLayeredCausalGraph(constructs, edges);
    const after = buildLayeredCausalGraph(
      constructs.map((item, index) => ({ ...item, name: `renamed ${index}` })),
      edges,
    );
    expect(after.graph).toEqual(before.graph);
    expect(after.edgeMeta).toEqual(before.edgeMeta);
    expect(after.segmentMeta).toEqual(before.segmentMeta);
  });
});
