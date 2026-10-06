import { describe, expect, it } from "vitest";
import { baseline, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { buildLayeredCausalGraph } from "@/lib/dag/build-layered-causal-graph";

const constructs = [baseline, treatment, outcome];
const edges = modelFixture.edges;
const dynamicConstructIds = [treatment.id, outcome.id];

describe("buildLayeredCausalGraph", () => {
  it("keeps authored edges and persistence in distinct temporal slots", () => {
    const built = buildLayeredCausalGraph(constructs, edges, dynamicConstructIds);
    const varying = new Set(dynamicConstructIds);
    for (const id of varying) expect(built.edgeMeta.has(`self:${id}`)).toBe(true);
    for (const edge of edges) {
      expect(built.edgeMeta.get(edge.id)).toMatchObject({
        source: varying.has(edge.cause.id) ? `${edge.cause.id}__p` : edge.cause.id,
        target: edge.effect.id,
        isSelf: false,
      });
    }
    expect(built.edgeMeta.get(`self:${outcome.id}`)).toMatchObject({
      source: `${outcome.id}__p`,
      target: outcome.id,
      crossSlice: true,
      isSelf: true,
    });
  });

  it("preserves topology and graph identity when every display name changes", () => {
    const before = buildLayeredCausalGraph(constructs, edges, dynamicConstructIds);
    const after = buildLayeredCausalGraph(
      constructs.map((item, index) => ({ ...item, name: `renamed ${index}` })),
      edges,
      dynamicConstructIds,
    );
    expect(after.graph).toEqual(before.graph);
    const identities = (bundle: typeof before) =>
      [...bundle.edgeMeta.values()].map(({ cause, effect, ...edge }) => ({
        ...edge,
        cause: cause.id,
        effect: effect.id,
      }));
    expect(identities(after)).toEqual(identities(before));
    expect(after.segmentMeta).toEqual(before.segmentMeta);
  });
});
