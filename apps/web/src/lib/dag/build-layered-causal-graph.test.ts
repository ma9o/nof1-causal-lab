import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { describe, expect, it } from "vitest";
import { constructs, edges } from "@/components/dag/__fixtures__/dag-base-fixtures";
import { demoModel, demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { buildLayeredCausalGraph } from "@/lib/dag/build-layered-causal-graph";

describe("buildLayeredCausalGraph", () => {
  it("keeps authored edges and persistence in distinct temporal slots", () => {
    const built = buildLayeredCausalGraph(
      constructs,
      edges,
      demoModelSnapshot.findings.graph.dynamic_construct_ids,
    );
    const varying = new Set(demoModelSnapshot.findings.graph.dynamic_construct_ids);
    for (const id of varying) expect(built.edgeMeta.has(`self:${id}`)).toBe(true);
    for (const edge of edges) {
      expect(built.edgeMeta.get(edge.id)).toMatchObject({
        source: varying.has(edge.cause.id) ? `${edge.cause.id}__p` : edge.cause.id,
        target: edge.effect.id,
        isSelf: false,
      });
    }
    const outcome = fixtureValue(constructs.find((item) => item.id === demoModel.default_outcome));
    expect(built.edgeMeta.get(`self:${outcome.id}`)).toMatchObject({
      source: `${outcome.id}__p`,
      target: outcome.id,
      crossSlice: true,
      isSelf: true,
    });
  });

  it("preserves topology and graph identity when every display name changes", () => {
    const before = buildLayeredCausalGraph(
      constructs,
      edges,
      demoModelSnapshot.findings.graph.dynamic_construct_ids,
    );
    const after = buildLayeredCausalGraph(
      constructs.map((item, index) => ({ ...item, name: `renamed ${index}` })),
      edges,
      demoModelSnapshot.findings.graph.dynamic_construct_ids,
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
