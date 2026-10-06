import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { indexModel } from "@/lib/model-asset/entities";
import { describe, expect, it } from "vitest";
import type { CausalEdgeSpec, ModelSnapshot, ModelSpec } from "@nof1-causal-lab/api-types";
import { baseline, modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot, emptySnapshot, fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { availableGraphLayers, graphEntities } from "@/lib/dag/layered-model";

const draftEndpoint = (endpoint: CausalEdgeSpec["cause"]): CausalEdgeSpec["cause"] =>
  "name" in endpoint ? { ...endpoint, dynamics: [], coefficients: [] } : endpoint;
const measuredModel: ModelSpec = {
  ...modelFixture,
  edges: modelFixture.edges.map((edge) => ({
    ...edge,
    cause: draftEndpoint(edge.cause),
    effect: draftEndpoint(edge.effect),
  })),
  parameters: [],
  distributions: {},
};
const structureEndpoint = (endpoint: CausalEdgeSpec["cause"]): CausalEdgeSpec["cause"] =>
  "name" in endpoint ? { ...endpoint, indicators: [] } : endpoint;
const structural: ModelSnapshot = {
  ...authoredSnapshot,
  model: {
    ...measuredModel,
    edges: measuredModel.edges.map((edge) => ({
      ...edge,
      cause: structureEndpoint(edge.cause),
      effect: structureEndpoint(edge.effect),
    })),
  },
  dispositions: null,
  authoring_prior_densities: {},
};
const measured: ModelSnapshot = {
  ...authoredSnapshot,
  model: measuredModel,
  authoring_prior_densities: {},
  graph: {
    construct_ids: [treatment.id, outcome.id],
    dynamic_construct_ids: [treatment.id, outcome.id],
    edge_ids: ["edge:00000000000000000002"],
    status: {},
  },
};

describe("semantic graph layers", () => {
  it("uses the facts available in each snapshot", () => {
    expect(availableGraphLayers(emptySnapshot)).toEqual([]);
    expect(availableGraphLayers(structural)).toEqual(["structure"]);
    expect(availableGraphLayers(measured)).toEqual(["structure", "measurement", "design"]);
    expect(availableGraphLayers(fittedSnapshot)).toEqual([
      "structure",
      "measurement",
      "design",
      "specification",
      "fit",
    ]);
    expect(availableGraphLayers({ ...fittedSnapshot, fit: null })).not.toContain("fit");
  });

  it("renders the backend selection while preserving the full structural model for inspection", () => {
    expect(
      graphEntities(structural, indexModel(structural.model)).constructs.map((item) => item.id),
    ).toEqual([baseline.id, treatment.id, outcome.id]);
    const graph = graphEntities(measured, indexModel(measured.model));
    expect(graph.constructs).toHaveLength(2);
    expect(graph.constructs.map((item) => item.id)).not.toContain(baseline.id);
    expect(graph.constructs.map((item) => item.id)).toEqual(measured.graph.construct_ids);
    expect(graph.edges.map((item) => item.id)).toEqual(measured.graph.edge_ids);
    // Identification status does not override the backend's retained-state selection.
    const marked = {
      ...measured,
      graph: {
        ...measured.graph,
        status: {
          ...measured.graph.status,
          [fixtureValue(graph.constructs[0]).id]: "blocking" as const,
        },
      },
    };
    expect(graphEntities(marked, indexModel(marked.model))).toEqual(graph);
    expect(fixtureValue(measured.model).edges).toHaveLength(2);
    expect(graphEntities(emptySnapshot, indexModel(emptySnapshot.model)).constructs).toEqual([]);
  });
});
