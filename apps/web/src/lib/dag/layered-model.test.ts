import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { indexModel } from "@/lib/model-asset/entities";
import { describe, expect, it } from "vitest";
import { demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { availableGraphLayers, graphEntities } from "@/lib/dag/layered-model";

describe("semantic graph layers", () => {
  it("uses the facts available at each committed revision", () => {
    expect(availableGraphLayers(demoSnapshotAt(0))).toEqual([]);
    expect(availableGraphLayers(demoSnapshotAt(3))).toEqual(["structure"]);
    expect(availableGraphLayers(demoSnapshotAt(4))).toEqual(["structure", "measurement", "design"]);
    expect(availableGraphLayers(demoSnapshotAt(8))).toEqual([
      "structure",
      "measurement",
      "design",
      "specification",
      "fit",
    ]);
    const base = demoSnapshotAt(8);
    const revised = {
      ...base,
      findings: {
        ...base.findings,
        fit: {
          ...fixtureValue(base.findings.fit),
          source: { ...fixtureValue(base.findings.fit).source, validity: "stale" as const },
        },
      },
    };
    expect(availableGraphLayers(revised)).not.toContain("fit");
  });

  it("renders the backend selection while preserving the full structural model for inspection", () => {
    const structural = demoSnapshotAt(3);
    const measured = structuredClone(demoSnapshotAt(4));
    expect(graphEntities(structural, indexModel(structural.model?.value)).constructs).toHaveLength(
      17,
    );
    const graph = graphEntities(measured, indexModel(measured.model?.value));
    expect(graph.constructs).toHaveLength(13);
    expect(graph.constructs.map((item) => item.name)).not.toContain(
      "neuroadaptation_dependence_state",
    );
    expect(graph.constructs.map((item) => item.name)).not.toContain(
      "duration_current_escitalopram_use",
    );
    expect(graph.constructs.map((item) => item.id)).toEqual(measured.findings.graph.construct_ids);
    expect(graph.edges.map((item) => item.id)).toEqual(measured.findings.graph.edge_ids);
    // Identification status does not override the backend's retained-state selection.
    const marked = {
      ...measured,
      findings: {
        ...measured.findings,
        graph: {
          ...measured.findings.graph,
          status: {
            ...measured.findings.graph.status,
            [fixtureValue(graph.constructs[0]).id]: "blocking" as const,
          },
        },
      },
    };
    expect(graphEntities(marked, indexModel(marked.model?.value))).toEqual(graph);
    expect(fixtureValue(measured.model).value.edges).toHaveLength(32);
    expect(
      graphEntities(demoSnapshotAt(2), indexModel(demoSnapshotAt(2).model?.value)).constructs,
    ).toEqual([]);
  });
});
