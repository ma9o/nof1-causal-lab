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
    const revised = structuredClone(demoSnapshotAt(8));
    revised.findings.fit!.source.validity = "stale";
    expect(availableGraphLayers(revised)).not.toContain("fit");
  });

  it("renders the backend selection while preserving the full structural model for inspection", () => {
    const structural = demoSnapshotAt(3);
    const measured = structuredClone(demoSnapshotAt(4));
    expect(graphEntities(structural).constructs).toHaveLength(17);
    const graph = graphEntities(measured);
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
    measured.findings.graph_status[graph.constructs[0].id] = "blocking";
    expect(graphEntities(measured)).toEqual(graph);
    expect(measured.model!.value.edges).toHaveLength(32);
    expect(graphEntities(demoSnapshotAt(2)).constructs).toEqual([]);
  });
});
