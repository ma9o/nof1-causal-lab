import { describe, expect, it } from "vitest";
import { modelFixture, outcome, treatment } from "@/lib/__fixtures__/model";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { entityFailures } from "./inspector";
import { modelConstructs, modelEdges } from "@/lib/model-accessors";
import { indexModel, resolveEntity } from "./entities";
import type { EntitySelection } from "./selection";

/** Reorder the same document without changing entity identity or ownership. */
function reversedGraph() {
  return {
    ...modelFixture,
    constructs: Object.fromEntries(Object.entries(modelFixture.constructs).reverse()),
    edges: Object.fromEntries(Object.entries(modelFixture.edges).reverse()),
  };
}

describe("scoped model inspection", () => {
  it("reads the same entities when document entries move", () => {
    const reversed = reversedGraph();
    const constructs = modelConstructs(modelFixture);
    const selections: EntitySelection[] = [
      ...modelEdges(modelFixture).map((item) => ({ kind: "edge" as const, id: item.id })),
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

  it("attributes recorded data and identification issues without requiring a fit", () => {
    const indicator = fixtureValue(outcome.indicators[0]);
    const model = {
      ...authoredSnapshot,
      profile: {
        indicators: {
          [indicator.observation.id]: {
            profile: null,
            checks: {},
            issues: [
              {
                indicator_id: indicator.observation.id,
                issue_type: "missing",
                severity: "warning" as const,
                message: "Missing observations.",
              },
            ],
          },
        },
        dataset_issues: [],
        is_valid: true,
      },
      identification: {
        outcome: outcome.id,
        treatments: {
          [treatment.id]: { status: "not_identified" as const, confounders: [], notes: null },
        },
      },
    };
    expect(entityFailures(model, indicator)).toEqual(["Data quality: outcome_reading"]);
    expect(entityFailures(model, outcome)).toEqual(["Data quality: outcome_reading"]);
    expect(entityFailures(model, treatment)).toEqual(["Identification against ★: treatment"]);
    expect(entityFailures(authoredSnapshot, outcome)).toEqual([]);
  });
});
