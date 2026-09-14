import { modelConstructs } from "@/lib/model-accessors";
import { describe, expect, it } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { constructStatuses } from "./construct-statuses";

describe("constructStatuses", () => {
  it("preserves the backend findings when construct labels change", () => {
    const snapshot = structuredClone(demoModelSnapshot);
    const model = snapshot.model!.value;
    const first = modelConstructs(model)[0];
    snapshot.findings.graph_status = { [first.id]: "blocking" };
    first.name = "a renamed construct";
    expect(constructStatuses(snapshot)).toEqual({ "a renamed construct": "blocking" });
  });

  it("does not invent findings for incomplete models", () => {
    const snapshot = structuredClone(demoModelSnapshot);
    snapshot.findings.graph_status = {};
    expect(constructStatuses(snapshot)).toEqual({});
    snapshot.model = null;
    expect(constructStatuses(snapshot)).toEqual({});
  });
});
