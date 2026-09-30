import { describe, expect, it } from "vitest";
import { demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { modelConstructs } from "@/lib/model-accessors";
import { lawCurves, lawLabel, ownLawUses } from "./laws";

describe("law curves", () => {
  const authored = demoSnapshotAt(7);
  const fitted = demoSnapshotAt(8);
  const construct = modelConstructs(authored.model!.value).find(
    (item) => item.name === "internalizing_symptom_burden",
  )!;

  it("follows a construct's own dynamics and noise laws by their authored roles", () => {
    expect(ownLawUses(construct).map((use) => use.role)).toEqual(["decay", "diffusion_scale"]);
  });

  it("draws authored laws as written until a fit supersedes them with posteriors", () => {
    const [persistence] = lawCurves(authored, ownLawUses(construct));
    expect(persistence).toMatchObject({ kind: "authored", family: "Beta", posteriors: [] });
    expect(persistence.prior.length).toBeGreaterThan(0);
    expect(lawLabel(persistence)).toBe("persistence");

    const [decay] = lawCurves(fitted, ownLawUses(construct));
    expect(decay).toMatchObject({ kind: "fitted", family: null, stale: false });
    expect(decay.posteriors.map((marginal) => marginal.subject.parameter_id)).toEqual([
      persistence.parameter.id,
    ]);
    expect(lawLabel(decay)).toBe("decay");
  });

  it("marks posteriors conditioned on another panel", () => {
    const revised = structuredClone(fitted);
    revised.findings.fit!.source.validity = "stale";
    expect(lawCurves(revised, ownLawUses(construct)).map((curve) => curve.stale)).toEqual([
      true,
      true,
    ]);
  });
});
