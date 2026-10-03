import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { describe, expect, it } from "vitest";
import { workbenchJournal } from "@/components/__fixtures__/workbench";
import { attemptError, latestSeq } from "./journal";

describe("owned journal outcomes", () => {
  it("presents rejection and execution failure details from their respective variants", () => {
    expect(
      attemptError({ status: "rejected", reason: "scientific_inputs", detail: "inputs missing" }),
    ).toBe("inputs missing");
    const raised = fixtureValue(workbenchJournal[5]).record.attempt.outcome;
    expect(attemptError(raised)).toContain("ProposalError: Parameter proposal failed.");
    expect(attemptError(fixtureValue(workbenchJournal[0]).record.attempt.outcome)).toBeNull();
  });
  it("excludes failed attempts and read-only comparison leaves from scientific revisions", () => {
    expect(latestSeq(workbenchJournal.slice(0, 6))).toBe(5);
    expect(latestSeq(workbenchJournal)).toBe(12);
  });
  it("keeps a successful simulation checkpoint without an artifact output", () => {
    const simulation = fixtureValue(workbenchJournal[8]);
    const outcome = simulation.record.attempt.outcome;
    expect(outcome.status).toBe("applied");
    if (outcome.status !== "applied") throw new Error("Expected a successful fixture simulation");
    expect(outcome.effects.produced).toEqual([]);
    expect(latestSeq(workbenchJournal.slice(0, 9))).toBe(9);
  });
});
