import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { indexModel } from "@/lib/model-asset/entities";
import type { Applied, StudyRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { EditDetails } from "./edit-details";
import { ActionFindings } from "../action-findings";
import { humanize } from "@/lib/model-asset/selection";

const hooks = vi.hoisted(() => ({ snapshot: vi.fn(), diff: vi.fn() }));
vi.mock("@/lib/hooks/use-model-snapshot", () => ({ useModelSnapshot: hooks.snapshot }));
vi.mock("@/lib/hooks/use-model-diff", () => ({ useModelDiff: hooks.diff }));

const context: ScopeContext = {
  ticks: [],
  dataDiff: null,
  model: demoModelSnapshot,
  entities: indexModel(demoModelSnapshot.model?.value),
  select: vi.fn(),
};
const applied: Applied<null> = {
  status: "applied",
  result: null,
  effects: {
    checks: null,
    retracted: [],
    produced: [
      {
        artifact_id: "model",
        revision: "edited-model",
        derived_from: { model: "archived-authorship-base" },
        model_inputs: {},
        consumed_model_inputs: {},
        produced_by: "edit_model",
        created_at: "2026-09-30T00:00:00Z",
      },
    ],
  },
};
const tick: StudyRevision = {
  commit_id: "rewritten-edit",
  parent_ids: ["preceding-commit"],
  record: {
    seq: 5,
    attempt_id: null,
    ts: "2026-09-30T00:00:00Z",
    branch: "main",
    messages: [],
    trace_ids: [],
    attempt: { action: "edit_model", request: null, outcome: applied },
  },
};

describe("edit change summaries after history compaction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hooks.snapshot.mockReturnValue({
      data: demoModelSnapshot,
      isPlaceholderData: false,
      error: null,
    });
    hooks.diff.mockReturnValue({ data: undefined, error: null });
  });

  it("compares to the preceding commit, even when the authorship base differs", () => {
    renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(hooks.snapshot).toHaveBeenCalledWith(
      context.model.workspace_id,
      "preceding-commit",
      "main",
      true,
    );
    expect(hooks.diff).toHaveBeenCalledWith(
      context.model.workspace_id,
      "preceding-commit",
      "rewritten-edit",
    );
  });

  it("shows an initial summary when the preceding commit has no model", () => {
    hooks.snapshot.mockReturnValue({
      data: { ...demoModelSnapshot, model: null },
      isPlaceholderData: false,
      error: null,
    });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(hooks.diff).toHaveBeenCalledWith(context.model.workspace_id, "preceding-commit", null);
    expect(html).toContain("Model created");
    expect(html).not.toContain("Reading model changes");
  });

  it("waits for the selected parent instead of using a previous query's placeholder", () => {
    hooks.snapshot.mockReturnValue({
      data: demoModelSnapshot,
      isPlaceholderData: true,
      error: null,
    });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain("Reading model changes");
    expect(hooks.diff).toHaveBeenCalledWith(context.model.workspace_id, "preceding-commit", null);
  });

  it("renders served graph and law changes in model terms, without definition paths or IDs", () => {
    const edge = fixtureValue(context.entities.edges[0]);
    const parameter = fixtureValue(context.entities.parameters[0]);
    hooks.diff.mockReturnValue({
      data: {
        constructs: [edge.cause, edge.effect].map((construct) => ({
          kind: "added",
          after: { kind: "construct", id: construct.id },
        })),
        edges: [{ kind: "added", after: { kind: "edge", id: edge.id } }],
        before_dispositions: [],
        after_dispositions: [],
        before_dynamic_construct_ids: [],
        after_dynamic_construct_ids: [],
        beforeModel: fixtureValue(context.model.model).value,
        afterModel: fixtureValue(context.model.model).value,
        parameters: [{ kind: "added", after: parameter }],
        changed_inputs: ["compilation", "belief"],
      },
      error: null,
    });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain(
      humanize(fixtureValue(context.entities.constructById.get(edge.cause.id)).name),
    );
    expect(html).toContain(humanize(parameter.name));
    expect(html).toContain("compilation, belief");
    expect(html).not.toContain("/internal/definition/path");
    expect(html).not.toMatch(/(?:construct|edge|parameter):/);
  });

  it("shows served check reasons unchanged and omits passing checks", () => {
    const indicator = fixtureValue(context.entities.indicators[0]);
    const checkedResult: Applied<null> = {
      ...applied,
      effects: {
        ...applied.effects,
        checks: {
          input_keys: {},
          question: null,
          predictive: null,
          reused: [],
          specification: [
            {
              kind: "evaluated",
              subject: "model_execution",
              outcome: "failed",
              evidence: `${indicator.observation.id} requires a likelihood.`,
            },
            {
              kind: "evaluated",
              subject: "passed_check",
              outcome: "passed",
              evidence: "Do not list passing checks.",
            },
          ],
        },
      },
    };
    const html = renderToStaticMarkup(
      createElement(ActionFindings, { context, applied: checkedResult }),
    );
    expect(html).toContain(indicator.observation.id);
    expect(html).toContain("requires a likelihood.");
    expect(html).not.toContain("Do not list passing checks.");
  });
});
