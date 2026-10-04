import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { indexModel } from "@/lib/model-asset/entities";
import type { Applied, TimelineRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { EditDetails } from "./edit-details";
import { ActionFindings } from "../action-findings";
import { humanize } from "@/lib/model-asset/selection";

const hooks = vi.hoisted(() => ({ diff: vi.fn() }));
vi.mock("@/lib/hooks/use-model-diff", () => ({ useModelDiff: hooks.diff }));

const context: ScopeContext = {
  ticks: [],
  dataDiff: null,
  result: undefined,
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
const tick: TimelineRevision = {
  commit_id: "rewritten-edit",
  parent_ids: ["preceding-commit"],
  record: {
    seq: 5,
    ts: "2026-09-30T00:00:00Z",
    messages: [],
    trace_ids: [],
    attempt: { action: "edit_model", request: { action: "edit_model", expected_revision: "archived-authorship-base", panel_revision: null, model: fixtureValue(demoModelSnapshot.model).value }, outcome: applied },
  },
};

describe("edit change summaries after history compaction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hooks.diff.mockReturnValue({ data: undefined, error: null });
  });

  it("compares the exact authored base named by the call", () => {
    renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(hooks.diff).toHaveBeenCalledWith(context.model.workspace_id, "archived-authorship-base", "rewritten-edit");
  });

  it("shows an initial summary when the call names no base", () => {
    const request = fixtureValue(tick.record.attempt.request);
    if (request.action !== "edit_model") throw new Error("Expected edit fixture");
    const created = { ...tick, record: { ...tick.record, attempt: { ...tick.record.attempt, request: { ...request, expected_revision: null } } } };
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick: created }));
    expect(hooks.diff).toHaveBeenCalledWith(context.model.workspace_id, null, null);
    expect(html).toContain("Model created");
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
