import { modelResult } from "@/components/__fixtures__/action-results";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import { indexModel } from "@/lib/model-asset/entities";
import type { Applied, ActionSuccess, TimelineRevision } from "@nof1-causal-lab/api-types";
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
  model: authoredSnapshot,
  entities: indexModel(authoredSnapshot.model),
  select: vi.fn(),
};
const applied: Applied<null> = {
  status: "applied",
  result: null,
  effects: {
    retracted: [],
    reports: {},
    produced: [
      {
        artifact_id: "model",
        revision: "edited-model",
        derived_from: { model: "archived-authorship-base" },
        produced_by: "edit_model",
        created_at: "2026-09-30T00:00:00Z",
      },
    ],
  },
};
const tick: TimelineRevision = {
  call_id: `call:${"2".repeat(64)}`,
  commit_id: "rewritten-edit",
  parent_ids: ["preceding-commit"],
  record: {
    seq: 5,
    ts: "2026-09-30T00:00:00Z",
    messages: [],
    trace_ids: [],
    attempt: {
      action: "edit_model",
      request: {
        action: "edit_model",
        reasoning: null,
        input: {
          parent_ref: "archived-authorship-base",
          model: fixtureValue(authoredSnapshot.model),
        },
      },
      outcome: applied,
    },
  },
};

describe("edit change summaries after history compaction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hooks.diff.mockReturnValue({ data: undefined, error: null });
  });

  it("compares the exact authored base named by the call", () => {
    renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(hooks.diff).toHaveBeenCalledWith(
      context.model.workspace_id,
      "archived-authorship-base",
      "rewritten-edit",
      context.ticks,
    );
  });

  it("shows an initial summary when the parent is a question", () => {
    const request = fixtureValue(tick.record.attempt.request);
    if (request.action !== "edit_model") throw new Error("Expected edit fixture");
    const created = {
      ...tick,
      record: {
        ...tick.record,
        attempt: {
          ...tick.record.attempt,
          request: {
            ...request,
            input: {
              ...request.input,
              parent_ref: fixtureValue(authoredSnapshot.state.current.question).revision,
            },
          },
          outcome: {
            ...applied,
            effects: {
              ...applied.effects,
              produced: applied.effects.produced.map((artifact) => ({
                ...artifact,
                derived_from: {
                  question: fixtureValue(authoredSnapshot.state.current.question).revision,
                },
              })),
            },
          },
        },
      },
    };
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick: created }));
    expect(hooks.diff).toHaveBeenCalledWith(context.model.workspace_id, null, null, context.ticks);
    expect(html).toContain("Model created");
  });

  it("shows when no comparison has been recorded", () => {
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain("No saved comparison for these model versions.");
    expect(html).not.toContain("Reading model changes");
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
        before_model: fixtureValue(context.model.model),
        after_model: fixtureValue(context.model.model),
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
    const served: ActionSuccess = {
      status: "success",
      call_id: `call:${"2".repeat(64)}`,
      action: "edit_model",
      commit_id: tick.commit_id,
      messages: [],
      body: {
        ...modelResult(context.model),
        checks: {
          question: null,
          predictive: null,
          reused: ["specification"],
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
      createElement(ActionFindings, {
        context: { ...context, result: served },
        call: served,
      }),
    );
    expect(html).toContain(indicator.observation.id);
    expect(html).toContain("requires a likelihood.");
    expect(html).not.toContain("Do not list passing checks.");
  });
});
