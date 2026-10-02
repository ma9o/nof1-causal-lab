import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { indexModel } from "@/lib/model-asset/entities";
import type { ModelEditResult, StudyRevision } from "@nof1-causal-lab/api-types";
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
const result: ModelEditResult = {
  action: "edit_model",
  checks: null,
  retracted: [],
  base: { workspace_id: "DEMO", revision: "archived-authorship-base", path: "model.json" },
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
    attempt: { action: "edit_model", request: null, outcome: { status: "applied", result } },
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
      context.model.context.workspace_id,
      "preceding-commit",
      "main",
      true,
    );
    expect(hooks.diff).toHaveBeenCalledWith(
      context.model.context.workspace_id,
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
    expect(hooks.diff).toHaveBeenCalledWith(
      context.model.context.workspace_id,
      "preceding-commit",
      null,
    );
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
    expect(hooks.diff).toHaveBeenCalledWith(
      context.model.context.workspace_id,
      "preceding-commit",
      null,
    );
  });

  it("renders served graph and law changes in model terms, without definition paths or IDs", () => {
    const edge = fixtureValue(context.entities.edges[0]);
    const parameter = fixtureValue(context.entities.parameters[0]);
    hooks.diff.mockReturnValue({
      data: {
        graph: {
          constructs: [edge.cause, edge.effect].map((construct) => ({
            construct_id: construct.id,
            change: {
              kind: "added",
              after: fixtureValue(context.entities.constructById.get(construct.id)),
            },
          })),
          edges: [{ edge_id: edge.id, change: { kind: "added", after: edge } }],
        },
        parameters: [{ parameter_id: parameter.id, change: { kind: "added", after: parameter } }],
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
    const checkedResult: ModelEditResult = {
      ...result,
      checks: {
        input_keys: {},
        predictive: null,
        reused: [],
        specification: {
          findings: [
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
      createElement(ActionFindings, { context, result: checkedResult }),
    );
    expect(html).toContain(indicator.observation.id);
    expect(html).toContain("requires a likelihood.");
    expect(html).not.toContain("Do not list passing checks.");
  });
});
