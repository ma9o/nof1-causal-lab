import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { indexModel } from "@/lib/model-asset/entities";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { EditDetails } from "./edit-details";
import { ActionFindings } from "../action-findings";
import { humanize } from "@/lib/model-asset/selection";

const hooks = vi.hoisted(() => ({ snapshot: vi.fn(), diff: vi.fn() }));
vi.mock("@/lib/hooks/use-model-snapshot", () => ({ useModelSnapshot: hooks.snapshot }));
vi.mock("@/lib/hooks/use-model-diff", () => ({ useModelDiff: hooks.diff }));

const context: ScopeContext = {
  ticks: [],
  dataDiff: { data: undefined, error: null },
  model: demoModelSnapshot,
  entities: indexModel(demoModelSnapshot.model?.value),
  select: vi.fn(),
};
const tick: JournalTick = {
  messages: [],
  seq: 5,
  commitId: "rewritten-edit",
  parentIds: ["preceding-commit"],
  branch: "main",
  ts: "2026-09-30T00:00:00Z",
  action: "edit_model",
  inputs: {},
  status: "applied",
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
  error: null,
  traceIds: [],
  checks: null,
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
    const edge = context.entities.edges[0];
    const parameter = context.entities.parameters[0];
    hooks.diff.mockReturnValue({
      data: {
        graph: {
          constructs: [edge.cause, edge.effect].map((construct) => ({
            construct_id: construct.id,
            change: "added",
            after: context.entities.constructById.get(construct.id),
          })),
          edges: [{ edge_id: edge.id, change: "added", after: edge }],
        },
        parameters: [{ parameter_id: parameter.id, change: "added", after: parameter }],
        changed_inputs: ["compilation", "belief"],
        definition_changes: [{ path: "/internal/definition/path" }],
      },
      error: null,
    });
    const html = renderToStaticMarkup(createElement(EditDetails, { context, tick }));
    expect(html).toContain(humanize(context.entities.constructById.get(edge.cause.id)!.name));
    expect(html).toContain(humanize(parameter.name));
    expect(html).toContain("compilation, belief");
    expect(html).not.toContain("/internal/definition/path");
    expect(html).not.toMatch(/(?:construct|edge|parameter):/);
  });

  it("shows served check reasons unchanged and omits passing checks", () => {
    const indicator = context.entities.indicators[0];
    const checkedTick: JournalTick = {
      ...tick,
      checks: {
        input_keys: {},
        reused: [],
        specification: {
          findings: [
            {
              check: "model_execution",
              status: "failed",
              message: `${indicator.id} requires a likelihood.`,
            },
            { check: "passed_check", status: "passed", message: "Do not list passing checks." },
          ],
        },
      },
    };
    const html = renderToStaticMarkup(
      createElement(ActionFindings, { context, tick: checkedTick }),
    );
    expect(html).toContain(indicator.id);
    expect(html).toContain("requires a likelihood.");
    expect(html).not.toContain("Do not list passing checks.");
  });
});
