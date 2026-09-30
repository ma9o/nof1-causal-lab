import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { indexModel } from "@/lib/model-asset/entities";
import type { JournalTick } from "@/lib/model-asset/journal";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { EditDetails } from "./edit-details";

const hooks = vi.hoisted(() => ({ snapshot: vi.fn(), diff: vi.fn() }));
vi.mock("@/lib/hooks/use-model-snapshot", () => ({ useModelSnapshot: hooks.snapshot }));
vi.mock("@/lib/hooks/use-model-diff", () => ({ useModelDiff: hooks.diff }));

const context: ScopeContext = {
  model: demoModelSnapshot,
  entities: indexModel(demoModelSnapshot.model?.value),
  ticks: [],
  select: vi.fn(),
};
const tick: JournalTick = {
  seq: 5,
  attemptId: null,
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
      produced_by: "write:model",
      created_at: "2026-09-30T00:00:00Z",
    },
  ],
  revision: "edited-model",
  derived: [],
  retracted: [],
  error: null,
  errorType: null,
  traceIds: [],
  diagnostics: {},
  messages: [],
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
    expect(html).toContain("construct");
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
});
