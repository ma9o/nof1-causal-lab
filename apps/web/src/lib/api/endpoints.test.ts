import { createModelClient, type TimelineRevision } from "@nof1-causal-lab/api-types";
import { beforeEach, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { getLLMTraceForAction, readActionResult } from "./endpoints";

const { fetcher } = vi.hoisted(() => ({
  fetcher: vi.fn<(request: Request) => Promise<Response>>(),
}));
vi.mock("./client", () => ({
  apiClient: createModelClient({ baseUrl: "http://viewer", fetch: fetcher }),
}));
beforeEach(() => fetcher.mockReset());

const comparison: TimelineRevision = {
  call_id: `call:${"3".repeat(64)}`,
  commit_id: "3".repeat(40),
  parent_ids: [],
  record: {
    seq: 1,
    ts: "2026-10-06T12:00:00Z",
    messages: [],
    trace_ids: [],
    attempt: {
      action: "model_diff",
      request: {
        action: "model_diff",
        input: { before_ref: "1".repeat(40), after_ref: "2".repeat(40) },
        reasoning: "Compare the revised assumptions before fitting.",
      },
      outcome: {
        status: "applied",
        result: null,
        effects: { produced: [], retracted: [], reports: { "model-diff": "4".repeat(40) } },
      },
    },
  },
};

it("reads a recorded comparison by call ID without submitting inputs", async () => {
  const result = {
    call_id: comparison.call_id,
    action: "model_diff",
    status: "success",
    commit_id: comparison.commit_id,
    body: {},
    messages: [],
  };
  fetcher.mockResolvedValue(Response.json(result));
  expect(await readActionResult("user-1", comparison)).toEqual(result);
  const [call] = fixtureValue(fetcher.mock.calls.at(0));
  expect(decodeURIComponent(call.url)).toBe(
    `http://viewer/api/studies/user-1/model_diff/${comparison.call_id}`,
  );
  expect(call.method).toBe("GET");
  expect(await call.text()).toBe("");
});

it("rejects a historical entry without a call ID before reading", async () => {
  await expect(readActionResult("user-1", { ...comparison, call_id: null })).rejects.toThrow(
    "This historical attempt has no retained call identity",
  );
  expect(fetcher).not.toHaveBeenCalled();
});

it("reports unavailable saved results without retrying execution", async () => {
  fetcher.mockResolvedValue(Response.json({ status: "running" }));
  await expect(readActionResult("user-1", comparison)).rejects.toThrow(
    "The saved successful call is unavailable",
  );
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it("reads retained traces from a failed call's messages", async () => {
  const trace = {
    messages: [{ role: "assistant", content: "Inspecting the observations." }],
    model: "test",
    total_time_seconds: 1,
    usage: { input_tokens: 3, output_tokens: 5, reasoning_tokens: null },
  };
  fetcher.mockResolvedValue(
    Response.json({
      call_id: comparison.call_id,
      action: "model_diff",
      status: "failed",
      commit_id: comparison.commit_id,
      body: null,
      messages: [{ kind: "trace", timestamp: comparison.record.ts, trace_id: "worker", trace }],
    }),
  );
  expect(await getLLMTraceForAction("user-1", comparison, ["worker"])).toEqual(trace);
  expect(fixtureValue(fetcher.mock.calls[0])[0].method).toBe("GET");
});
