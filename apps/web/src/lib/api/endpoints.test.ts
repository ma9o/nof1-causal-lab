import { createModelClient, type LLMTrace } from "@nof1-causal-lab/api-types";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { getLLMTraceForAction, uploadFile } from "./endpoints";

const { fetcher } = vi.hoisted(() => ({
  fetcher: vi.fn<(request: Request) => Promise<Response>>(),
}));
vi.mock("./client", () => ({
  apiClient: createModelClient({ baseUrl: "http://viewer", fetch: fetcher }),
}));
beforeEach(() => fetcher.mockReset());

describe("uploadFile", () => {
  it("sends a multipart body with its filename and workspace", async () => {
    fetcher.mockResolvedValue(Response.json({ path: "/uploads/test.json" }));
    const file = new File(["content"], "test.json", { type: "application/json" });
    expect(await uploadFile(file, "user-1")).toEqual({ path: "/uploads/test.json" });
    const [request] = fixtureValue(fetcher.mock.calls.at(0));
    expect(request.url).toBe("http://viewer/api/upload");
    expect(request.method).toBe("POST");
    expect(request.headers.get("Content-Type")).toContain("multipart/form-data; boundary=");
    const form = await request.formData();
    expect(form.get("workspaceId")).toBe("user-1");
    const uploaded = form.get("file");
    if (!(uploaded instanceof File)) throw new Error("Missing uploaded file");
    expect(uploaded.name).toBe(file.name);
    expect(await uploaded.text()).toBe("content");
  });

  it("throws on upload failure", async () => {
    fetcher.mockResolvedValue(Response.json({ detail: "Too large" }, { status: 413 }));
    await expect(uploadFile(new File(["x"], "big.json"), "user-1")).rejects.toThrow(
      "Upload failed: 413",
    );
  });
});

describe("recorded action traces", () => {
  it("merges the named traces in order without losing usage or messages", async () => {
    const traces: LLMTrace[] = [
      {
        messages: [
          {
            role: "assistant",
            content: "A",
            reasoning: null,
            tool_calls: null,
            tool_call_id: null,
            tool_name: null,
            tool_result: null,
            tool_is_error: false,
          },
        ],
        model: "model-a",
        total_time_seconds: 1,
        usage: { input_tokens: 2, output_tokens: 3, reasoning_tokens: null },
      },
      {
        messages: [
          {
            role: "assistant",
            content: "B",
            reasoning: null,
            tool_calls: null,
            tool_call_id: null,
            tool_name: null,
            tool_result: null,
            tool_is_error: false,
          },
        ],
        model: "model-b",
        total_time_seconds: 4,
        usage: { input_tokens: 5, output_tokens: 6, reasoning_tokens: 7 },
      },
    ];
    for (const trace of traces) fetcher.mockResolvedValueOnce(Response.json(trace));
    const commit = "a".repeat(40);
    expect(await getLLMTraceForAction("DEMO", commit, ["construct:a", "construct:b"])).toEqual({
      messages: traces.flatMap((trace) => trace.messages),
      model: "model-b",
      total_time_seconds: 5,
      usage: { input_tokens: 7, output_tokens: 9, reasoning_tokens: 7 },
    });
    expect(fetcher.mock.calls.map(([request]) => request.url)).toEqual([
      `http://viewer/api/studies/DEMO/traces/${commit}/construct%3Aa`,
      `http://viewer/api/studies/DEMO/traces/${commit}/construct%3Ab`,
    ]);
  });

  it("makes no requests when an action has no traces", async () => {
    expect((await getLLMTraceForAction("DEMO", "b".repeat(40), [])).messages).toEqual([]);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("reports an unavailable recorded trace", async () => {
    fetcher.mockResolvedValue(Response.json({ detail: "Trace missing" }, { status: 404 }));
    await expect(getLLMTraceForAction("DEMO", "b".repeat(40), ["missing"])).rejects.toThrow(
      "Trace API error 404",
    );
  });
});
