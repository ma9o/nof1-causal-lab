import { createModelClient } from "@nof1-causal-lab/api-types";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GET } from "@/app/api/episodes/[workspaceId]/model/[[...path]]/route";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("model read transport", () => {
  it.each([200, 404, 422])("preserves the backend body and status %i", async (status) => {
    vi.stubEnv("TOOL_SERVER_URL", "http://python:8100");
    const body = status === 200 ? '{"seq":7,"constructs":[]}' : '{"detail":"Invalid revision"}';
    const fetch = vi.fn().mockResolvedValue(
      new Response(body, {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);
    const request = new Request("http://localhost:3000/api/episodes/DEMO/model?at_seq=7");
    const response = await GET(request);
    expect(await response.text()).toBe(body);
    expect(response.status).toBe(status);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    expect(fetch).toHaveBeenCalledExactlyOnceWith(
      "http://python:8100/api/episodes/DEMO/model?at_seq=7",
      {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal: request.signal,
      },
    );
  });

  it("uses generated path and revision parameters for collection reads", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response("[]", {
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = createModelClient({ baseUrl: "http://python:8100", fetch });
    const result = await client.GET("/api/episodes/{workspace_id}/model/constructs", {
      params: { path: { workspace_id: "DEMO" }, query: { at_seq: 0 } },
    });
    expect(result.data).toEqual([]);
    expect(fetch.mock.calls[0][0].url).toBe(
      "http://python:8100/api/episodes/DEMO/model/constructs?at_seq=0",
    );
  });
});
