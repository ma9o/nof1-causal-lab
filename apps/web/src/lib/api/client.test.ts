import { describe, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { apiClient } from "./client";

describe("generated facade client", () => {
  it("returns the endpoint's response and accepts custom headers", async () => {
    const body = { actions_enabled: false };
    const fetch = vi.fn<(request: Request) => Promise<Response>>(async () => Response.json(body));
    const result = await apiClient.GET("/api/capabilities", {
      baseUrl: "http://viewer",
      fetch,
      headers: { Authorization: "Bearer token" },
    });
    expect(result.data).toEqual(body);
    const [request] = fixtureValue(fetch.mock.calls.at(0));
    expect(request.url).toBe("http://viewer/api/capabilities");
    expect(request.headers.get("Authorization")).toBe("Bearer token");
    expect(request.headers.has("Content-Type")).toBe(false);
  });

  it("serializes the endpoint's dispatch input as JSON", async () => {
    const fetch = vi.fn<(request: Request) => Promise<Response>>(async () =>
      Response.json({ attempt_id: "new-attempt" }, { status: 202 }),
    );
    const body = {
      action: "edit_model",
      expected_revision: null,
      model: { question: "Why?" },
    } as const;
    const result = await apiClient.POST("/api/studies/{workspace_id}/actions", {
      baseUrl: "http://viewer",
      fetch,
      params: { path: { workspace_id: "DEMO" } },
      body,
    });
    const [request] = fixtureValue(fetch.mock.calls.at(0));
    expect(request.method).toBe("POST");
    expect(request.headers.get("Content-Type")).toBe("application/json");
    expect(await request.json()).toEqual(body);
    expect(result.data).toEqual({ attempt_id: "new-attempt" });
  });

  it("retains unsuccessful transport status and payload for the caller", async () => {
    const result = await apiClient.GET("/api/capabilities", {
      baseUrl: "http://viewer",
      fetch: async () => Response.json({ detail: "Unavailable" }, { status: 503 }),
    });
    expect(result.data).toBeUndefined();
    expect(result.response.status).toBe(503);
    expect(result.error).toEqual({ detail: "Unavailable" });
  });

  it("propagates network errors", async () => {
    await expect(
      apiClient.GET("/api/capabilities", {
        baseUrl: "http://viewer",
        fetch: async () => {
          throw new TypeError("Failed to fetch");
        },
      }),
    ).rejects.toThrow("Failed to fetch");
  });
});
