import { EpisodeRunError, createStudy } from "@/lib/server/episode-runs";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  createStudy: vi.fn(),
}));

import { POST } from "./route";

describe("POST /api/runs", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requires a query", async () => {
    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123" }),
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: "query is required",
    });
  });

  it("rejects malformed workspace ids", async () => {
    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "../etc", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(400);
    expect(createStudy).not.toHaveBeenCalled();
  });

  it("creates the workspace without starting an optional recipe", async () => {
    vi.mocked(createStudy).mockResolvedValue({ attempt_id: "accepted-attempt" });

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: "USER123",
          query: " Why is sleep worse after travel? ",
        }),
      }),
    );

    expect(response.status).toBe(202);
    await expect(response.json()).resolves.toEqual({
      workspaceId: "USER123",
      attempt_id: "accepted-attempt",
    });
    expect(createStudy).toHaveBeenCalledWith("USER123", "Why is sleep worse after travel?");
  });

  it("returns a workspace revision conflict", async () => {
    vi.mocked(createStudy).mockRejectedValue(
      new EpisodeRunError(409, "auto-run already active for USER123"),
    );

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: "A run is already active for this workspace.",
    });
  });

  it("maps facade errors to their HTTP status", async () => {
    vi.mocked(createStudy).mockRejectedValue(new EpisodeRunError(403, "facade is read-only"));

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "facade is read-only" });
  });

  it("returns 502 on unexpected launch failures", async () => {
    vi.mocked(createStudy).mockRejectedValue(new Error("boom"));

    const response = await POST(
      new Request("http://localhost/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workspaceId: "USER123", query: "Why?" }),
      }),
    );

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({
      error: "Failed to create workspace",
    });
  });
});
