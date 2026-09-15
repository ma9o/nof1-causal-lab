import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/episode-runs", () => ({
  EpisodeRunError: class EpisodeRunError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  startStudyRecipe: vi.fn(),
}));

import { EpisodeRunError, startStudyRecipe } from "@/lib/server/episode-runs";
import { POST } from "./route";

function makeRequest(): Request {
  return new Request("http://localhost/api/analysis/user-1/recompute", { method: "POST" });
}

describe("POST /api/analysis/[workspaceId]/recompute", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("rejects malformed workspace ids", async () => {
    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "../etc" }),
    });

    expect(response.status).toBe(400);
    expect(startStudyRecipe).not.toHaveBeenCalled();
  });

  it("starts the auto-run driver", async () => {
    vi.mocked(startStudyRecipe).mockResolvedValue();

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true, workspaceId: "user-1" });
    expect(startStudyRecipe).toHaveBeenCalledWith("user-1");
  });

  it("treats an already-running auto-run (facade 409) as success", async () => {
    vi.mocked(startStudyRecipe).mockRejectedValue(
      new EpisodeRunError(409, "auto-run already active for user-1"),
    );

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ ok: true, workspaceId: "user-1" });
  });

  it("maps facade failures to their HTTP status", async () => {
    vi.mocked(startStudyRecipe).mockRejectedValue(new EpisodeRunError(502, "facade unavailable"));

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(502);
  });

  it("returns 502 on unexpected failures", async () => {
    vi.mocked(startStudyRecipe).mockRejectedValue(new Error("boom"));

    const response = await POST(makeRequest(), {
      params: Promise.resolve({ workspaceId: "user-1" }),
    });

    expect(response.status).toBe(502);
    await expect(response.json()).resolves.toEqual({ error: "Failed to start recompute" });
  });
});
