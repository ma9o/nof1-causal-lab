import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server/study-runs", () => ({
  StudyRunError: class StudyRunError extends Error {
    constructor(
      public status: number,
      message: string,
    ) {
      super(message);
    }
  },
  getStudyTrace: vi.fn(),
}));

import { getStudyTrace } from "@/lib/server/study-runs";
import { GET } from "./route";

describe("GET /api/traces/[workspaceId]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("merges the named traces of one recorded action", async () => {
    vi.mocked(getStudyTrace)
      .mockResolvedValueOnce({
        messages: [{ role: "assistant", content: "A", tool_is_error: false }],
        model: "model-a",
        total_time_seconds: 1,
        usage: { input_tokens: 2, output_tokens: 3, reasoning_tokens: null },
      })
      .mockResolvedValueOnce({
        messages: [{ role: "assistant", content: "B", tool_is_error: false }],
        model: "model-b",
        total_time_seconds: 4,
        usage: { input_tokens: 5, output_tokens: 6, reasoning_tokens: 7 },
      });

    const response = await GET(
      new Request(
        `http://localhost/api/traces/DEMO?commitId=${"a".repeat(40)}&trace=construct-a&trace=construct-b`,
      ),
      { params: Promise.resolve({ workspaceId: "DEMO" }) },
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      messages: [
        { role: "assistant", content: "A", tool_is_error: false },
        { role: "assistant", content: "B", tool_is_error: false },
      ],
      model: "model-b",
      total_time_seconds: 5,
      usage: { input_tokens: 7, output_tokens: 9, reasoning_tokens: 7 },
    });
    expect(getStudyTrace).toHaveBeenNthCalledWith(1, "DEMO", "a".repeat(40), "construct-a");
    expect(getStudyTrace).toHaveBeenNthCalledWith(2, "DEMO", "a".repeat(40), "construct-b");
  });

  it("returns 404 when the action names no traces", async () => {
    const response = await GET(
      new Request(`http://localhost/api/traces/DEMO?commitId=${"b".repeat(40)}`),
      { params: Promise.resolve({ workspaceId: "DEMO" }) },
    );

    expect(response.status).toBe(404);
    expect(getStudyTrace).not.toHaveBeenCalled();
  });
});
