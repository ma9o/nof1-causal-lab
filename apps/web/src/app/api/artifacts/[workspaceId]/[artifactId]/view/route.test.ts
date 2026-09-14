import { EpisodeRunError, getModelView } from "@/lib/server/episode-runs";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";
vi.mock("@/lib/server/episode-runs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/server/episode-runs")>()),
  getModelView: vi.fn(),
}));
beforeEach(() => vi.clearAllMocks());
const request = new Request("http://localhost/api/artifacts/DEMO/inference_report/view");
const context = {
  params: Promise.resolve({ workspaceId: "DEMO", artifactId: "inference_report" }),
};
describe("artifact view proxy", () => {
  it("preserves an unavailable projection as 404", async () => {
    vi.mocked(getModelView).mockRejectedValue(new EpisodeRunError(404, "No compatible view"));
    expect((await GET(request, context)).status).toBe(404);
  });
  it("passes the backend projection through unchanged", async () => {
    vi.mocked(getModelView).mockResolvedValue({ marker: "revision-pinned" } as never);
    expect(await (await GET(request, context)).json()).toEqual({ marker: "revision-pinned" });
    expect(getModelView).toHaveBeenCalledWith("DEMO", "inference_report");
  });
});
