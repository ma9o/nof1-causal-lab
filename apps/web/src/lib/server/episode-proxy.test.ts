import { afterEach, expect, it, vi } from "vitest";
import { proxyEpisodeRequest } from "./episode-proxy";

vi.mock("@/lib/runtime-urls", () => ({ getToolServerUrl: () => "http://tool-server" }));
afterEach(() => vi.unstubAllGlobals());

it("preserves branch, expected head, payload and backend conflicts for submissions", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValue(Response.json({ detail: "Branch conflict" }, { status: 409 }));
  vi.stubGlobal("fetch", fetcher);
  const body = JSON.stringify({ action: "edit_model", expected_revision: 1, model: {} });
  const path = "/api/episodes/STUDY/actions?branch=alternative&expected_head=abc";
  const response = await proxyEpisodeRequest(
    new Request(`http://web${path}`, {
      method: "POST",
      body,
    }),
  );
  expect(fetcher).toHaveBeenCalledWith(
    `http://tool-server${path}`,
    expect.objectContaining({
      method: "POST",
      body,
    }),
  );
  expect(response.status).toBe(409);
  expect(await response.json()).toEqual({ detail: "Branch conflict" });
});
