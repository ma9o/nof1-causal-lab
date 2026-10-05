import { afterEach, expect, it, vi } from "vitest";
import { proxyStudyRequest } from "./study-proxy";

vi.mock("@/lib/runtime-urls", () => ({ getToolServerUrl: () => "http://tool-server" }));
afterEach(() => vi.unstubAllGlobals());

it("preserves named inputs, payload and backend rejections for calls", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValue(Response.json({ detail: "Input unavailable" }, { status: 422 }));
  vi.stubGlobal("fetch", fetcher);
  const body = JSON.stringify({
    action: "edit_model",
    expected_revision: "a".repeat(40),
    model: {},
  });
  const path = "/api/studies/STUDY/edit_model";
  const response = await proxyStudyRequest(
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
  expect(response.status).toBe(422);
  expect(await response.json()).toEqual({ detail: "Input unavailable" });
});
