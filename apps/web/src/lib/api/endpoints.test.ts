import { createModelClient } from "@nof1-causal-lab/api-types";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { callAction, uploadFile } from "./endpoints";

const { fetcher } = vi.hoisted(() => ({
  fetcher: vi.fn<(request: Request) => Promise<Response>>(),
}));
vi.mock("./client", () => ({
  apiClient: createModelClient({ baseUrl: "http://viewer", fetch: fetcher }),
}));
beforeEach(() => fetcher.mockReset());

it("submits a model comparison with its reasoning through the action route", async () => {
  const request = {
    action: "model_diff" as const,
    before: "1".repeat(40),
    after: "2".repeat(40),
    reasoning: "Compare the revised assumptions before fitting.",
  };
  const result = {
    kind: "running",
    attempt_id: "0f17a770-5d1e-4c2b-9a3f-6b8e2d4c1a90",
    request,
    messages: [],
    events: [],
  };
  fetcher.mockResolvedValue(Response.json(result));
  expect(await callAction("user-1", request)).toEqual(result);
  const [call] = fixtureValue(fetcher.mock.calls.at(0));
  expect(call.url).toBe("http://viewer/api/studies/user-1/model_diff");
  expect(call.method).toBe("POST");
  expect(await call.json()).toEqual(request);
});

describe("uploadFile", () => {
  it("sends a multipart body with its filename and workspace", async () => {
    fetcher.mockResolvedValue(Response.json("user-1/input/test.json"));
    const file = new File(["content"], "test.json", { type: "application/json" });
    expect(await uploadFile(file, "user-1")).toEqual("user-1/input/test.json");
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
