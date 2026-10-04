import { createModelClient } from "@nof1-causal-lab/api-types";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { uploadFile } from "./endpoints";

const { fetcher } = vi.hoisted(() => ({
  fetcher: vi.fn<(request: Request) => Promise<Response>>(),
}));
vi.mock("./client", () => ({
  apiClient: createModelClient({ baseUrl: "http://viewer", fetch: fetcher }),
}));
beforeEach(() => fetcher.mockReset());

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

