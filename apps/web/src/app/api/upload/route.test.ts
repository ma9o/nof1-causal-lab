import { afterEach, describe, expect, it, vi } from "vitest";
import { POST } from "./route";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";

describe("POST /api/upload", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("proxies the uploaded file to the facade upload endpoint", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          path: "NEWSPACE/input/data.csv",
        }),
      ),
    );

    const file = new File(["hello"], "data.csv", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "NEWSPACE");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      path: "NEWSPACE/input/data.csv",
    });

    const [url, init] = fixtureValue(vi.mocked(fetch).mock.calls.at(0));
    expect(url).toBe("http://localhost:8100/api/upload");
    expect(init?.method).toBe("POST");
    const body = init?.body;
    if (!(body instanceof FormData)) throw new Error("Missing multipart upload");
    expect(body.get("workspaceId")).toBe("NEWSPACE");
    const proxiedFile = body.get("file");
    if (!(proxiedFile instanceof File)) throw new Error("Missing proxied file");
    expect(proxiedFile.name).toBe("data.csv");
    expect(proxiedFile.size).toBe(file.size);
    expect(proxiedFile.type).toBe("text/csv");
  });

  it("rejects a multipart entry without a named file before proxying", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const file = new File(["hello"], "", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "BROKEN");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({ error: "No file provided" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("rejects malformed workspace ids", async () => {
    vi.stubGlobal("fetch", vi.fn());
    const file = new File(["hello"], "data.csv", { type: "text/csv" });
    const formData = new FormData();
    formData.set("workspaceId", "../etc");
    formData.set("file", file);

    const response = await POST(
      new Request("http://localhost/api/upload", {
        method: "POST",
        body: formData,
      }),
    );

    expect(response.status).toBe(400);
    expect(fetch).not.toHaveBeenCalled();
  });
});
