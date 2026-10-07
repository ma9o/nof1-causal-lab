import createClient, { type Client, type ClientOptions, type ParseAs } from "openapi-fetch";
import type { paths } from "./generated/model-api";
import { decodeModelMessage } from "./msgpack";

type WireRequest = (
  method: Parameters<Client<paths>["request"]>[0],
  path: string,
  options?: { parseAs?: ParseAs },
) => Promise<{ data?: unknown; error?: unknown; response: Response }>;

/** The same typed read client works directly against Python or through the Next.js proxy. */
export function createModelClient(options?: ClientOptions): Client<paths> {
  const raw = createClient<paths>(options);
  const send = raw.request as WireRequest;
  const request = (async (...args: Parameters<WireRequest>) => {
    const [method, path, options] = args;
    const requested = options?.parseAs;
    if (requested !== undefined && requested !== "json") return send(...args);
    const result = await send(method, path, { ...options, parseAs: "stream" });
    const response = result.response;
    if (!response.ok || response.status === 204 || method === "head") return result;
    const contentType = response.headers.get("Content-Type")?.split(";")[0];
    let data: unknown;
    switch (contentType) {
      case "application/msgpack":
        data = decodeModelMessage(new Uint8Array(await response.arrayBuffer()));
        break;
      case "application/json":
        data = await response.json();
        break;
      default:
        throw new Error(`Unsupported API response media type: ${contentType}`);
    }
    // Both codecs restore the same operation-owned schema at this transport boundary.
    return { ...result, data };
  }) as typeof raw.request;
  return {
    ...raw,
    request,
    GET: (path, ...init) => request("get", path, ...init),
    POST: (path, ...init) => request("post", path, ...init),
    PUT: (path, ...init) => request("put", path, ...init),
    PATCH: (path, ...init) => request("patch", path, ...init),
    DELETE: (path, ...init) => request("delete", path, ...init),
    HEAD: (path, ...init) => request("head", path, ...init),
    OPTIONS: (path, ...init) => request("options", path, ...init),
    TRACE: (path, ...init) => request("trace", path, ...init),
  };
}
