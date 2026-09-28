import { getToolServerUrl } from "@/lib/runtime-urls";

/** Preserve scientific payloads, branch selection and optimistic heads end to end. */
export async function proxyEpisodeRequest(request: Request) {
  const url = new URL(request.url);
  const response = await fetch(`${getToolServerUrl()}${url.pathname}${url.search}`, {
    method: request.method,
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: request.method === "GET" ? undefined : await request.text(),
    cache: "no-store",
    signal: request.signal,
  });
  const headers = new Headers({ "Cache-Control": "no-store" });
  const contentType = response.headers.get("Content-Type");
  if (contentType !== null) headers.set("Content-Type", contentType);
  return new Response(response.body, { status: response.status, headers });
}
