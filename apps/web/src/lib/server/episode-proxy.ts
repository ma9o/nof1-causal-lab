import { getToolServerUrl } from "@/lib/runtime-urls";

/** Preserve backend validation and version selection for scientific reads. */
export async function proxyEpisodeRead(request: Request) {
  const url = new URL(request.url);
  const response = await fetch(`${getToolServerUrl()}${url.pathname}${url.search}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal: request.signal,
  });
  const headers = new Headers({ "Cache-Control": "no-store" });
  const contentType = response.headers.get("Content-Type");
  if (contentType !== null) headers.set("Content-Type", contentType);
  return new Response(response.body, { status: response.status, headers });
}
