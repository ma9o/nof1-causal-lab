import { getToolServerUrl } from "@/lib/runtime-urls";

/** Forward the four scientific action contracts, including their validation results. */
export async function POST(request: Request) {
  const response = await fetch(`${getToolServerUrl()}${new URL(request.url).pathname}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
    signal: request.signal,
  });
  return new Response(response.body, {
    status: response.status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
