import { apiClient } from "@/lib/api/client";

export function getCapabilitiesQueryKey() {
  return ["capabilities"] as const;
}

export async function getCapabilities() {
  const { data, response } = await apiClient.GET("/api/workspaces", { cache: "no-store" });
  if (data === undefined) throw new Error(`Capabilities error ${response.status}`);
  const enabled = response.headers.get("X-Actions-Enabled");
  if (enabled !== "true" && enabled !== "false") throw new Error("Workspace response has no action capability");
  return enabled === "true";
}
