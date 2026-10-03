import { apiClient } from "@/lib/api/client";

export function getCapabilitiesQueryKey() {
  return ["capabilities"] as const;
}

export async function getCapabilities() {
  const { data, response } = await apiClient.GET("/api/actions-enabled", { cache: "no-store" });
  if (data === undefined) throw new Error(`Capabilities error ${response.status}`);
  return data;
}
