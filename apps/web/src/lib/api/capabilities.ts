import { apiFetch } from "@/lib/api/client";

export interface FacadeCapabilities {
  actions_enabled: boolean;
}

export function getCapabilitiesQueryKey() {
  return ["capabilities"] as const;
}

export async function getCapabilities(): Promise<FacadeCapabilities> {
  return apiFetch<FacadeCapabilities>("/api/capabilities", { cache: "no-store" });
}
