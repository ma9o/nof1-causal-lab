import { modelConstructs } from "@/lib/model-accessors";
import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
export type ConstructStatus = "observed" | "marginalized" | "blocking";

/** Label the backend's construct findings for the graph renderer. */
export function constructStatuses(snapshot: ModelSnapshot): Record<string, ConstructStatus> {
  return Object.fromEntries(
    (modelConstructs(snapshot.model?.value) ?? []).flatMap((construct) => {
      const status = snapshot.findings.graph_status[construct.id];
      return status ? [[construct.name, status]] : [];
    }),
  );
}
