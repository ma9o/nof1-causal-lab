import type { InterventionSpec } from "@nof1-causal-lab/api-types";

export function formatInterventionValue(intervention: InterventionSpec): string {
  return `set ${intervention.value.toFixed(1)}`;
}
