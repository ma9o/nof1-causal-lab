import type { ConstructId, InterventionSpec } from "@nof1-causal-lab/api-types";
import type { SimulationWithEffects } from "@/lib/simulation-report";

export function formatInterventionValue(intervention: InterventionSpec): string {
  return `set ${intervention.value.toFixed(1)}`;
}

/** do(...) description joining the assignments in the scenario. */
export function formatScenarioActionDescription(result: SimulationWithEffects): string {
  return result.design.interventions
    .map(
      (clamp) =>
        `do(${result.causal_result.labels[clamp.target]} ${formatInterventionValue(clamp)})`,
    )
    .join(", ");
}

export function getSimulationDays(result: SimulationWithEffects): number[] {
  return result.times;
}

export function getNodeReferenceSeries(
  result: SimulationWithEffects,
  nodeId: ConstructId,
): number[] | null {
  const series = result.predictive.states[nodeId]?.reference;
  return series?.kind === "numeric" && series.mean.every((value) => value !== null)
    ? series.mean
    : null;
}

export function getNodeActionSeries(
  result: SimulationWithEffects,
  nodeId: ConstructId,
): number[] | null {
  const series = result.predictive.states[nodeId]?.action;
  return series?.kind === "numeric" && series.mean.every((value) => value !== null)
    ? series.mean
    : null;
}
