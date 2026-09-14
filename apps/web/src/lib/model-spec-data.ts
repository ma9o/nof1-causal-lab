import type { IndicatorSpec, ParameterSpec } from "@nof1-causal-lab/api-types";
import { referencedParameterIds } from "./model-accessors";

/** Display the parameters referenced by this indicator’s likelihood coefficients. */
export function collectModelSpecObservationPriorTerms({
  indicator,
  parameters,
}: {
  indicator: IndicatorSpec;
  parameters: ParameterSpec[];
}): ParameterSpec[] {
  const ids = referencedParameterIds(indicator.likelihood);
  return parameters.filter((parameter) => ids.has(parameter.id));
}
