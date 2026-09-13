import type { LikelihoodSpec, ParameterSpec } from "@nof1-causal-lab/api-types";

/** Observation priors belong to their parameters; owners identify the measurement. */
export function collectModelSpecObservationPriorTerms({
  likelihood,
  parameters,
}: {
  likelihood: LikelihoodSpec;
  parameters: ParameterSpec[];
}): ParameterSpec[] {
  return parameters.filter((parameter) =>
    parameter.owners.some(
      (owner) => owner.kind === "indicator" && owner.id === likelihood.indicator_id,
    ),
  );
}
