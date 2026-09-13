import {
  type LikelihoodSpec,
  type PriorProposal,
  type StatisticalModelSpecData,
} from "@nof1-causal-lab/api-types";

export function collectModelSpecUiPriors(data: StatisticalModelSpecData): PriorProposal[] {
  return data.statistical_model_spec.parameters.flatMap((parameter) => {
    const prior = data.authored_priors[parameter.id];
    return prior ? [prior] : [];
  });
}

export interface ModelSpecObservationPriorTerm {
  parameterName: string;
  prior?: PriorProposal;
}

/** Priors follow scientific definitions; execution locations stay in the compiler. */
export function collectModelSpecObservationPriorTerms({
  likelihood,
  priors,
  parameters,
}: {
  likelihood: LikelihoodSpec;
  priors: PriorProposal[];
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[];
}): ModelSpecObservationPriorTerm[] {
  const priorByParameter = new Map(priors.map((prior) => [prior.parameter_id, prior]));
  return parameters
    .filter((parameter) =>
      parameter.owners.some(
        (owner) => owner.kind === "indicator" && owner.id === likelihood.indicator_id,
      ),
    )
    .map((parameter) => ({
      parameterName: parameter.name,
      prior: priorByParameter.get(parameter.id),
    }));
}
