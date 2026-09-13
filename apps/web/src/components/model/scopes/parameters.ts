import { distributionText } from "@/lib/utils/distribution-format";
import type { PosteriorArtifact } from "@nof1-causal-lab/api-types";
import type { PosteriorRow, PriorRow } from "../scope-primitives";

export function parametersForOwner(
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[],
  id: string,
) {
  return parameters.filter((parameter) => parameter.owners.some((owner) => owner.id === id));
}

export function priorRows(
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[],
): PriorRow[] {
  return parameters.map(({ name, role, prior_transform, prior }) => {
    return {
      parameter: name,
      role: `${role.replaceAll("_", " ")}${prior_transform === "dt_persistence_to_ct_decay" ? " (prior: persistence; posterior: decay rate)" : prior_transform === "dt_effect_to_ct_rate" ? " (prior: interval effect; posterior: rate)" : ""}`,
      prior: prior ? distributionText(prior) : null,
    };
  });
}

export function posteriorRows(
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[],
  posterior: PosteriorArtifact | undefined,
): PosteriorRow[] {
  const marginals = posterior?.posterior_marginals ?? [];
  const diagnostics = posterior?.assessment.mcmc_diagnostics?.per_parameter ?? [];
  const ids = new Set(parameters.map((parameter) => parameter.id));
  return marginals
    .filter((marginal) => ids.has(marginal.subject.parameter_id))
    .map((marginal) => {
      const diagnostic = diagnostics.find(
        (entry) =>
          entry.subject.parameter_id === marginal.subject.parameter_id &&
          entry.subject.element_id === marginal.subject.element_id,
      );
      return {
        ...marginal,
        rhat: diagnostic?.r_hat ?? null,
        ess: diagnostic?.ess_bulk ?? null,
      };
    });
}
