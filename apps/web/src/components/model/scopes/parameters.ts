import type { Expression, InferenceReport, ModelSpec } from "@nof1-causal-lab/api-types";
import { modelConstructs, referencedParameterIds } from "@/lib/model-accessors";
import { distributionText } from "@/lib/utils/distribution-format";
import type { PosteriorRow, PriorRow } from "../scope-primitives";

function expressionComponentsForConstruct(expression: Expression, id: string): Expression[] {
  if (expression.kind === "state") return expression.construct_id === id ? [expression] : [];
  if (expression.kind === "binary") {
    const { left, right } = expression;
    if (
      (left.kind === "state" && left.construct_id === id) ||
      (right.kind === "state" && right.construct_id === id)
    )
      return [expression];
    return [
      ...expressionComponentsForConstruct(left, id),
      ...expressionComponentsForConstruct(right, id),
    ];
  }
  if (expression.kind === "call") {
    return expression.arguments.flatMap((argument) =>
      expressionComponentsForConstruct(argument, id),
    );
  }
  return [];
}

export function parametersForOwner(model: ModelSpec | null | undefined, id: string) {
  if (!model) return [];
  const components = [
    ...modelConstructs(model).filter((construct) => construct.id === id),
    ...modelConstructs(model).flatMap((construct) => [
      ...construct.coefficients.filter((item) =>
        item.construct_ids.some((identity) => identity === id),
      ),
      ...construct.indicators.flatMap((indicator) =>
        Object.values(indicator.likelihood?.law.arguments ?? {}).flatMap((expression) =>
          expressionComponentsForConstruct(expression, id),
        ),
      ),
    ]),
    ...modelConstructs(model).flatMap((construct) =>
      construct.indicators.filter((indicator) => indicator.id === id),
    ),
    ...model.edges
      .filter(
        (edge) =>
          edge.id === id ||
          edge.cause.id === id ||
          edge.effect.id === id ||
          edge.mechanisms.some(
            (term) => expressionComponentsForConstruct(term.expression, id).length > 0,
          ),
      )
      .flatMap((edge) => edge.mechanisms),
  ];
  const ids = referencedParameterIds(components);
  return model.parameters.filter((parameter) => ids.has(parameter.id));
}

export function priorRows(
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[],
  distributions: ModelSpec["distributions"],
): PriorRow[] {
  return parameters.map(({ name, description, distribution_transform, distribution, value }) => {
    return {
      parameter: name,
      role: `${description}${distribution_transform === "dt_persistence_to_ct_decay" ? " (prior: persistence; posterior: decay rate)" : distribution_transform === "dt_effect_to_ct_rate" ? " (prior: interval effect; posterior: rate)" : ""}`,
      prior:
        value != null
          ? `Fixed: ${value}`
          : distribution
            ? distributionText(distributions[distribution])
            : null,
    };
  });
}

export function posteriorRows(
  parameters: import("@nof1-causal-lab/api-types").ParameterSpec[],
  posterior: InferenceReport | undefined,
): PosteriorRow[] {
  const marginals = posterior?.posterior_marginals ?? [];
  const ids = new Set(parameters.map((parameter) => parameter.id));
  return marginals.filter((marginal) => ids.has(marginal.subject.parameter_id));
}
