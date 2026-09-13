import type {
  DistributionFamily,
  LikelihoodSpec,
  LinkFunction,
  PriorDistributionFamily,
  PriorProposal,
  StructuralPlan,
} from "@nof1-causal-lab/api-types";

export type LabeledLikelihood = LikelihoodSpec & { label: string };

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
export function textify(name: string): string {
  return name.replace(/_/g, " ");
}

/** Build the g⁻¹(·) wrapper for a link function around a linear predictor. */
const LINK_INVERSE: Record<LinkFunction, (predictor: string) => string> = {
  identity: (p) => p,
  log: (p) => `\\exp(${p})`,
  inverse: (p) => `(${p})^{-1}`,
  logit: (p) => `\\sigma(${p})`,
  probit: (p) => `\\Phi(${p})`,
  cumulative_logit: (p) => `\\text{cumlogit}^{-1}(${p})`,
  softmax: (p) => `\\text{softmax}(${p})`,
};

export function linkInverse(link: string, predictor: string): string {
  const fn = LINK_INVERSE[link as LinkFunction];
  return fn ? fn(predictor) : `g^{-1}(${predictor})`;
}

/** Map distribution family enum to LaTeX name. */
const DIST_LATEX: Record<DistributionFamily, string> = {
  gaussian: "\\mathcal{N}",
  student_t: "t_{\\nu}",
  poisson: "\\text{Poisson}",
  gamma: "\\text{Gamma}",
  bernoulli: "\\text{Bernoulli}",
  negative_binomial: "\\text{NegBin}",
  beta: "\\text{Beta}",
  ordered_logistic: "\\text{OrdLogistic}",
  categorical: "\\text{Categorical}",
};

export function distName(dist: string): string {
  return DIST_LATEX[dist as DistributionFamily] ?? `\\text{${dist}}`;
}

/** Build a single observation-model line, inlining the latent construct when known. */
export function likelihoodLine(
  lik: LabeledLikelihood,
  constructName?: string,
  options?: { includeMeasurementError?: boolean },
): string {
  const v = `\\text{${textify(lik.label)}}`;
  const predictor = constructName
    ? `\\lambda_{${v}} \\, \\eta_{\\text{${textify(constructName)}}}(t)`
    : `\\mu_{${v}}`;
  const mu = linkInverse(lik.link, predictor);
  const d = distName(lik.distribution);
  const measurementErrorVariance = `\\sigma_{${v}}^{2}`;
  const includeMeasurementError = options?.includeMeasurementError ?? false;
  const withMeasurementError = (args: string) =>
    includeMeasurementError ? `${args},\\; ${measurementErrorVariance}` : args;

  if (lik.distribution === "gaussian" || lik.distribution === "student_t") {
    return `y_{${v}}(t) &\\sim ${d}(${mu},\\; ${measurementErrorVariance})`;
  }
  if (lik.distribution === "beta") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(
      `${mu}\\,\\phi,\\; (1 - ${mu})\\,\\phi`,
    )})`;
  }
  if (lik.distribution === "gamma") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(`\\kappa,\\; ${mu}/\\kappa`)})`;
  }
  if (lik.distribution === "negative_binomial") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(`r,\\; ${mu}`)})`;
  }
  return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(mu)})`;
}

/** Parse a parameter name into Greek letter + subscript. */
export function paramSymbol(name: string): string {
  if (name.startsWith("t0_mean_")) {
    const state = name.slice("t0_mean_".length);
    return `\\mu_{0,\\,\\text{${textify(state)}}}`;
  }
  if (name.startsWith("t0_sd_")) {
    const state = name.slice("t0_sd_".length);
    return `\\sigma_{0,\\,\\text{${textify(state)}}}`;
  }

  const greekMap: Record<string, string> = {
    beta: "\\beta",
    rho: "\\rho",
    sigma: "\\sigma",
    lambda: "\\lambda",
    tau: "\\tau",
    cor: "\\psi",
  };

  const parts = name.split("_");
  const greek = greekMap[parts[0]];
  if (greek && parts.length > 1) {
    return `${greek}_{\\text{${parts.slice(1).join(" ")}}}`;
  }
  return `\\text{${textify(name)}}`;
}

/** Strip the & alignment marker from a priorLine result for inline display. */
export function priorLatex(prior: PriorProposal, parameterName: string): string {
  return priorLine(prior, parameterName).replace(/&/g, "");
}

const PRIOR_DIST_LATEX: Record<PriorDistributionFamily, string> = {
  Normal: "\\mathcal{N}",
  HalfNormal: "\\text{HalfNormal}",
  Beta: "\\text{Beta}",
  Gamma: "\\text{Gamma}",
  Uniform: "\\text{Uniform}",
  TruncatedNormal: "\\text{TruncNormal}",
  Exponential: "\\text{Exp}",
  LogNormal: "\\text{LogNormal}",
  Delta: "\\Delta",
};

function priorDistributionLatex(prior: PriorProposal): string {
  const vals = Object.values(prior.params).map((v) => String(v));
  const d =
    PRIOR_DIST_LATEX[prior.distribution as PriorDistributionFamily] ??
    `\\text{${prior.distribution}}`;
  return `${d}(${vals.join(",\\; ")})`;
}

/** Map a prior distribution name + params to LaTeX. */
export function priorLine(prior: PriorProposal, parameterName: string): string {
  return `${paramSymbol(parameterName)} &\\sim ${priorDistributionLatex(prior)}`;
}

export function observationParameterSymbol({
  parameterName,
  likelihood,
}: {
  parameterName: string;
  likelihood: LabeledLikelihood;
}): string {
  const variableText = `\\text{${textify(likelihood.label)}}`;

  if (parameterName.startsWith(`lambda_${likelihood.label}_`)) {
    return `\\lambda_{${variableText}}`;
  }
  if (parameterName === `obs_sd_${likelihood.label}`) {
    return `\\sigma_{${variableText}}`;
  }
  if (parameterName === "obs_df") {
    return "\\nu";
  }
  if (parameterName === "obs_shape") {
    return "\\kappa";
  }
  if (parameterName === "obs_r") {
    return "r";
  }
  if (parameterName === "obs_concentration") {
    return "\\phi";
  }
  if (parameterName === "obs_ordered_base") {
    return `\\boldsymbol{\\tau}_{${variableText}}`;
  }
  if (parameterName === "obs_ordered_gaps") {
    return `\\Delta\\boldsymbol{\\tau}_{${variableText}}`;
  }
  if (parameterName === "obs_cat_intercepts") {
    return `\\boldsymbol{\\alpha}_{${variableText}}`;
  }
  if (parameterName === "obs_cat_slopes") {
    return `\\boldsymbol{\\beta}_{${variableText}}`;
  }

  return paramSymbol(parameterName);
}

export function observationPriorLatex({
  parameterName,
  prior,
  likelihood,
}: {
  parameterName: string;
  prior: PriorProposal;
  likelihood: LabeledLikelihood;
}): string {
  return `${observationParameterSymbol({ parameterName, likelihood })} \\sim ${priorDistributionLatex(prior)}`;
}

function observationParameterShownInLikelihood({
  parameterName,
  likelihood,
  hasConstruct,
}: {
  parameterName: string;
  likelihood: LabeledLikelihood;
  hasConstruct: boolean;
}): boolean {
  if (parameterName.startsWith(`lambda_${likelihood.label}_`)) {
    return hasConstruct;
  }
  if (parameterName === `obs_sd_${likelihood.label}`) {
    return likelihood.distribution === "gaussian" || likelihood.distribution === "student_t";
  }
  if (parameterName === "obs_df") {
    return likelihood.distribution === "student_t";
  }
  if (parameterName === "obs_shape") {
    return likelihood.distribution === "gamma";
  }
  if (parameterName === "obs_r") {
    return likelihood.distribution === "negative_binomial";
  }
  if (parameterName === "obs_concentration") {
    return likelihood.distribution === "beta";
  }
  return false;
}

export function observationParameterDefinitionLatex({
  parameterName,
  likelihood,
}: {
  parameterName: string;
  likelihood: LabeledLikelihood;
}): string {
  const symbol = observationParameterSymbol({ parameterName, likelihood });

  if (parameterName === `obs_sd_${likelihood.label}`) {
    return `${symbol} &: \\text{measurement-error SD}`;
  }
  if (parameterName === "obs_df") {
    return `${symbol} &: \\text{Student-t degrees of freedom}`;
  }
  if (parameterName === "obs_shape") {
    return `${symbol} &: \\text{Gamma shape}`;
  }
  if (parameterName === "obs_r") {
    return `${symbol} &: \\text{negative-binomial dispersion}`;
  }
  if (parameterName === "obs_concentration") {
    return `${symbol} &: \\text{Beta concentration}`;
  }
  if (parameterName === "obs_ordered_base") {
    return `${symbol} &: \\text{ordered thresholds}`;
  }
  if (parameterName === "obs_ordered_gaps") {
    return `${symbol} &: \\text{threshold gaps}`;
  }
  if (parameterName === "obs_cat_intercepts") {
    return `${symbol} &: \\text{categorical intercepts}`;
  }
  if (parameterName === "obs_cat_slopes") {
    return `${symbol} &: \\text{categorical slopes}`;
  }

  return `${symbol}`;
}

export function observationEquationLatex({
  likelihood,
  constructName,
  parameterNames,
}: {
  likelihood: LabeledLikelihood;
  constructName?: string;
  parameterNames?: string[];
}): string {
  const measurementErrorParameterName = `obs_sd_${likelihood.label}`;
  const hasMeasurementError = (parameterNames ?? []).includes(measurementErrorParameterName);
  const mainLine = likelihoodLine(likelihood, constructName, {
    includeMeasurementError:
      hasMeasurementError &&
      likelihood.distribution !== "gaussian" &&
      likelihood.distribution !== "student_t",
  });
  const supplementalLines = (parameterNames ?? [])
    .filter(
      (parameterName) =>
        !(parameterName === measurementErrorParameterName && hasMeasurementError) &&
        !observationParameterShownInLikelihood({
          parameterName,
          likelihood,
          hasConstruct: !!constructName,
        }),
    )
    .map((parameterName) =>
      observationParameterDefinitionLatex({
        parameterName,
        likelihood,
      }),
    );

  if (supplementalLines.length === 0) {
    return mainLine;
  }

  return `\\begin{aligned}${[mainLine, ...supplementalLines].join(" \\\\ ")}\\end{aligned}`;
}

export interface ConfounderGroup {
  confounder: string;
  states: string[];
  pairs: { s1: string; s2: string }[];
}

export function confounderGroups(plan: StructuralPlan): ConfounderGroup[] {
  const groups = new Map<string, { states: Set<string>; pairs: { s1: string; s2: string }[] }>();
  for (const dependency of plan.induced_dependencies) {
    if (dependency.kind !== "innovation_correlation") continue;
    const [s1, s2] = dependency.between.map((id) => plan.semantics.constructs[id].name);
    for (const id of dependency.source_confounder_ids) {
      const group = groups.get(id) ?? { states: new Set<string>(), pairs: [] };
      group.states.add(s1);
      group.states.add(s2);
      group.pairs.push({ s1, s2 });
      groups.set(id, group);
    }
  }
  return Array.from(groups, ([id, group]) => ({
    confounder: plan.semantics.constructs[id].name,
    states: [...group.states],
    pairs: group.pairs,
  }));
}

/** Render LaTeX for a single confounder group (raw LaTeX, no KaTeX rendering). */
export function confounderGroupLatex(group: ConfounderGroup): string {
  const confTex = `\\text{${textify(group.confounder)}}`;
  const stateList = group.states.map((s) => `\\text{${textify(s)}}`).join(",\\, ");

  const lines: string[] = [];
  lines.push(`U_{${confTex}} &\\to \\{${stateList}\\}`);
  const epsilons = group.states.map((s) => `\\varepsilon_{\\text{${textify(s)}}}`).join(",\\, ");
  lines.push(`(${epsilons}) &\\sim \\mathcal{N}(\\mathbf{0},\\, \\Psi_{${confTex}})`);
  for (const { s1, s2 } of group.pairs) {
    const t1 = `\\text{${textify(s1)}}`;
    const t2 = `\\text{${textify(s2)}}`;
    lines.push(`\\psi_{${t1},\\,${t2}} &\\neq 0`);
  }

  return `\\begin{aligned}\n${lines.join(" \\\\\n")}\n\\end{aligned}`;
}
