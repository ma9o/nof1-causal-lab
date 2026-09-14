import { distributionArgumentText } from "./distribution-format";
import type { NumPyroDistribution } from "@nof1-causal-lab/api-types";

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
export function textify(name: string): string {
  return name.replace(/_/g, " ");
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
export function priorLatex(prior: NumPyroDistribution, parameterName: string): string {
  return priorLine(prior, parameterName).replace(/&/g, "");
}

const PRIOR_DIST_LATEX: Record<string, string> = {
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

function priorDistributionLatex(prior: NumPyroDistribution): string {
  const vals = Object.values(prior.params)
    .map(distributionArgumentText)
    .map((value) => value.replaceAll("_", "\\_"));
  const d = PRIOR_DIST_LATEX[prior.distribution as string] ?? `\\text{${prior.distribution}}`;
  return `${d}(${vals.join(",\\; ")})`;
}

/** Map a prior distribution name + params to LaTeX. */
export function priorLine(prior: NumPyroDistribution, parameterName: string): string {
  return `${paramSymbol(parameterName)} &\\sim ${priorDistributionLatex(prior)}`;
}

/** Use the same scientific parameter labels as the server-rendered conditional law. */
export function observationParameterSymbol({ parameterName }: { parameterName: string }): string {
  return `\\theta_{\\text{${textify(parameterName)}}}`;
}

export function observationPriorLatex({
  parameterName,
  prior,
}: {
  parameterName: string;
  prior: NumPyroDistribution;
}): string {
  return `${observationParameterSymbol({ parameterName })} \\sim ${priorDistributionLatex(prior)}`;
}
