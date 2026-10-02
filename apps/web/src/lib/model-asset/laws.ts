import type {
  CausalEdgeSpec,
  ConstructSpec,
  DensityPoint,
  IndicatorSpec,
  ModelSnapshot,
  ObservationLawSpec,
  Expression,
  ParameterSpec,
  PosteriorMarginal,
} from "@nof1-causal-lab/api-types";
import { type CoefficientUse, coefficientUses } from "@/lib/model-accessors";
import { assertNever } from "@/lib/assert-never";
import { humanize } from "./selection";

/** One parameter's recorded laws at the viewed version, arranged on a single axis. */
export interface LawCurve {
  use: CoefficientUse;
  parameter: ParameterSpec;
  /**
   * `fitted`: the fit's conditioned input law beside its posteriors, on the quantity scale.
   * `authored`: the law as written, on its authoring scale.
   */
  kind: "fitted" | "authored";
  prior: readonly DensityPoint[];
  posteriors: readonly PosteriorMarginal[];
  /** The fit was conditioned on another panel than the one selected at this version. */
  stale: boolean;
  /** The authored law's family; a fitted law is known by its posteriors. */
  family: string | null;
}

/** Laws named by an entity's own terms: construct dynamics and noise, edge mechanisms, or measurement. */
export function ownLawUses(entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec) {
  if ("dynamics" in entity)
    return coefficientUses([...entity.dynamics.map((d) => d.expression), ...entity.coefficients]);
  if ("mechanisms" in entity) return coefficientUses(entity.mechanisms.map((m) => m.expression));
  return coefficientUses(entity.likelihood ? observationOperands(entity.likelihood.law) : []);
}

/** Traverse the typed operands; probability evaluation belongs to the pipeline. */
function observationOperands(law: ObservationLawSpec): readonly Expression[] {
  switch (law.distribution) {
    case "Delta":
      return [law.v];
    case "Normal":
      return [law.loc, law.scale];
    case "StudentT":
      return [law.df, law.loc, law.scale];
    case "Poisson":
      return [law.rate];
    case "Gamma":
      return [law.concentration, law.rate];
    case "BernoulliLogits":
      return [law.logits];
    case "BernoulliProbs":
      return [law.probs];
    case "NegativeBinomial2":
      return [law.mean, law.concentration];
    case "Beta":
      return [law.concentration1, law.concentration0];
    case "OrderedLogistic":
      return [law.predictor, law.cutpoints];
    case "Categorical":
      return [law.logits];
    default:
      return assertNever(law);
  }
}

/** A fit's posteriors supersede the laws authored at its version; unplotted laws are omitted. */
export function lawCurves(model: ModelSnapshot, uses: readonly CoefficientUse[]): LawCurve[] {
  const parameters = new Map(
    (model.model?.value.parameters ?? []).map((parameter) => [parameter.id, parameter]),
  );
  const fit = model.findings.fit;
  return uses.flatMap((use): LawCurve[] => {
    const parameter = parameters.get(use.parameterId);
    if (!parameter) return [];
    const posteriors = (fit?.value.report.posterior_marginals ?? []).filter(
      (marginal) => marginal.subject.parameter_id === use.parameterId,
    );
    if (fit && posteriors.length > 0)
      return [
        {
          use,
          parameter,
          kind: "fitted",
          prior: fit.value.prior_densities[use.parameterId] ?? [],
          posteriors,
          stale: fit.source.validity === "stale",
          family: null,
        },
      ];
    const prior = model.findings.diagnostics?.prior_densities[use.parameterId] ?? [];
    const law = parameter.distribution
      ? model.model?.value.distributions[parameter.distribution]
      : null;
    return prior.length > 0
      ? [
          {
            use,
            parameter,
            kind: "authored",
            prior,
            posteriors: [],
            stale: false,
            family: law?.distribution ?? null,
          },
        ]
      : [];
  });
}

/** Authored persistence and interval-effect laws describe another quantity than their role. */
export function lawLabel(curve: LawCurve): string {
  if (curve.kind === "authored") {
    if (curve.parameter.transform.kind === "dt_persistence_to_ct_decay") return "persistence";
    if (curve.parameter.transform.kind === "dt_effect_to_ct_rate") return "interval effect";
  }
  return humanize(curve.use.role);
}
