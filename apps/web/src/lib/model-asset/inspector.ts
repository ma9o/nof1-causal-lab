import type {
  ConstructId,
  IndicatorId,
  InferenceReport,
  ParameterSpec,
  PosteriorEstimate,
} from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "./scope";
import { humanize } from "./selection";

export interface PosteriorRow extends PosteriorEstimate {
  parameter: string;
}

export function posteriorRows(
  parameters: ParameterSpec[],
  posterior: InferenceReport | undefined,
): PosteriorRow[] {
  const marginals = posterior?.posterior_marginals ?? [];
  const ids = new Set(parameters.map((parameter) => parameter.id));
  return marginals.filter((marginal) => ids.has(marginal.subject.parameter_id));
}

const DISPOSITION_LABEL: Record<string, string> = {
  unsupported: "unsupported",
  retained_state: "retained state",
  marginalized: "marginalized",
  identification_only: "identification-only",
  retained_edge: "retained edge",
  projected_edge: "projected edge",
  manifest: "manifest",
  excluded_indicator: "excluded indicator",
};

export function dispositionLabel(disposition: string): string {
  return DISPOSITION_LABEL[disposition] ?? humanize(disposition);
}

/** These selectors only arrange recorded findings for their owning entities. */
export function constructPresentation(context: ScopeContext, id: ConstructId) {
  const { model, entities } = context;
  const construct = entities.constructById.get(id);
  if (!construct) return null;
  const inEdges = entities.edges.filter((edge) => edge.effect.id === id);
  const outEdges = entities.edges.filter((edge) => edge.cause.id === id);
  const indicators = construct.indicators;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.target.id === id,
  );
  const finding = model.findings.identification?.value.treatments[id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const admission =
    model.findings.prior_predictive?.value.diagnostics.filter((item) => item.construct_id === id) ??
    [];
  const namesFor = (ids: ConstructId[]) =>
    ids.map((id) => humanize(entities.constructById.get(id)!.name)).join(", ");

  return {
    model,
    entities,
    construct,
    inEdges,
    outEdges,
    indicators,
    disposition,
    identified,
    notIdentified,
    admission,
    namesFor,
  };
}

export function indicatorPresentation(context: ScopeContext, id: IndicatorId) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.target.id === id,
  );
  const audit = context.model.findings.validation_report?.value.indicators[id];
  const counts = context.model.data.measurements?.value.per_indicator_counts[id];
  const likelihood = indicator.likelihood;
  const predictive = context.model.findings.predictive;
  const checks =
    (predictive?.source.validity === "fresh"
      ? predictive.value.predictive_checks?.per_variable_warnings
      : []
    )?.filter((item) => item.indicator_id === id) ?? [];
  const issues = audit?.issues.filter((issue) => issue.severity !== "info") ?? [];
  return {
    indicator,
    disposition,
    audit,
    counts,
    likelihood,
    predictive,
    checks,
    issues,
  };
}
