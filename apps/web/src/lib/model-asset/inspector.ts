import type {
  ConstructId,
  ConstructSpec,
  CausalEdgeSpec,
  IndicatorId,
  IndicatorSpec,
  ModelSnapshot,
  StructuralDisposition,
} from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "./scope";

const DISPOSITION_LABEL: Record<StructuralDisposition, string> = {
  unsupported: "unsupported",
  retained_state: "retained state",
  marginalized: "marginalized",
  identification_only: "identification-only",
  retained_edge: "retained edge",
  projected_edge: "projected edge",
  manifest: "manifest",
  excluded_indicator: "excluded indicator",
};

export function dispositionLabel(disposition: StructuralDisposition): string {
  return DISPOSITION_LABEL[disposition];
}

/** These selectors only arrange recorded findings for their owning entities. */
export function constructPresentation(context: ScopeContext, id: ConstructId) {
  const { model, entities } = context;
  const construct = entities.constructById.get(id);
  if (!construct) return null;
  const indicators = construct.indicators;
  const disposition = context.model.dispositions?.value.find((item) => item.target.id === id);

  return {
    model,
    entities,
    construct,
    indicators,
    disposition,
  };
}

export function indicatorPresentation(context: ScopeContext, id: IndicatorId) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
  const disposition = context.model.dispositions?.value.find((item) => item.target.id === id);
  const data = context.model.validation_report?.value.data ?? context.model.profile?.value;
  const audit = data?.indicators[id];
  const counts = context.model.measurements?.value.per_indicator_counts[id];
  const likelihood = indicator.likelihood;
  const predictive = context.model.predictive;
  const checks =
    (predictive?.source.validity === "fresh" && predictive.value.evaluation.kind === "evaluated"
      ? predictive.value.evaluation.predictive_checks?.per_variable_warnings
      : []
    )?.filter((item) => item.subject.target.id === id) ?? [];
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

/** Aggregate recorded failures at their visible owner; never judge diagnostic numbers here. */
export function entityFailures(
  model: ModelSnapshot,
  entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec,
): string[] {
  return [
    ...(model.entity_failures["observation" in entity ? entity.observation.id : entity.id] ?? []),
  ];
}
