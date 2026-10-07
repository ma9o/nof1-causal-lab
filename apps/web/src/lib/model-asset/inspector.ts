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
  const disposition = context.model.dispositions?.find((item) => item.target.id === id);

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
  const disposition = context.model.dispositions?.find((item) => item.target.id === id);
  const profile = context.model.profile?.indicators[id];
  const compatibility = context.model.validation_report?.data.indicators[id];
  const audit =
    profile || compatibility
      ? {
          profile: profile?.profile ?? null,
          checks: { ...profile?.checks, ...compatibility?.checks },
          issues: [...(profile?.issues ?? []), ...(compatibility?.issues ?? [])],
        }
      : null;
  const counts = context.model.measurements?.per_indicator_counts[id];
  const likelihood = indicator.likelihood;
  const issues = audit?.issues.filter((issue) => issue.severity !== "info") ?? [];
  return {
    indicator,
    disposition,
    audit,
    counts,
    likelihood,
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
