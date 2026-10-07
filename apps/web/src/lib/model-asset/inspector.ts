import type {
  ConstructId,
  ConstructSpec,
  CausalEdgeSpec,
  IndicatorId,
  IndicatorSpec,
  ModelSnapshot,
} from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "./scope";
import { recordedEntityFailures } from "./entity-findings";

/** These selectors only arrange recorded findings for their owning entities. */
export function constructPresentation(context: ScopeContext, id: ConstructId) {
  const { model, entities } = context;
  const construct = entities.constructById.get(id);
  if (!construct) return null;
  const indicators = construct.indicators;

  return {
    model,
    entities,
    construct,
    indicators,
  };
}

export function indicatorPresentation(context: ScopeContext, id: IndicatorId) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
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
  const likelihood = indicator.likelihood;
  const issues = audit?.issues.filter((issue) => issue.severity !== "info") ?? [];
  return {
    indicator,
    audit,
    likelihood,
    issues,
  };
}

/** Aggregate recorded failures at their visible owner; never judge diagnostic numbers here. */
export function entityFailures(
  model: ModelSnapshot,
  entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec,
): string[] {
  const identity = "observation" in entity ? entity.observation.id : entity.id;
  return [
    ...(recordedEntityFailures(model)[identity] ?? []),
    ...("dynamics" in entity &&
    model.identification?.treatments[entity.id]?.status === "not_identified"
      ? [`Identification against ★: ${entity.name}`]
      : []),
  ];
}
