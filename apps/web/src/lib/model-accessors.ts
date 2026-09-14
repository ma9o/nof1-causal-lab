import type { ConstructSpec, ModelSpec } from "@nof1-causal-lab/api-types";

/** Follow the serialized graph's endpoint definitions; references carry identity only. */
export function modelConstructs(model: ModelSpec | null | undefined): ConstructSpec[] {
  return (model?.edges ?? []).flatMap((edge) =>
    [edge.cause, edge.effect].filter((endpoint): endpoint is ConstructSpec => "name" in endpoint),
  );
}

export function modelIndicators(model: ModelSpec | null | undefined) {
  return modelConstructs(model).flatMap((construct) => construct.indicators);
}

export function indicatorOwners(constructs: readonly ConstructSpec[]) {
  return new Map(
    constructs.flatMap((construct) =>
      construct.indicators.map((indicator) => [indicator.id, construct] as const),
    ),
  );
}

/** Follow serialized coefficient references for presentation; labels do not identify parameters. */
export function referencedParameterIds(component: unknown): Set<string> {
  const ids = new Set<string>();
  const visit = (value: unknown) => {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    const record = value as Record<string, unknown>;
    if (record.kind === "coefficient" && typeof record.value === "string") {
      ids.add(record.value);
      return;
    }
    Object.values(record).forEach(visit);
  };
  visit(component);
  return ids;
}
