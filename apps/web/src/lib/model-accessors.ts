import type {
  CoefficientExpression,
  CoefficientRole,
  ConstructSpec,
  ModelSpec,
  ParameterId,
} from "@nof1-causal-lab/api-types";

/** Follow the serialized graph's endpoint definitions; references carry identity only. */
export function modelConstructs(model: ModelSpec | null | undefined): ConstructSpec[] {
  return (model?.edges ?? []).flatMap((edge) =>
    [edge.cause, edge.effect].filter((endpoint): endpoint is ConstructSpec => "name" in endpoint),
  );
}

/** A coefficient operand that names a parameter, with the quantity its authored role declares. */
export interface CoefficientUse {
  role: CoefficientRole;
  parameterId: ParameterId;
}

/** Follow serialized coefficient references in declaration order, keeping each parameter's first role. */
export function coefficientUses(component: unknown): CoefficientUse[] {
  const uses = new Map<ParameterId, CoefficientUse>();
  const visit = (value: unknown) => {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    const record = value as Record<string, unknown>;
    if (record.kind === "coefficient") {
      const operand = record as unknown as CoefficientExpression;
      if (typeof operand.value === "string" && !uses.has(operand.value))
        uses.set(operand.value, { role: operand.role, parameterId: operand.value });
      return;
    }
    Object.values(record).forEach(visit);
  };
  visit(component);
  return [...uses.values()];
}
