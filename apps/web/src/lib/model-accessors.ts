import type {
  CoefficientRole,
  ConstructSpec,
  Expression,
  ModelSpec,
  ParameterId,
} from "@nof1-causal-lab/api-types";
import { assertNever } from "./assert-never";

/** Enumerate present entries without losing the trusted API's named key type. */
export function presentEntries<Key extends string, Value>(
  record: Readonly<Partial<Record<Key, Value | undefined>>>,
): [Key, Value][] {
  return Object.entries<Value | undefined>(record).flatMap(([key, value]) => {
    if (value === undefined) return [];
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion -- Object.entries erases the key type of this trusted API-owned sparse map.
    return [[key as Key, value] satisfies [Key, Value]];
  });
}

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
export function coefficientUses(expressions: readonly Expression[]): CoefficientUse[] {
  const uses = new Map<ParameterId, CoefficientUse>();
  const visit = (operand: Expression): void => {
    switch (operand.kind) {
      case "coefficient":
        if (typeof operand.value === "string" && !uses.has(operand.value))
          uses.set(operand.value, { role: operand.role, parameterId: operand.value });
        return;
      case "binary":
        visit(operand.left);
        visit(operand.right);
        return;
      case "call":
        operand.arguments.forEach(visit);
        return;
      case "literal":
      case "state":
        return;
      default:
        assertNever(operand);
    }
  };
  expressions.forEach(visit);
  return [...uses.values()];
}
