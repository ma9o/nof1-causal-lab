import { presentEntries } from "@/lib/model-accessors";
import type { JsonArray, JsonValue, NumPyroDistribution } from "@nof1-causal-lab/api-types";

const isArray: (value: JsonValue) => value is JsonArray = Array.isArray;

/** Render native constructor arguments; probability calculations stay in Python. */
export function distributionArgumentText(value: JsonValue): string {
  if (typeof value === "number") return String(Number(value.toPrecision(4)));
  if (isArray(value)) return `[${value.map(distributionArgumentText).join(", ")}]`;
  if (value === null || typeof value !== "object") return String(value);
  if (value.array !== undefined) return distributionArgumentText(value.array);
  if (value.tuple !== undefined) return distributionArgumentText(value.tuple);
  if ("float" in value) return String(value.float);
  const constructorName = value.distribution ?? value.transform ?? value.constraint;
  const params = value.params;
  if (
    typeof constructorName === "string" &&
    params &&
    typeof params === "object" &&
    !isArray(params)
  ) {
    return `${constructorName}(${presentEntries(params)
      .filter(([name]) => name !== "validate_args")
      .map(([name, argument]) => `${name}=${distributionArgumentText(argument)}`)
      .join(", ")})`;
  }
  return `{${presentEntries(value)
    .map(([name, argument]) => `${name}: ${distributionArgumentText(argument)}`)
    .join(", ")}}`;
}

export function distributionText(prior: NumPyroDistribution): string {
  return `${prior.distribution}(${presentEntries(prior.params)
    .filter(([name]) => name !== "validate_args")
    .map(([name, value]) => `${name}=${distributionArgumentText(value)}`)
    .join(", ")})`;
}
