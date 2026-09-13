import type { JsonValue, NumPyroDistribution } from "@nof1-causal-lab/api-types";

/** Render native constructor arguments; probability calculations stay in Python. */
export function distributionArgumentText(value: JsonValue): string {
  if (Array.isArray(value)) return `[${value.map(distributionArgumentText).join(", ")}]`;
  if (value === null || typeof value !== "object") return String(value);
  if ("array" in value) return distributionArgumentText(value.array);
  if ("tuple" in value) return distributionArgumentText(value.tuple);
  if ("float" in value) return String(value.float);
  const constructor = value.distribution ?? value.transform ?? value.constraint;
  const params = value.params;
  if (
    typeof constructor === "string" &&
    params &&
    typeof params === "object" &&
    !Array.isArray(params)
  ) {
    return `${constructor}(${Object.entries(params)
      .map(([name, argument]) => `${name}=${distributionArgumentText(argument)}`)
      .join(", ")})`;
  }
  return `{${Object.entries(value)
    .map(([name, argument]) => `${name}: ${distributionArgumentText(argument)}`)
    .join(", ")}}`;
}

export function distributionText(prior: NumPyroDistribution): string {
  return `${prior.distribution}(${Object.entries(prior.params)
    .map(([name, value]) => `${name}=${distributionArgumentText(value)}`)
    .join(", ")})`;
}
