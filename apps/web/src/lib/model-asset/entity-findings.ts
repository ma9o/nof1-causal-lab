import type {
  ConstructId,
  EdgeId,
  IndicatorId,
  ModelSnapshot,
  ParameterId,
} from "@nof1-causal-lab/api-types";
import { indexModel } from "./entities";
import { ownLawUses } from "./laws";

const messages = new WeakMap<
  ModelSnapshot,
  Readonly<Partial<Record<ConstructId | EdgeId | IndicatorId, readonly string[]>>>
>();

/** Format retained verdicts by the model's declared coefficient references. */
export function recordedEntityFailures(
  model: ModelSnapshot,
): Readonly<Partial<Record<ConstructId | EdgeId | IndicatorId, readonly string[]>>> {
  const cached = messages.get(model);
  if (cached) return cached;
  const fit = model.fit;
  const entities = indexModel(model.model);
  const convergence = (parameters: readonly ParameterId[]) =>
    (fit?.convergence.assessments ?? []).flatMap((finding) =>
      finding.kind === "evaluated" &&
      finding.outcome === "failed" &&
      typeof finding.subject !== "string" &&
      parameters.includes(finding.subject.parameter.parameter_id)
        ? [`Parameter convergence: ${finding.subject.label}`]
        : [],
    );
  const indicators = Object.fromEntries(
    entities.indicators.map((indicator) => {
      const issues = [
        ...(model.profile?.indicators[indicator.observation.id]?.issues ?? []),
        ...(model.validation_report?.data.indicators[indicator.observation.id]?.issues ?? []),
      ];
      return [
        indicator.observation.id,
        [
          ...convergence(ownLawUses(indicator).map((use) => use.parameterId)),
          ...(issues.some((issue) => issue.severity !== "info")
            ? [`Data quality: ${indicator.observation.name}`]
            : []),
        ],
      ];
    }),
  );
  const result = {
    ...indicators,
    ...Object.fromEntries(
      entities.edges.map((edge) => [
        edge.id,
        [...new Set(convergence(ownLawUses(edge).map((use) => use.parameterId)))],
      ]),
    ),
    ...Object.fromEntries(
      entities.constructs.map((construct) => {
        const uses = [
          ...ownLawUses(construct),
          ...entities.edges
            .filter((edge) => edge.cause.id === construct.id || edge.effect.id === construct.id)
            .flatMap(ownLawUses),
        ];
        return [
          construct.id,
          [
            ...new Set([
              ...convergence(uses.map((use) => use.parameterId)),
              ...construct.indicators.flatMap(
                (indicator) => indicators[indicator.observation.id] ?? [],
              ),
            ]),
          ],
        ];
      }),
    ),
  };
  messages.set(model, result);
  return result;
}
