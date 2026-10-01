import type {
  ConstructId,
  ConstructSpec,
  CausalEdgeSpec,
  IndicatorId,
  IndicatorSpec,
  ModelSnapshot,
} from "@nof1-causal-lab/api-types";
import { ownLawUses } from "./laws";
import type { ScopeContext } from "./scope";
import { humanize } from "./selection";

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
  const indicators = construct.indicators;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.target.id === id,
  );
  const finding = model.findings.identification?.value.treatments[id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const namesFor = (ids: ConstructId[]) =>
    ids.map((id) => humanize(entities.constructById.get(id)!.name)).join(", ");

  return {
    model,
    entities,
    construct,
    indicators,
    disposition,
    identified,
    notIdentified,
    namesFor,
  };
}

export function indicatorPresentation(context: ScopeContext, id: IndicatorId) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.target.id === id,
  );
  const audit = (context.model.findings.validation_report ?? context.model.data.profile)?.value
    .indicators[id];
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

/** Aggregate recorded failures at their visible owner; never judge diagnostic numbers here. */
export function entityFailures(
  model: ModelSnapshot,
  entity: ConstructSpec | CausalEdgeSpec | IndicatorSpec,
): string[] {
  const failures: string[] = [];
  const label = "name" in entity ? humanize(entity.name) : entity.id;
  const { fit, predictive, identification, validation_report } = model.findings;
  const parameters = new Set(ownLawUses(entity).map((use) => use.parameterId));
  if (fit?.source.validity === "fresh") {
    for (const failure of fit.value.convergence.failures) {
      if (parameters.has(failure.subject.parameter_id))
        failures.push(`Parameter convergence: ${humanize(failure.parameter)}`);
    }
  }
  if (predictive?.source.validity === "fresh") {
    if (
      predictive.value.findings.some(
        (finding) =>
          finding.passed === false &&
          (finding.target === entity.id ||
            (finding.construct_id === entity.id &&
              !(
                "indicators" in entity &&
                entity.indicators.some((indicator) => indicator.id === finding.target)
              ))),
      ) ||
      predictive.value.predictive_checks?.per_variable_warnings.some(
        (check) => !check.passed && check.indicator_id === entity.id,
      )
    )
      failures.push(`Predictive checks: ${label}`);
  }
  const data = validation_report ?? model.data.profile;
  const audits: Partial<NonNullable<typeof data>["value"]["indicators"]> =
    data?.value.indicators ?? {};
  if (
    data?.source.validity === "fresh" &&
    audits[entity.id]?.issues.some((issue) => issue.severity !== "info")
  )
    failures.push(`Data quality: ${label}`);
  if ("indicators" in entity) {
    const treatments: Partial<NonNullable<typeof identification>["value"]["treatments"]> =
      identification?.value.treatments ?? {};
    if (
      identification?.source.validity === "fresh" &&
      treatments[entity.id]?.status === "not_identified"
    )
      failures.push(`Identification against ★: ${label}`);
    failures.push(...entity.indicators.flatMap((indicator) => entityFailures(model, indicator)));
  }
  return [...new Set(failures)];
}
