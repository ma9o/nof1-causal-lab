import type { ScopeContext } from "@/lib/model-asset/scope";
import type { JournalTick } from "@/lib/model-asset/journal";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";
import { PredictiveFindings } from "./simulation-evidence";
import { SpecificationFindings } from "./specification-findings";

/** Only this action's new checks and produced findings belong in its record. */
export function ActionFindings({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const { model, entities, select } = context;
  const produced = new Set(tick.produced.map((artifact) => artifact.artifact_id));
  const reused = tick.checks?.reused ?? [];
  const identification =
    produced.has("identification_report") && !reused.includes("identification")
      ? model.findings.identification
      : null;
  const validation =
    produced.has("validation_report") && !reused.includes("compatibility")
      ? model.findings.validation_report
      : null;
  const data = validation ?? (produced.has("data_profile") ? model.data.profile : null);
  const predictive = !reused.includes("predictive") ? tick.checks?.predictive : null;
  const targets = new Set([
    ...(predictive?.findings.map((finding) =>
      entities.indicators.some((indicator) => indicator.id === finding.target)
        ? finding.target
        : finding.construct_id,
    ) ?? []),
    ...(predictive?.predictive_checks?.per_variable_warnings.map(
      (finding) => finding.indicator_id,
    ) ?? []),
  ]);
  const indicators = entities.indicators.filter((indicator) => targets.has(indicator.id));
  const constructs = entities.constructs.filter((construct) => targets.has(construct.id));
  const dataIssues =
    data &&
    (data.value.dataset_issues.length > 0 ||
      Object.values(data.value.indicators).some((audit) => audit.issues.length > 0));
  return (
    <>
      {tick.extractionPartial && (
        <Section title="Extraction incomplete">
          <Hint issue>Some extraction workers failed. The prepared data is incomplete.</Hint>
        </Section>
      )}
      {tick.checks &&
        tick.checks.specification.findings.length > 0 &&
        !reused.includes("specification") && (
          <Section title="Specification">
            <SpecificationFindings report={tick.checks.specification} />
          </Section>
        )}
      {identification && Object.keys(identification.value.treatments).length > 0 && (
        <Section title="Identification" source={identification.source}>
          {entities.constructs
            .filter((construct) => identification.value.treatments[construct.id])
            .map((construct) => (
              <OwnerLink
                key={construct.id}
                onClick={() => select({ kind: "construct", id: construct.id })}
              >
                {humanize(construct.name)} ·{" "}
                {humanize(identification.value.treatments[construct.id].status)}
              </OwnerLink>
            ))}
        </Section>
      )}
      {dataIssues && (
        <Section title="Data findings" source={data.source}>
          {data.value.dataset_issues.map((issue) => (
            <div key={issue.issue_type + issue.message} className="flex gap-2">
              {issue.severity !== "info" && (
                <StatusIcon status={issue.severity === "error" ? "failed" : "warning"} />
              )}
              <Hint issue={issue.severity !== "info"}>{issue.message}</Hint>
            </div>
          ))}
          {Object.entries(data.value.indicators)
            .filter(([, audit]) => audit.issues.length > 0)
            .map(([id, audit]) => {
              const indicator = entities.indicators.find((item) => item.id === id);
              return indicator ? (
                <OwnerLink key={id} onClick={() => select({ kind: "indicator", id: indicator.id })}>
                  {humanize(indicator.name)}
                </OwnerLink>
              ) : (
                <Hint key={id} issue>
                  {audit.issues.map((issue) => issue.message).join(" ")}
                </Hint>
              );
            })}
        </Section>
      )}
      {validation && validation.value.preflight.findings.length > 0 && (
        <Section title="Compatibility" source={validation.source}>
          <SpecificationFindings report={validation.value.preflight} />
        </Section>
      )}
      {predictive && (
        <Section title="Predictive checks">
          {predictive.status === "not_evaluated" && (
            <Hint>{predictive.detail ?? humanize(predictive.reason ?? "Not evaluated")}</Hint>
          )}
          <PredictiveFindings
            entities={entities}
            findings={predictive.findings.filter(
              (finding) =>
                !finding.construct_id &&
                !entities.indicators.some((indicator) => indicator.id === finding.target),
            )}
          />
          {indicators.map((indicator) => (
            <OwnerLink
              key={indicator.id}
              onClick={() => select({ kind: "indicator", id: indicator.id })}
            >
              {humanize(indicator.name)}
            </OwnerLink>
          ))}
          {constructs.map((construct) => (
            <OwnerLink
              key={construct.id}
              onClick={() => select({ kind: "construct", id: construct.id })}
            >
              {humanize(construct.name)}
            </OwnerLink>
          ))}
        </Section>
      )}
    </>
  );
}
