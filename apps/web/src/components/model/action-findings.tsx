import type { ScopeContext } from "@/lib/model-asset/scope";
import type { JournalTick } from "@/lib/model-asset/journal";
import { resolveEntity } from "@/lib/model-asset/entities";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";

/** Only this action's new checks and produced findings belong in its record. */
export function ActionFindings({ context, tick }: { context: ScopeContext; tick: JournalTick }) {
  const { model, entities, select } = context;
  const produced = new Set(tick.produced.map((artifact) => artifact.artifact_id));
  const reused = tick.checks?.reused ?? [];
  const identification =
    produced.has("identification_report") && !reused.includes("identification")
      ? model.findings.identification?.value
      : null;
  const validation =
    produced.has("validation_report") && !reused.includes("compatibility")
      ? model.findings.validation_report?.value
      : null;
  const data = validation ?? (produced.has("data_profile") ? model.data.profile?.value : null);
  const predictive = !reused.includes("predictive") ? tick.checks?.predictive : null;
  const findings: Array<{
    label: string;
    reason: string;
    status: "failed" | "warning" | "not_evaluated";
    owner?: EntitySelection;
  }> = [];
  for (const finding of [
    ...(!reused.includes("specification") ? (tick.checks?.specification.findings ?? []) : []),
    ...(validation?.preflight.findings ?? []),
  ]) {
    if (finding.status !== "passed")
      findings.push({
        label: humanize(finding.check),
        reason: finding.message,
        status: finding.status,
      });
  }
  for (const issue of [
    ...(data?.dataset_issues ?? []),
    ...Object.entries(data?.indicators ?? {}).flatMap(([id, audit]) =>
      audit.issues.map((issue) => ({ ...issue, indicator_id: id })),
    ),
  ]) {
    if (issue.severity === "info") continue;
    const indicator = entities.indicators.find((item) => item.id === issue.indicator_id);
    const variable = model.data.metadata?.value.variables.find(
      (item) => item.id === issue.indicator_id,
    );
    findings.push({
      label: humanize(indicator?.name ?? variable?.name ?? "Dataset"),
      reason: issue.message,
      status: issue.severity === "error" ? "failed" : "warning",
      owner: indicator ? { kind: "indicator", id: indicator.id } : undefined,
    });
  }
  for (const construct of entities.constructs) {
    const finding = identification?.treatments[construct.id];
    if (finding?.status === "not_identified")
      findings.push({
        label: humanize(construct.name),
        reason: [
          "Not identified.",
          finding.confounders.length
            ? `Confounded by ${finding.confounders.map((id) => humanize(entities.constructById.get(id)!.name)).join(", ")}.`
            : "",
          finding.notes,
        ]
          .filter(Boolean)
          .join(" "),
        status: "failed",
        owner: { kind: "construct", id: construct.id },
      });
  }
  for (const finding of predictive?.findings ?? []) {
    if (finding.passed !== false) continue;
    const indicator = entities.indicators.find((item) => item.id === finding.target);
    const edge = entities.edges.find((item) => item.id === finding.target);
    const construct = entities.constructs.find((item) => item.id === finding.construct_id);
    const owner: EntitySelection | undefined = indicator
      ? { kind: "indicator", id: indicator.id }
      : edge
        ? { kind: "edge", id: edge.id }
        : construct
          ? { kind: "construct", id: construct.id }
          : undefined;
    const entity = owner && resolveEntity(entities, owner);
    findings.push({
      label: `${humanize(finding.check)}${entity ? ` · ${entity.label}` : ""}`,
      reason: finding.note,
      status: "failed",
      owner,
    });
  }
  for (const check of predictive?.predictive_checks?.per_variable_warnings ?? []) {
    if (check.passed) continue;
    const indicator = entities.indicatorById.get(check.indicator_id)!;
    findings.push({
      label: `${humanize(check.check_type)} · ${humanize(indicator.name)}`,
      reason: check.message,
      status: "failed",
      owner: { kind: "indicator", id: indicator.id },
    });
  }
  if (predictive?.status === "not_evaluated")
    findings.push({
      label: "Predictive checks",
      reason: predictive.detail ?? humanize(predictive.reason!).toLowerCase(),
      status: "not_evaluated",
    });
  if (tick.messages.some((message) => message.label === "EXTRACTION_PARTIAL"))
    findings.push({
      label: "Extraction incomplete",
      reason: "Some extraction workers failed. The prepared data is incomplete.",
      status: "warning",
    });
  return (
    <>
      {identification && Object.keys(identification.treatments).length > 0 && (
        <Section title="Identification">
          <Hint>
            {
              Object.values(identification.treatments).filter(
                (finding) => finding.status === "identified",
              ).length
            }{" "}
            identified;{" "}
            {
              Object.values(identification.treatments).filter(
                (finding) => finding.status === "not_identified",
              ).length
            }{" "}
            not identified.
          </Hint>
        </Section>
      )}
      {findings.length > 0 && (
        <Section title="Problems">
          {findings.map((finding, index) => (
            <div key={index} className="flex items-start gap-2">
              <StatusIcon status={finding.status} />
              <p className="min-w-0">
                {finding.owner ? (
                  <OwnerLink onClick={() => select(finding.owner!)}>{finding.label}</OwnerLink>
                ) : (
                  finding.label
                )}
                : {finding.reason}
              </p>
            </div>
          ))}
        </Section>
      )}
    </>
  );
}
