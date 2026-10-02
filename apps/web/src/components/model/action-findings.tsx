import { presentEntries } from "@/lib/model-accessors";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { ActionAttempt } from "@nof1-causal-lab/api-types";
import { resolveEntity } from "@/lib/model-asset/entities";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";

/** Only this action's new checks and produced findings belong in its record. */
export function ActionFindings({
  context,
  result,
}: {
  context: ScopeContext;
  result: Extract<ActionAttempt["outcome"], { status: "applied" }>["result"];
}) {
  const { model, entities, select } = context;
  const produced = new Set(result.produced.map((artifact) => artifact.artifact_id));
  const reused = result.checks?.reused ?? [];
  const identification =
    produced.has("identification_report") && !reused.includes("identification")
      ? model.findings.identification?.value
      : null;
  const validation =
    produced.has("validation_report") && !reused.includes("compatibility")
      ? model.findings.validation_report?.value
      : null;
  const data = validation ?? (produced.has("data_profile") ? model.data.profile?.value : null);
  const predictive = !reused.includes("predictive") ? result.checks?.predictive : null;
  const findings: Array<{
    label: string;
    reason: string | null;
    status: "passed" | "failed" | "warning" | "error" | "not_evaluated";
    owner?: EntitySelection;
  }> = [];
  for (const finding of [
    ...(!reused.includes("specification") ? (result.checks?.specification.findings ?? []) : []),
    ...(validation?.preflight.findings ?? []),
  ]) {
    if (finding.kind !== "evaluated" || finding.outcome !== "passed")
      findings.push({
        label: humanize(finding.subject),
        reason: finding.kind === "evaluated" ? finding.evidence : finding.detail,
        status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      });
  }
  for (const issue of [
    ...(data?.dataset_issues ?? []),
    ...presentEntries(data?.indicators ?? {}).flatMap(([id, audit]) =>
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
      status: issue.severity,
      ...(indicator ? { owner: { kind: "indicator" as const, id: indicator.id } } : {}),
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
            ? `Confounded by ${entities.constructs
                .filter((entity) => finding.confounders.includes(entity.id))
                .map((entity) => humanize(entity.name))
                .join(", ")}.`
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
    if (finding.kind === "evaluated" && finding.outcome === "passed") continue;
    const target = typeof finding.subject.target === "string" ? null : finding.subject.target.id;
    const indicator = entities.indicators.find((item) => item.id === target);
    const edge = entities.edges.find((item) => item.id === target);
    const construct = entities.constructs.find((item) => item.id === finding.subject.construct_id);
    const owner: EntitySelection | undefined = indicator
      ? { kind: "indicator", id: indicator.id }
      : edge
        ? { kind: "edge", id: edge.id }
        : construct
          ? { kind: "construct", id: construct.id }
          : undefined;
    const entity = owner && resolveEntity(entities, owner);
    findings.push({
      label: `${humanize(finding.subject.check)}${entity ? ` · ${entity.label}` : ""}`,
      reason:
        finding.kind === "evaluated"
          ? finding.evidence.map((item) => item.note).join("; ")
          : finding.detail,
      status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      ...(owner ? { owner } : {}),
    });
  }
  for (const check of predictive?.predictive_checks?.per_variable_warnings ?? []) {
    if (check.kind === "evaluated" && check.outcome === "passed") continue;
    const target = check.subject.target;
    const indicator =
      typeof target === "string"
        ? undefined
        : entities.indicators.find((item) => item.id === target.id);
    if (!indicator) continue;
    findings.push({
      label: `${humanize(check.subject.check)} · ${humanize(indicator.name)}`,
      reason: check.kind === "evaluated" ? check.evidence.note : check.detail,
      status: check.kind === "evaluated" ? check.outcome : "not_evaluated",
      owner: { kind: "indicator", id: indicator.id },
    });
  }
  if (predictive?.status === "not_evaluated")
    findings.push({
      label: "Predictive checks",
      reason:
        predictive.detail ??
        (predictive.reason === null ? null : humanize(predictive.reason).toLowerCase()),
      status: "not_evaluated",
    });
  if (
    result.action === "prepare_data" &&
    result.workers.some((worker) => worker.status === "failed")
  )
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
              presentEntries(identification.treatments)
                .map(([, finding]) => finding)
                .filter((finding) => finding.status === "identified").length
            }{" "}
            identified;{" "}
            {
              presentEntries(identification.treatments)
                .map(([, finding]) => finding)
                .filter((finding) => finding.status === "not_identified").length
            }{" "}
            not identified.
          </Hint>
        </Section>
      )}
      {findings.length > 0 && (
        <Section title="Problems">
          {findings.map(({ owner, ...finding }, index) => (
            <div key={index} className="flex items-start gap-2">
              <StatusIcon status={finding.status} />
              <p className="min-w-0">
                {owner ? (
                  <OwnerLink onClick={() => select(owner)}>{finding.label}</OwnerLink>
                ) : (
                  finding.label
                )}
                {finding.reason !== null && <>: {finding.reason}</>}
              </p>
            </div>
          ))}
        </Section>
      )}
    </>
  );
}
