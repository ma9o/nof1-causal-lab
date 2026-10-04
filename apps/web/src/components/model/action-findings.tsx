import { presentEntries } from "@/lib/model-accessors";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { ActionAttempt } from "@nof1-causal-lab/api-types";
import { resolveEntity } from "@/lib/model-asset/entities";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";

/** Current findings for the saved call remain visible when read from cache. */
export function ActionFindings({
  context,
  applied,
}: {
  context: ScopeContext;
  applied: Extract<ActionAttempt["outcome"], { status: "applied" }>;
}) {
  const { model, entities, select } = context;
  const { result, effects } = applied;
  const checks = context.result?.checks;
  const produced = new Set(effects.produced.map((artifact) => artifact.artifact_id));
  const identification = checks ? model.identification?.value : null;
  const validation = checks ? model.validation_report?.value : null;
  const data = validation?.data ?? (produced.has("panel") ? model.profile?.value : null);
  const predictive = checks?.predictive;
  const findings: Array<{
    label: string;
    reason: string | null;
    status: "passed" | "failed" | "warning" | "error" | "not_evaluated";
    owner?: EntitySelection;
  }> = [];
  for (const finding of [...(checks?.specification ?? []), ...(validation?.preflight ?? [])]) {
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
    const indicator = entities.indicators.find(
      (item) => item.observation.id === issue.indicator_id,
    );
    const variable = model.metadata?.value.variables.find((item) => item.id === issue.indicator_id);
    findings.push({
      label: humanize(indicator?.observation.name ?? variable?.name ?? "Dataset"),
      reason: issue.message,
      status: issue.severity,
      ...(indicator ? { owner: { kind: "indicator" as const, id: indicator.observation.id } } : {}),
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
  for (const finding of checks?.question?.findings ?? []) {
    if (finding.kind === "evaluated" && finding.outcome === "passed") continue;
    const subject = finding.subject;
    const construct =
      subject.check === "outcome"
        ? subject.outcome.id
        : subject.check === "window"
          ? null
          : subject.target.id;
    const entity = entities.constructs.find((item) => item.id === construct);
    findings.push({
      label:
        subject.check === "outcome"
          ? "Question outcome"
          : `${subject.query} · ${humanize(subject.check)}${
              subject.check === "window"
                ? ""
                : ` · ${entity ? humanize(entity.name) : (construct ?? "")}`
            }`,
      reason: finding.kind === "evaluated" ? finding.evidence : finding.detail,
      status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      ...(entity ? { owner: { kind: "construct" as const, id: entity.id } } : {}),
    });
  }
  for (const finding of predictive?.evaluation.kind === "evaluated"
    ? predictive.evaluation.findings
    : []) {
    if (finding.kind === "evaluated" && finding.outcome === "passed") continue;
    const target = typeof finding.subject.target === "string" ? null : finding.subject.target.id;
    const indicator = entities.indicators.find((item) => item.observation.id === target);
    const edge = entities.edges.find((item) => item.id === target);
    const construct = entities.constructs.find((item) => item.id === finding.subject.construct_id);
    const owner: EntitySelection | undefined = indicator
      ? { kind: "indicator", id: indicator.observation.id }
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
  for (const check of predictive?.evaluation.kind === "evaluated"
    ? (predictive.evaluation.predictive_checks?.per_variable_warnings ?? [])
    : []) {
    if (check.kind === "evaluated" && check.outcome === "passed") continue;
    const target = check.subject.target;
    const indicator =
      typeof target === "string"
        ? undefined
        : entities.indicators.find((item) => item.observation.id === target.id);
    if (!indicator) continue;
    findings.push({
      label: `${humanize(check.subject.check)} · ${humanize(indicator.observation.name)}`,
      reason: check.kind === "evaluated" ? check.evidence.note : check.detail,
      status: check.kind === "evaluated" ? check.outcome : "not_evaluated",
      owner: { kind: "indicator", id: indicator.observation.id },
    });
  }
  if (predictive?.status === "not_evaluated")
    findings.push({
      label: "Predictive checks",
      reason:
        predictive.evaluation.kind === "unavailable"
          ? (predictive.evaluation.detail ?? humanize(predictive.evaluation.reason).toLowerCase())
          : null,
      status: "not_evaluated",
    });
  if (
    result != null &&
    "workers" in result &&
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
