import { presentEntries } from "@/lib/model-accessors";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { ActionSuccess } from "@nof1-causal-lab/api-types";
import { callModel } from "@/lib/model-asset/compose-call-view";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";

/** Current findings for the saved call remain visible when read from cache. */
export function ActionFindings({ context, call }: { context: ScopeContext; call: ActionSuccess }) {
  const { model, entities, select } = context;
  const checks = callModel(call)?.checks;
  const identification = call.action === "edit_model" ? call.body.identification : null;
  const validation = call.action === "fit" ? call.body.checks.validation : null;
  const data = validation?.data ?? (call.action === "prepare_data" ? call.body.profile : null);
  const question = call.action === "fit" ? call.body.checks.question : checks?.question;
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
    const variable = model.metadata?.variables.find((item) => item.id === issue.indicator_id);
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
  for (const finding of question?.findings ?? []) {
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
  if (
    call.action === "prepare_data" &&
    call.messages.some(
      (message) => message.kind === "log" && message.label === "EXTRACTION_PARTIAL",
    )
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
