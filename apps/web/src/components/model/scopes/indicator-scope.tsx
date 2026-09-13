import type { IndicatorId } from "@nof1-causal-lab/api-types";
import { humanize } from "../model-selection";
import {
  ArtifactChip,
  Hint,
  KeyValue,
  OwnerLink,
  PosteriorTable,
  PriorTable,
  Section,
  Tag,
} from "../scope-primitives";
import { dispositionLabel } from "./construct-scope";
import { parametersForOwner, posteriorRows, priorRows } from "./parameters";
import { chipFor, type ScopeContext } from "./scope-context";

export function IndicatorScope({ context, id }: { context: ScopeContext; id: IndicatorId }) {
  const indicator = context.entities.indicatorById.get(id);
  if (!indicator) return null;
  const disposition = context.model.dispositions?.value.find((item) => item.source_id === id);
  const audit = context.model.validation_report?.value.indicators[id];
  const counts = context.model.measurements?.value.per_indicator_counts[id];
  const likelihood = context.model.specification?.value.statistical_model_spec.likelihoods.find((item) => item.indicator_id === id);
  const parameters = parametersForOwner(context.model.compiled_parameters?.value ?? [], id);
  const fitted = posteriorRows(parameters, context.model.fit?.value.posterior);
  const checks = context.model.fit?.value.posterior.assessment.ppc.per_variable_warnings.filter((item) => item.indicator_id === id) ?? [];
  const issues = audit?.validation.issues.filter((issue) => issue.severity !== "info") ?? [];
  const checkEntries = audit ? Object.entries(audit.validation.checks) : [];
  const okChecks = checkEntries.filter(([, status]) => status === "ok").length;
  return (
    <>
      <Section
        title="Measurement"
        chips={<ArtifactChip {...chipFor(context, "measurement_structure")} />}
      >
        <div className="text-[11px]">
          <OwnerLink
            onClick={() => context.select({ kind: "construct", id: indicator.construct_id })}
          >
            {context.entities.constructById.get(indicator.construct_id)!.name}
          </OwnerLink>
        </div>
        <div className="flex flex-wrap gap-1">
          <Tag>{indicator.measurement_dtype}</Tag>
          <Tag tone="secondary">{indicator.aggregation}</Tag>
          <Tag tone="secondary">
            window {indicator.observation_window ?? context.model.measurement_structure?.value.measurement_structure.model_clock}
          </Tag>
          <Tag tone="secondary">{indicator.extraction_mode}</Tag>
          <Tag>{indicator.construct_polarity}</Tag>
        </div>
        <Hint>{indicator.how_to_measure}</Hint>
        <div className="flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground">
          sources:
          {indicator.source_columns.map((column) => (
            <span
              key={column}
              className="rounded border border-foreground px-1.5 font-mono text-[9px] text-foreground"
            >
              {column}
            </span>
          ))}
        </div>
      </Section>
      {disposition ? (
        <Section title="Design" chips={<ArtifactChip {...chipFor(context, "structural_plan")} />}>
          <KeyValue
            rows={[
              [
                "disposition",
                <Tag
                  key="d"
                  tone={disposition.disposition === "excluded_indicator" ? "warning" : "secondary"}
                >
                  {dispositionLabel(disposition.disposition)}
                </Tag>,
              ],
              ["reason", disposition.reason],
            ]}
          />
        </Section>
      ) : null}
      {audit ? (
        <Section
          title="Evidence"
          chips={
            <>
              <ArtifactChip {...chipFor(context, "measurements")} />
              <ArtifactChip {...chipFor(context, "validation_report")} />
            </>
          }
        >
          <div className="flex flex-wrap gap-1">
            <Tag tone={issues.length > 0 ? "warning" : "success"}>
              {okChecks}/{checkEntries.length} checks ok
            </Tag>
            {checkEntries
              .filter(([, status]) => status !== "ok")
              .map(([check, status]) => (
                <Tag key={check} tone={status === "error" ? "destructive" : "warning"}>
                  {humanize(check)} · {status}
                </Tag>
              ))}
          </div>
          {counts != null ? <Hint>{counts.toLocaleString()} observations extracted</Hint> : null}
          {issues.map((issue) => (
            <Hint key={`${issue.issue_type}-${issue.message}`} issue>
              {humanize(issue.issue_type)}: {issue.message}
            </Hint>
          ))}
        </Section>
      ) : null}
      {likelihood ? (
        <Section
          title="Model"
          chips={<ArtifactChip {...chipFor(context, "statistical_model_spec")} />}
        >
          <div className="flex flex-wrap gap-1">
            <Tag tone="secondary">{likelihood.distribution}</Tag>
            <Tag>{likelihood.link} link</Tag>
            {likelihood.standardized ? <Tag>standardized</Tag> : null}
          </div>
          <PriorTable rows={priorRows(parameters, context.model.specification?.value.resolved_priors ?? [])} />
        </Section>
      ) : null}
      {checks.length > 0 || (parameters.length > 0 && fitted.length > 0) ? (
        <Section title="Fit" chips={<ArtifactChip {...chipFor(context, "posterior")} />}>
          {checks.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {checks.map((check) => (
                <Tag key={check.check_type} tone={check.passed ? "success" : "destructive"}>
                  {check.check_type} {check.passed ? "✓" : "✗"} {check.value.toFixed(2)}
                </Tag>
              ))}
            </div>
          ) : null}
          <PosteriorTable rows={fitted} />
        </Section>
      ) : null}
    </>
  );
}
