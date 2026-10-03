import type { ConstructId, ConstructSpec } from "@nof1-causal-lab/api-types";
import { constructPresentation, dispositionLabel } from "@/lib/model-asset/inspector";
import { ownLawUses } from "@/lib/model-asset/laws";
import { humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import {
  Callout,
  Hint,
  KeyValue,
  OwnerLink,
  Prose,
  Section,
  StatusIcon,
} from "../scope-primitives";
import { LawSections, SimulatedHistory } from "./law-sections";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { MechanismResponse } from "./mechanism-response";
import { PredictiveFindings } from "../simulation-evidence";

export function ConstructScope({ context, id }: { context: ScopeContext; id: ConstructId }) {
  const scope = constructPresentation(context, id);
  if (!scope) return null;
  const { model, construct, indicators, disposition } = scope;
  const equations = model;
  const predictive = model.predictive;
  const findings = (
    predictive?.value.evaluation.kind === "evaluated" ? predictive.value.evaluation.findings : []
  ).filter(
    (finding) =>
      finding.subject.construct_id === id &&
      !indicators.some(
        (indicator) =>
          typeof finding.subject.target !== "string" &&
          indicator.observation.id === finding.subject.target.id,
      ),
  );
  return (
    <>
      <Section title="Structure">
        <Prose>{construct.description}</Prose>
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">Properties</summary>
          <div className="mt-2">
            <KeyValue
              rows={[
                ["Role", construct.role],
                ["Time", humanize(construct.temporal_status)],
              ]}
            />
          </div>
        </details>
      </Section>
      {indicators.length > 0 && (
        <Section title="Indicators">
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {indicators.map((indicator) => (
              <li key={indicator.observation.id}>
                <OwnerLink
                  onClick={() =>
                    context.select({ kind: "indicator", id: indicator.observation.id })
                  }
                >
                  {humanize(indicator.observation.name)}
                </OwnerLink>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {[equations.state_equations[id], equations.confounder_equations[id]].flatMap(
        (latex, index) =>
          latex === undefined
            ? []
            : [
                <Section key={index} title="Equation" wide>
                  <Katex latex={latex} />
                </Section>,
              ],
      )}
      <LawSections context={context} uses={ownLawUses(construct)} />
      {construct.dynamics.length > 0 && <MechanismResponse context={context} owner={id} />}
      <SimulatedHistory context={context} id={id} kind="states" />
      {disposition && disposition.disposition !== "retained_state" && (
        <Section
          title={dispositionLabel(disposition.disposition)}
          {...(model.dispositions?.source === undefined
            ? {}
            : { source: model.dispositions.source })}
        >
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
      {model.identification?.value.treatments[id] && (
        <Section title="Identification" source={model.identification.source}>
          <IdentificationFinding context={context} construct={construct} />
        </Section>
      )}
      {findings.length > 0 && (
        <Section
          title="Predictive checks"
          {...(predictive?.source === undefined ? {} : { source: predictive.source })}
          wide
        >
          <PredictiveFindings findings={findings} entities={context.entities} />
        </Section>
      )}
    </>
  );
}

export function IdentificationFinding({
  context,
  construct,
}: {
  context: ScopeContext;
  construct: ConstructSpec;
}) {
  const finding = context.model.identification?.value.treatments[construct.id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const namesFor = (ids: readonly ConstructId[]) =>
    context.entities.constructs
      .filter((entity) => ids.includes(entity.id))
      .map((entity) => humanize(entity.name))
      .join(", ");
  return (
    <>
      {identified && (
        <Callout tone="ok">
          <div className="flex items-start gap-2">
            <StatusIcon status="passed" label="Identified" />
            <span>{humanize(identified.method)}</span>
          </div>
          {identified.marginalized_confounders.length > 0 && (
            <p className="mt-2">
              Marginalized confounders: {namesFor(identified.marginalized_confounders)}.
            </p>
          )}
          <p className="mt-2 break-words font-mono">{identified.estimand}</p>
        </Callout>
      )}
      {notIdentified && (
        <Callout tone="bad">
          <div className="flex items-start gap-2">
            <StatusIcon status="failed" label="Not identified" />
            <span>
              {namesFor(notIdentified.confounders)} confound this treatment under the current
              design.{notIdentified.notes ? ` ${notIdentified.notes}` : ""}
            </span>
          </div>
        </Callout>
      )}
    </>
  );
}
