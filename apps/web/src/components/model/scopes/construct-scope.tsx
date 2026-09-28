import type { ConstructId } from "@nof1-causal-lab/api-types";
import { constructPresentation, dispositionLabel } from "@/lib/model-asset/inspector";
import { humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import {
  Callout,
  Hint,
  KeyValue,
  OwnerLink,
  ParameterLinks,
  Prose,
  Section,
  StatusIcon,
} from "../scope-primitives";
import { parametersForOwner } from "./parameters";

export function ConstructScope({ context, id }: { context: ScopeContext; id: ConstructId }) {
  const scope = constructPresentation(context, id);
  if (!scope) return null;
  const {
    model,
    entities,
    construct,
    inEdges,
    outEdges,
    indicators,
    disposition,
    identified,
    notIdentified,
    admission,
    namesFor,
  } = scope;
  const parameters = parametersForOwner(model.model?.value, id);
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
        {inEdges.length + outEdges.length > 0 && (
          <ul className="mt-2 flex list-none flex-col gap-2 p-0">
            {[
              ...inEdges.map((edge) => ({ edge, incoming: true })),
              ...outEdges.map((edge) => ({ edge, incoming: false })),
            ].map(({ edge, incoming }) => (
              <li key={edge.id} className="flex items-start gap-2">
                <span
                  className="text-muted-foreground"
                  title={incoming ? "Incoming relationship" : "Outgoing relationship"}
                  aria-hidden="true"
                >
                  {incoming ? "←" : "→"}
                </span>
                <OwnerLink onClick={() => context.select({ kind: "edge", id: edge.id })}>
                  {humanize(
                    entities.constructById.get(incoming ? edge.cause.id : edge.effect.id)!.name,
                  )}
                </OwnerLink>
              </li>
            ))}
          </ul>
        )}
      </Section>
      {indicators.length > 0 && (
        <Section title="Indicators">
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {indicators.map((indicator) => (
              <li key={indicator.id}>
                <OwnerLink onClick={() => context.select({ kind: "indicator", id: indicator.id })}>
                  {humanize(indicator.name)}
                </OwnerLink>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {disposition && disposition.disposition !== "retained_state" && (
        <Section
          title={dispositionLabel(disposition.disposition)}
          source={model.findings.dispositions?.source}
        >
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
      {(identified || notIdentified) && (
        <Section title="Identification" source={model.findings.identification?.source}>
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
        </Section>
      )}
      {parameters.length > 0 && (
        <Section title="Parameters">
          <ParameterLinks parameters={parameters} onSelect={context.select} />
        </Section>
      )}
      {admission.length > 0 && (
        <Section title="Prior checks" source={model.findings.prior_predictive?.source}>
          <details open={admission.some((entry) => !entry.passed)}>
            <summary className="cursor-pointer text-muted-foreground">Inspect checks</summary>
            <ul className="mt-2 space-y-2">
              {admission.map((entry) => (
                <li key={`${entry.check}-${entry.mode}`} className="flex items-start gap-2">
                  <StatusIcon status={entry.passed ? "passed" : "failed"} />
                  <span>
                    {humanize(entry.check)}
                    <Hint>{entry.value}</Hint>
                  </span>
                </li>
              ))}
            </ul>
          </details>
        </Section>
      )}
    </>
  );
}
