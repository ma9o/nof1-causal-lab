import type { ConstructId } from "@nof1-causal-lab/api-types";
import { signColor } from "@/components/dag/core/palette";
import { Button } from "@/components/ui/button";
import { formatPlain, formatSigned, humanize } from "../model-selection";
import { rankingQueryKey } from "../queries";
import {
  ArtifactChip,
  FactChip,
  Callout,
  Hint,
  KeyValue,
  OwnerLink,
  PosteriorTable,
  PriorTable,
  Prose,
  Section,
  Tag,
} from "../scope-primitives";
import { parametersForOwner, posteriorRows, priorRows } from "./parameters";
import { chipFor, has, type ScopeContext } from "./scope-context";

const DISPOSITION_LABEL: Record<string, string> = {
  retained_state: "retained state",
  known_input: "known input",
  marginalized: "marginalized",
  identification_only: "identification-only",
  retained_edge: "retained edge",
  projected_edge: "projected edge",
  manifest: "manifest",
  excluded_indicator: "excluded indicator",
  known_input_source: "known-input source",
};

export function dispositionLabel(disposition: string): string {
  return DISPOSITION_LABEL[disposition] ?? humanize(disposition);
}

export function ConstructScope({ context, id }: { context: ScopeContext; id: ConstructId }) {
  const { model, entities, queries } = context;
  const construct = entities.constructById.get(id);
  if (!construct) return null;
  const inEdges = entities.edges.filter((edge) => edge.effect.id === id);
  const outEdges = entities.edges.filter((edge) => edge.cause.id === id);
  const indicators = construct.indicators;
  const scientificOnly = construct.usage?.kind === "scientific_only" ? construct.usage : null;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.source_id === id,
  );
  const finding =
    model.findings.identification?.value.status.identifiable_treatments[id] ??
    model.findings.identification?.value.status.non_identifiable_treatments[id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const parameters = parametersForOwner(context.model.model?.value, id);
  const priorParameters = parametersForOwner(context.model.model?.value, id);
  const priors = priorRows(priorParameters);
  const admission =
    model.findings.admission_report?.value.prior_predictive_diagnostics.filter(
      (item) => item.construct_id === id,
    ) ?? [];
  const fitted = posteriorRows(parameters, context.model.findings.fit?.value.report);
  const rankingQuery = queries.find((query) => query.key === rankingQueryKey(id));
  const effect = rankingQuery?.posterior ?? null;
  const temporal = model.findings.baseline_report?.value.intervention_results.find(
    (effect) => effect.treatment_id === id,
  )?.temporal;
  const namesFor = (ids: ConstructId[]) =>
    ids.map((id) => entities.constructById.get(id)?.name ?? id).join(", ");
  const effectStale = context.model.findings.baseline_report?.source.validity === "stale";

  return (
    <>
      <Section title="Structure" chips={<ArtifactChip {...chipFor(context, "model")} />}>
        <Prose>{construct.description}</Prose>
        <KeyValue
          rows={[
            ["role", construct.role],
            ["temporal", construct.temporal_status.replace("_", "-")],
            [
              "default query outcome",
              model.model?.value.default_outcome?.id === construct.id ? "yes" : "no",
            ],
          ]}
        />
        {inEdges.length + outEdges.length > 0 ? (
          <>
            <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Edges
            </div>
            <ul className="m-0 flex list-none flex-col gap-1 p-0">
              {inEdges.map((edge) => (
                <li key={edge.id} className="flex min-w-0 items-center gap-1.5 text-[11px]">
                  <span className="flex-none text-muted-foreground">←</span>
                  <OwnerLink
                    onClick={() =>
                      context.select({
                        kind: "edge",
                        id: edge.id,
                      })
                    }
                  >
                    {entities.constructById.get(edge.cause.id)!.name}
                  </OwnerLink>
                  <Tag>{edge.lagged ? "t−1 → t" : "same t"}</Tag>
                </li>
              ))}
              {outEdges.map((edge) => (
                <li key={edge.id} className="flex min-w-0 items-center gap-1.5 text-[11px]">
                  <span className="flex-none text-muted-foreground">→</span>
                  <OwnerLink
                    onClick={() =>
                      context.select({
                        kind: "edge",
                        id: edge.id,
                      })
                    }
                  >
                    {entities.constructById.get(edge.effect.id)!.name}
                  </OwnerLink>
                  <Tag>{edge.lagged ? "t−1 → t" : "same t"}</Tag>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </Section>
      {model.model?.value.measurement_clock ? (
        <Section title="Measurement" chips={<ArtifactChip {...chipFor(context, "model")} />}>
          {indicators && indicators.length > 0 ? (
            <ul className="m-0 flex list-none flex-col gap-1 p-0">
              {indicators.map((indicator) => (
                <li key={indicator.name}>
                  <OwnerLink
                    onClick={() => context.select({ kind: "indicator", id: indicator.id })}
                  >
                    {indicator.name}
                  </OwnerLink>
                </li>
              ))}
            </ul>
          ) : (
            <Hint>No indicator declared.</Hint>
          )}
          {scientificOnly ? (
            <Hint>
              <b>scientific-only:</b> {scientificOnly.reason}
            </Hint>
          ) : null}
        </Section>
      ) : null}
      {disposition ? (
        <Section
          title="Design"
          chips={
            <>
              <ArtifactChip {...chipFor(context, "model")} />
              <ArtifactChip {...chipFor(context, "model")} />
              {has(context, "identification_report") ? (
                <ArtifactChip {...chipFor(context, "identification_report")} />
              ) : null}
            </>
          }
        >
          <KeyValue
            rows={[
              [
                "disposition",
                <Tag
                  key="disposition"
                  tone={
                    disposition.disposition === "retained_state"
                      ? "success"
                      : disposition.disposition === "known_input"
                        ? "secondary"
                        : "warning"
                  }
                >
                  {dispositionLabel(disposition.disposition)}
                </Tag>,
              ],
              ["reason", disposition.reason],
            ]}
          />
          {identified ? (
            <Callout tone="ok">
              <b>Identified</b> via {identified.method.replaceAll("_", "-")}. Marginalized
              confounders:{" "}
              {identified.marginalized_confounders.length > 0
                ? namesFor(identified.marginalized_confounders)
                : "none"}
              .
            </Callout>
          ) : notIdentified ? (
            <Callout tone="bad">
              <b>Not identified.</b> {namesFor(notIdentified.confounders)} confound this treatment
              under the current design.{notIdentified.notes ? ` ${notIdentified.notes}` : ""}
            </Callout>
          ) : (
            <Hint>
              Not a treatment: identification is assessed per treatment against the outcome.
            </Hint>
          )}
        </Section>
      ) : null}
      {priors.length > 0 ? (
        <Section title="Model" chips={<ArtifactChip {...chipFor(context, "model")} />}>
          <PriorTable rows={priors} />
          {admission.map((entry) => (
            <Hint key={`${entry.check}-${entry.mode}`}>
              prior-predictive {humanize(entry.check)}: {entry.passed ? "passed" : "review"} ·{" "}
              {entry.value}
            </Hint>
          ))}
        </Section>
      ) : null}
      {fitted.length > 0 ? (
        <Section title="Fit" chips={<FactChip source={context.model.findings.fit?.source} />}>
          <PosteriorTable rows={fitted} />
        </Section>
      ) : null}
      {effect && rankingQuery ? (
        <Section
          title="Effect on the outcome"
          chips={<ArtifactChip {...chipFor(context, "baseline_report")} />}
        >
          <Hint>
            do(+1 latent unit) on this construct from the steady state
            {context.outcome ? (
              <>
                {" "}
                · outcome <b>{humanize(context.outcome)}</b>
              </>
            ) : null}
          </Hint>
          <div className={effectStale ? "opacity-55" : undefined}>
            <div
              className="font-mono text-[22px] font-semibold"
              style={{ color: signColor(effect.mean) }}
            >
              {formatSigned(effect.mean, 3)}
            </div>
            <Hint>
              {effect.mean > 0 ? "raises" : effect.mean < 0 ? "lowers" : "leaves"} the outcome at
              the steady state · 95% [{formatPlain(effect.lower_95)}, {formatPlain(effect.upper_95)}
              ] · P&gt;0 {Math.round(effect.prob_positive * 100)}% ·{" "}
              {effectStale ? "posterior · stale" : "posterior"}
            </Hint>
          </div>
          {temporal ? (
            <Hint>
              <span className="font-mono">
                {temporal.horizons
                  .map(({ day, effect }) => `${day} d ${formatSigned(effect)}`)
                  .join(" · ")}
                {" · "}peak {formatSigned(temporal.peak_effect)} at day {temporal.time_to_peak_days}
              </span>
            </Hint>
          ) : null}
          <div className="mt-1 flex flex-wrap gap-1.5">
            <Button
              type="button"
              size="sm"
              onClick={() => context.select({ kind: "query", key: rankingQuery.key })}
            >
              Open the query
            </Button>
          </div>
        </Section>
      ) : null}
    </>
  );
}
