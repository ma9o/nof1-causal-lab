import type { EdgeId } from "@nof1-causal-lab/api-types";
import {
  ArtifactChip,
  FactChip,
  Hint,
  KeyValue,
  OwnerLink,
  PosteriorTable,
  PriorTable,
  Prose,
  Section,
  Tag,
} from "../scope-primitives";
import { dispositionLabel } from "./construct-scope";
import { parametersForOwner, posteriorRows, priorRows } from "./parameters";
import { chipFor, has, type ScopeContext } from "./scope-context";

export function EdgeScope({ context, id }: { context: ScopeContext; id: EdgeId }) {
  const edge = context.entities.edgeById.get(id);
  if (!edge) return null;
  const { lagged } = edge;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.source_id === id,
  );
  const cause = context.entities.constructById.get(edge.cause.id)!.name;
  const effect = context.entities.constructById.get(edge.effect.id)!.name;
  const parameters = parametersForOwner(context.model.model?.value, edge.id);
  const priorParameters = parametersForOwner(context.model.model?.value, id);
  const priors = priorRows(priorParameters);
  const fitted = posteriorRows(parameters, context.model.findings.fit?.value.report);
  return (
    <>
      <Section title="Structure" chips={<ArtifactChip {...chipFor(context, "model")} />}>
        <div className="flex flex-wrap items-center gap-1 text-[11px]">
          <Tag>{lagged ? "t−1 → t" : "same t"}</Tag>
          <OwnerLink onClick={() => context.select({ kind: "construct", id: edge.cause.id })}>
            {cause}
          </OwnerLink>
          <span className="text-muted-foreground">→</span>
          <OwnerLink onClick={() => context.select({ kind: "construct", id: edge.effect.id })}>
            {effect}
          </OwnerLink>
        </div>
        <Prose>{edge.description}</Prose>
        {edge.sources.length > 0 ? (
          <>
            <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Sources
            </div>
            <ul className="m-0 flex list-none flex-col gap-1 p-0">
              {edge.sources.map((source) => (
                <li key={source.title}>
                  <Hint>
                    {source.url ? (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noreferrer"
                        className="underline underline-offset-[3px]"
                      >
                        {source.title}
                      </a>
                    ) : (
                      source.title
                    )}
                  </Hint>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <Hint>no sources cited</Hint>
        )}
      </Section>
      {disposition ? (
        <Section title="Design" chips={<ArtifactChip {...chipFor(context, "model")} />}>
          <KeyValue
            rows={[
              [
                "disposition",
                <Tag
                  key="d"
                  tone={disposition.disposition === "retained_edge" ? "success" : "warning"}
                >
                  {dispositionLabel(disposition.disposition)}
                </Tag>,
              ],
              ["reason", disposition.reason],
            ]}
          />
        </Section>
      ) : null}
      {has(context, "model") ? (
        <Section title="Model" chips={<ArtifactChip {...chipFor(context, "model")} />}>
          {priors.length > 0 ? (
            <PriorTable rows={priors} />
          ) : (
            <Hint>No parameter: the edge is projected out of the executable state.</Hint>
          )}
        </Section>
      ) : null}
      {fitted.length > 0 ? (
        <Section title="Fit" chips={<FactChip source={context.model.findings.fit?.source} />}>
          <PosteriorTable rows={fitted} />
        </Section>
      ) : null}
    </>
  );
}
