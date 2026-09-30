import type { EdgeId } from "@nof1-causal-lab/api-types";
import { dispositionLabel } from "@/lib/model-asset/inspector";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, Prose, Section } from "../scope-primitives";
import { LawSections } from "./law-sections";
import { MechanismResponse } from "./mechanism-response";

export function EdgeScope({ context, id }: { context: ScopeContext; id: EdgeId }) {
  const edge = context.entities.edgeById.get(id);
  if (!edge) return null;
  const disposition = context.model.findings.dispositions?.value.find(
    (item) => item.target.id === id,
  );
  return (
    <>
      <Section title="Relationship">
        <Prose>{edge.description}</Prose>
        {edge.sources.length > 0 && (
          <details>
            <summary className="cursor-pointer text-muted-foreground">Sources</summary>
            <ul className="mt-2 space-y-2">
              {edge.sources.map((source) => (
                <li key={source.title}>
                  {source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="underline underline-offset-2"
                    >
                      {source.title}
                    </a>
                  ) : (
                    source.title
                  )}
                </li>
              ))}
            </ul>
          </details>
        )}
      </Section>
      {edge.mechanisms.length > 0 && <MechanismResponse context={context} owner={id} />}
      <LawSections context={context} uses={ownLawUses(edge)} />
      {disposition && disposition.disposition !== "retained_edge" && (
        <Section
          title={dispositionLabel(disposition.disposition)}
          source={context.model.findings.dispositions?.source}
        >
          <Hint issue>{disposition.reason}</Hint>
        </Section>
      )}
    </>
  );
}
