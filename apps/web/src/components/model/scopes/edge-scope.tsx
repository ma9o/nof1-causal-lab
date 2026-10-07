import type { EdgeId } from "@nof1-causal-lab/api-types";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Prose, Section } from "../scope-primitives";
import { LawSections } from "./law-sections";

export function EdgeScope({ context, id }: { context: ScopeContext; id: EdgeId }) {
  const edge = context.entities.edgeById.get(id);
  if (!edge) return null;
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
      <LawSections context={context} uses={ownLawUses(edge)} />
    </>
  );
}
