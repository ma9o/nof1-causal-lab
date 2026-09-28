import type { EntityLink, EntityPresentation } from "@/lib/model-asset/entities";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { OwnerLink } from "./scope-primitives";

/** Render the selected entity and links to its related model entities. */
export function DefinitionContext({
  entity,
  onSelect,
}: {
  entity: EntityPresentation;
  onSelect: (selection: EntitySelection) => void;
}) {
  const { selection, relationships } = entity;
  const relationship = (link: EntityLink) => (
    <OwnerLink onClick={() => onSelect(link.selection)}>{link.label}</OwnerLink>
  );
  if (selection.kind === "edge") {
    return (
      <div aria-label="Edge connection" className="flex flex-wrap items-center gap-2 text-xs">
        {relationship(relationships[0])}
        <span aria-label="causes" className="text-muted-foreground">
          →
        </span>
        {relationship(relationships[1])}
      </div>
    );
  }
  if (selection.kind === "indicator") {
    return (
      <div className="text-xs">
        <p className="text-muted-foreground">Indicator of {relationship(relationships[0])}</p>
      </div>
    );
  }
  return null;
}
