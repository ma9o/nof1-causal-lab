import { modelConstructs } from "@/lib/model-accessors";
import type { ConstructSpec, ModelSpec, ParameterId } from "@nof1-causal-lab/api-types";
import { humanize, type EntitySelection } from "./selection";
import { ownLawUses } from "./laws";

/** Index the authored graph; these maps are derived, never a second model definition. */
export function indexModel(model: ModelSpec | undefined) {
  const constructs = modelConstructs(model);
  const edges = model?.edges ?? [];
  const indicators = constructs.flatMap((construct) => construct.indicators);
  const parameters = model?.parameters ?? [];
  return {
    constructs,
    edges,
    indicators,
    parameters,
    constructById: new Map(constructs.map((entity) => [entity.id, entity])),
    edgeById: new Map(edges.map((entity) => [entity.id, entity])),
    indicatorOwnerById: new Map(
      constructs.flatMap((construct) =>
        construct.indicators.map((indicator) => [indicator.id, construct] as const),
      ),
    ),
    indicatorById: new Map(indicators.map((entity) => [entity.id, entity])),
    parameterById: new Map(parameters.map((entity) => [entity.id, entity])),
  };
}

export type ModelEntities = ReturnType<typeof indexModel>;

export interface EntityLink {
  selection: EntitySelection;
  label: string;
}

export interface EntityPresentation extends EntityLink {
  relationships: EntityLink[];
}

/** Resolve ownership, labels and fields regardless of where endpoints serialize. */
export function resolveEntity(
  entities: ModelEntities,
  selection: EntitySelection,
): EntityPresentation | undefined {
  const constructLink = (construct: ConstructSpec): EntityLink => ({
    selection: { kind: "construct", id: construct.id },
    label: humanize(construct.name),
  });
  switch (selection.kind) {
    case "edge": {
      const edge = entities.edgeById.get(selection.id);
      if (!edge) return undefined;
      const relationships = [edge.cause, edge.effect].map((endpoint) =>
        constructLink(entities.constructById.get(endpoint.id)!),
      );
      return {
        selection,
        label: relationships.map((link) => link.label).join(" → "),
        relationships,
      };
    }
    case "indicator": {
      const indicator = entities.indicatorById.get(selection.id);
      return indicator
        ? {
            selection,
            label: humanize(indicator.name),
            relationships: [constructLink(entities.indicatorOwnerById.get(selection.id)!)],
          }
        : undefined;
    }
    case "construct": {
      const entity = entities.constructById.get(selection.id);
      return entity ? { selection, label: humanize(entity.name), relationships: [] } : undefined;
    }
  }
}

/** A parameter link opens the entity whose own law section displays it. */
export function parameterOwner(entities: ModelEntities, id: ParameterId) {
  for (const kind of ["edge", "indicator", "construct"] as const) {
    const entity = entities[`${kind}s`].find((entity) =>
      ownLawUses(entity).some((use) => use.parameterId === id),
    );
    if (entity) return resolveEntity(entities, { kind, id: entity.id } as EntitySelection);
  }
}
