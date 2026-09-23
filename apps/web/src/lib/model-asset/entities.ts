import { modelConstructs } from "@/lib/model-accessors";
import type {
  CausalEdgeSpec,
  ConstructSpec,
  IndicatorSpec,
  ModelSpec,
  ParameterSpec,
} from "@nof1-causal-lab/api-types";
import { humanize, type EditableSelection, type ModelSelection } from "./selection";

/** Index the authored graph; these maps are derived, never a second model definition. */
export function indexModel(model: ModelSpec | undefined) {
  const constructs = modelConstructs(model);
  const edges = model?.edges ?? [];
  const indicators = constructs.flatMap((construct) => construct.indicators);
  const parameters = model?.parameters ?? [];
  return {
    model,
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
  selection: EditableSelection;
  label: string;
}

export interface EntityPresentation extends EntityLink {
  definition:
    | ModelSpec
    | ConstructSpec
    | IndicatorSpec
    | ParameterSpec
    | Omit<CausalEdgeSpec, "cause" | "effect">;
  relationships: EntityLink[];
}

/** Viewing and editing share ownership, labels and fields regardless of where endpoints serialize. */
export function resolveEntity(
  entities: ModelEntities,
  selection: ModelSelection,
): EntityPresentation | undefined {
  const constructLink = (construct: ConstructSpec): EntityLink => ({
    selection: { kind: "construct", id: construct.id },
    label: humanize(construct.name),
  });
  switch (selection.kind) {
    case "version":
      return undefined;
    case "asset":
      return entities.model
        ? { selection, label: "Whole model", definition: entities.model, relationships: [] }
        : undefined;
    case "edge": {
      const edge = entities.edgeById.get(selection.id);
      if (!edge) return undefined;
      const relationships = [edge.cause, edge.effect].map((endpoint) =>
        constructLink(entities.constructById.get(endpoint.id)!),
      );
      const definition = {
        id: edge.id,
        description: edge.description,
        lagged: edge.lagged,
        mechanisms: edge.mechanisms,
        sources: edge.sources,
      };
      return {
        selection,
        label: relationships.map((link) => link.label).join(" → "),
        definition,
        relationships,
      };
    }
    case "indicator": {
      const indicator = entities.indicatorById.get(selection.id);
      return indicator
        ? {
            selection,
            label: humanize(indicator.name),
            definition: indicator,
            relationships: [constructLink(entities.indicatorOwnerById.get(selection.id)!)],
          }
        : undefined;
    }
    case "construct":
    case "parameter": {
      const entity =
        selection.kind === "construct"
          ? entities.constructById.get(selection.id)
          : entities.parameterById.get(selection.id);
      return entity
        ? { selection, label: humanize(entity.name), definition: entity, relationships: [] }
        : undefined;
    }
  }
}

export function entityOptions(entities: ModelEntities): EntityLink[] {
  const selections: EditableSelection[] = [
    ...entities.constructs.map(({ id }) => ({ kind: "construct" as const, id })),
    ...entities.indicators.map(({ id }) => ({ kind: "indicator" as const, id })),
    ...entities.edges.map(({ id }) => ({ kind: "edge" as const, id })),
    ...entities.parameters.map(({ id }) => ({ kind: "parameter" as const, id })),
  ];
  return [
    { selection: { kind: "asset" }, label: "Whole model" },
    ...selections.map((selection) => ({
      selection,
      label: `${selection.kind[0].toUpperCase()}${selection.kind.slice(1)} · ${resolveEntity(entities, selection)!.label}`,
    })),
  ];
}
