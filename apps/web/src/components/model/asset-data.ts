import { modelConstructs } from "@/lib/model-accessors";
import type { ArtifactId, ModelSnapshot } from "@nof1-causal-lab/api-types";
import type { AssetSnapshot } from "@/lib/model-asset/journal";

/** Index the single semantic projection; names are presentation values only. */
export function indexModel(model: ModelSnapshot) {
  const constructs = modelConstructs(model.model?.value) ?? [];
  const edges = model.model?.value.edges ?? [];
  const indicators = constructs.flatMap((construct) => construct.indicators);
  return {
    constructs,
    edges,
    indicators,
    constructById: new Map(constructs.map((entity) => [entity.id, entity])),
    edgeById: new Map(edges.map((entity) => [entity.id, entity])),
    indicatorOwnerById: new Map(
      constructs.flatMap((construct) =>
        construct.indicators.map((indicator) => [indicator.id, construct] as const),
      ),
    ),
    indicatorById: new Map(indicators.map((entity) => [entity.id, entity])),
  };
}

export function presentAt(snapshot: AssetSnapshot, artifactId: ArtifactId): boolean {
  return snapshot.versions[artifactId] != null;
}
