import type { ArtifactId, ModelSnapshot } from "@nof1-causal-lab/api-types";
import type { AssetSnapshot } from "@/lib/model-asset/journal";

/** Index the single semantic projection; names are presentation values only. */
export function indexModel(model: ModelSnapshot) {
  const constructs = model.latent_structure?.value.constructs ?? [];
  const edges = model.latent_structure?.value.edges ?? [];
  const indicators = model.measurement_structure?.value.measurement_structure.indicators ?? [];
  return {
    constructs,
    edges,
    indicators,
    constructById: new Map(constructs.map((entity) => [entity.id, entity])),
    edgeById: new Map(edges.map((entity) => [entity.id, entity])),
    indicatorById: new Map(indicators.map((entity) => [entity.id, entity])),
  };
}

export function presentAt(snapshot: AssetSnapshot, artifactId: ArtifactId): boolean {
  return snapshot.versions[artifactId] != null;
}
