import type { ArtifactFreshness } from "@/lib/api/analysis";
import { TRANSITIONS, type PipelineSectionId } from "@nof1-causal-lab/api-types";

/** Stale artifact ids grouped by the artifact transition or derivation that produced them. */
export type StaleArtifactsByProducer = Partial<Record<PipelineSectionId, string[]>>;

function isPipelineSectionId(value: unknown): value is PipelineSectionId {
  return typeof value === "string" && TRANSITIONS.some((transition) => transition.id === value);
}

function producerArtifactId(producedBy: string | null | undefined): PipelineSectionId | null {
  if (!producedBy) {
    return null;
  }
  const [kind, artifactId] = producedBy.split(":", 2);
  if ((kind !== "run" && kind !== "derive") || !isPipelineSectionId(artifactId)) {
    return null;
  }
  return artifactId;
}

/**
 * Group the machine's freshness report by producing artifact for display.
 *
 * A producer is stale iff any artifact it produced exists and is stale. Root
 * artifacts (null `produced_by`, e.g. the question) belong to no producer and
 * are never stale by construction, so they are skipped. Staleness itself is
 * computed backend-side; this is pure presentation grouping.
 */
export function groupStaleArtifactsByProducer(
  artifacts: readonly ArtifactFreshness[],
): StaleArtifactsByProducer {
  const byProducer: StaleArtifactsByProducer = {};
  for (const artifact of artifacts) {
    const producerId = producerArtifactId(artifact.produced_by);
    if (!artifact.exists || !artifact.stale || producerId === null) {
      continue;
    }
    (byProducer[producerId] ??= []).push(artifact.artifact_id);
  }
  return byProducer;
}

export function hasStaleArtifacts(artifacts: readonly ArtifactFreshness[]): boolean {
  return Object.keys(groupStaleArtifactsByProducer(artifacts)).length > 0;
}
