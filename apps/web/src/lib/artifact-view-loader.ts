import { getModelView } from "@/lib/server/episode-runs";
import type { ArtifactViewData, ArtifactViewId } from "@nof1-causal-lab/api-types";
import { ARTIFACT_VIEW_IDS } from "@nof1-causal-lab/api-types";

export async function loadArtifactView<K extends ArtifactViewId>(
  artifactId: K,
  workspaceId: string,
): Promise<ArtifactViewData<K>>;
export async function loadArtifactView(
  artifactId: string,
  workspaceId: string,
): Promise<ArtifactViewData>;
export async function loadArtifactView(
  artifactId: string,
  workspaceId: string,
): Promise<ArtifactViewData> {
  if (!ARTIFACT_VIEW_IDS.includes(artifactId as ArtifactViewId)) {
    throw new Error(`Unknown artifact view '${artifactId}'`);
  }
  return getModelView(workspaceId, artifactId as ArtifactViewId);
}
