import type {
  ArtifactFreshness,
  ArtifactId,
  SimulationResult,
  ModelSnapshot,
} from "@nof1-causal-lab/api-types";
import type { AssetSnapshot, JournalTick } from "@/lib/model-asset/journal";
import type { indexModel } from "../asset-data";
import type { ModelSelection } from "../model-selection";
import type { ModelQuery } from "../queries";

/** Everything a scope panel may read, plus the two things it may do: select and move the playhead. */
export interface ScopeContext {
  model: ModelSnapshot;
  entities: ReturnType<typeof indexModel>;
  snapshot: AssetSnapshot;
  current: AssetSnapshot;
  canSimulate: boolean;
  ticks: JournalTick[];
  artifacts: ArtifactFreshness[];
  question: string | undefined;
  queries: ModelQuery[];
  outcome: string | null;
  setSimulation: (key: string, result: SimulationResult) => void;
  select: (selection: ModelSelection) => void;
  viewAt: (seq: number | null) => void;
  focusConversation: (seq: number) => void;
}

export function chipFor(context: ScopeContext, artifactId: ArtifactId) {
  const status = context.artifacts.find((artifact) => artifact.artifact_id === artifactId);
  return {
    id: artifactId,
    version: context.snapshot.versions[artifactId] ?? null,
    stale: status?.stale === true,
    retracted: context.snapshot.retracted.includes(artifactId),
  };
}

export function has(context: ScopeContext, artifactId: ArtifactId): boolean {
  return context.snapshot.versions[artifactId] != null;
}
