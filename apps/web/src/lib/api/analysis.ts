import type {
  ArtifactFreshness,
  PipelineSectionId,
  RuntimeEvent,
  StudyRevision,
} from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";
export type {
  ArtifactFreshness,
  RuntimeEvent,
  StudyRevision,
} from "@nof1-causal-lab/api-types";

export interface AnalysisTransitionExecution {
  stateType: string;
  startTime: string | null;
  endTime: string | null;
}

export interface AnalysisTransitionRun {
  execution: AnalysisTransitionExecution | null;
}

export type AnalysisTransitionRuns = Record<PipelineSectionId, AnalysisTransitionRun>;

export interface AnalysisManifest {
  workspaceId: string;
  createdAt: string;
  question?: string;
  transitionOrder: PipelineSectionId[];
  transitionRuns: AnalysisTransitionRuns;
  /** Read-only artifact (e.g. a shared workspace): the UI hides LLM interaction. */
  readOnly: boolean;
}

export interface EpisodeProgressPayload {
  workspaceId: string;

  seq: number;
  artifacts: ArtifactFreshness[];
  /** Moves the machine accepts right now, as the facade reports them. */
  actions: import("@nof1-causal-lab/api-types").ScientificActionId[];
  transitions: StudyRevision[];
  branches: Record<string, string>;
  events: RuntimeEvent[];
}

export function getAnalysisManifestQueryKey(workspaceId: string) {
  return ["analysis", workspaceId, "manifest"] as const;
}

export async function getAnalysisManifest(workspaceId: string): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}`);
}

export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
): Promise<EpisodeProgressPayload> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return apiFetch<EpisodeProgressPayload>(`/api/analysis/${workspaceId}/progress${search}`);
}
