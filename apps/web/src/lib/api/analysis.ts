import type {
  ArtifactFreshness,
  PipelineSectionId,
  Move,
  RuntimeEvent,
  TransitionRecord,
} from "@nof1-causal-lab/api-types";
import { apiFetch } from "./client";
export type {
  ArtifactFreshness,
  Move,
  RuntimeEvent,
  TransitionRecord,
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
  nextOperation: import("@nof1-causal-lab/api-types").OperationId | null;
  workspaceId: string;
  autoRunning: boolean;
  seq: number;
  artifacts: ArtifactFreshness[];
  /** Moves the machine accepts right now, as the facade reports them. */
  legal: Move[];
  transitions: TransitionRecord[];
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

export async function recomputeStaleArtifacts(
  workspaceId: string,
): Promise<{ ok: true; workspaceId: string }> {
  return apiFetch<{ ok: true; workspaceId: string }>(`/api/analysis/${workspaceId}/recompute`, {
    method: "POST",
  });
}
