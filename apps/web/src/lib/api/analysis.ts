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
  /** The attempt the episode's Temporal workflow is executing, if any. */
  running: import("@nof1-causal-lab/api-types").RunningAction | null;
  transitions: StudyRevision[];
  branches: Record<string, string>;
  events: RuntimeEvent[];
}

/** The progress route omits the journal when it is unchanged since `knownSeq`. */
export type EpisodeProgressResponse = Omit<EpisodeProgressPayload, "transitions" | "branches"> & {
  transitions: StudyRevision[] | null;
  branches: Record<string, string> | null;
};

export function getAnalysisManifestQueryKey(workspaceId: string) {
  return ["analysis", workspaceId, "manifest"] as const;
}

export async function getAnalysisManifest(workspaceId: string): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/api/analysis/${workspaceId}`);
}

/** Without a known seq the journal is always included. */
export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
): Promise<EpisodeProgressPayload>;
export async function getEpisodeProgress(
  workspaceId: string,
  after: string | null,
  knownSeq: number | undefined,
): Promise<EpisodeProgressResponse>;
export async function getEpisodeProgress(
  workspaceId: string,
  after?: string | null,
  knownSeq?: number,
): Promise<EpisodeProgressResponse> {
  const params = new URLSearchParams();
  if (after) params.set("after", after);
  if (knownSeq !== undefined) params.set("seq", String(knownSeq));
  const search = params.size > 0 ? `?${params.toString()}` : "";
  return apiFetch<EpisodeProgressResponse>(`/api/analysis/${workspaceId}/progress${search}`);
}
