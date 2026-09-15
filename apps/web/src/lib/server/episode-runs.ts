import type {
  ArtifactViewData,
  ArtifactViewId,
  CapabilitiesResponse,
  EpisodeStatus,
  LLMTrace,
  MachineDescription,
  MoveOutcome,
  RuntimeEvent,
  TransitionRecord,
  TransitionTraceIndex,
} from "@nof1-causal-lab/api-types";
import { getToolServerUrl } from "@/lib/runtime-urls";

export type {
  ArtifactFreshness,
  ArtifactId,
  EpisodeState,
  EpisodeStatus,
  JournalStatus,
  JsonObject,
  MachineDescription,
  Move,
  MoveOutcome,
  Provenance,
  RetractedArtifact,
  RuntimeEvent,
  TransitionRecord,
  TransitionTraceIndex,
} from "@nof1-causal-lab/api-types";

const TOOL_SERVER = getToolServerUrl();

export class EpisodeRunError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function episodeFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${TOOL_SERVER}/api/episodes${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 409 || response.status === 404 ? response.status : 502,
      `Episode API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<T>;
}

export async function startEpisode(
  workspaceId: string,
  question?: string,
): Promise<EpisodeStatus & { ok: boolean; outcome: MoveOutcome | null }> {
  return episodeFetch("", {
    method: "POST",
    body: JSON.stringify({
      workspace_id: workspaceId,
      ...(question !== undefined ? { question } : {}),
    }),
  });
}

export async function startStudyRecipe(workspaceId: string): Promise<void> {
  await episodeFetch(`/${workspaceId}/recipes/observational-study`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function getEpisodeStatus(workspaceId: string): Promise<EpisodeStatus> {
  return episodeFetch(`/${workspaceId}`);
}

export async function getEpisodeTimeline(
  workspaceId: string,
): Promise<{ workspace_id: string; transitions: TransitionRecord[] }> {
  return episodeFetch(`/${workspaceId}/timeline`);
}

export async function getEpisodeEvents(
  workspaceId: string,
  after?: string | null,
): Promise<{ workspace_id: string; events: RuntimeEvent[] }> {
  const search = after ? `?${new URLSearchParams({ after }).toString()}` : "";
  return episodeFetch(`/${workspaceId}/events${search}`);
}

export async function getOperationTraceIndex(
  workspaceId: string,
  artifactId: string,
): Promise<TransitionTraceIndex> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${workspaceId}/operations/${artifactId}/traces`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 404 ? 404 : 502,
      `Trace API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<TransitionTraceIndex>;
}

export async function getEpisodeTrace(
  workspaceId: string,
  seq: number,
  subroutineId: string,
): Promise<LLMTrace> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${workspaceId}/traces/${seq}/${encodeURIComponent(subroutineId)}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new EpisodeRunError(
      response.status === 404 ? 404 : 502,
      `Trace API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<LLMTrace>;
}

export async function getMachineDescription(): Promise<MachineDescription> {
  const response = await fetch(`${TOOL_SERVER}/api/machine`, { cache: "no-store" });
  if (!response.ok) {
    throw new EpisodeRunError(502, `Machine description error ${response.status}`);
  }
  return response.json() as Promise<MachineDescription>;
}

/**
 * Facade deployment capabilities. A read-only facade (the hosted viewer's
 * backend) reports moves_enabled=false; the UI hides move affordances.
 */
export async function getFacadeCapabilities(): Promise<CapabilitiesResponse> {
  const response = await fetch(`${TOOL_SERVER}/api/capabilities`, { cache: "no-store" });
  if (!response.ok) {
    throw new EpisodeRunError(502, `Capabilities error ${response.status}`);
  }
  return response.json() as Promise<CapabilitiesResponse>;
}

export async function getModelView<K extends ArtifactViewId>(
  workspaceId: string,
  artifactId: K,
): Promise<ArtifactViewData<K>> {
  return episodeFetch(`/${workspaceId}/model/views/${artifactId}`);
}
