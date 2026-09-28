import type {
  ArtifactViewData,
  ArtifactViewId,
  CapabilitiesResponse,
  EpisodeStatus,
  LLMTrace,
  MachineDescription,
  ActionReceipt,
  RuntimeEvent,
  TimelineResponse,
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
  RetractedArtifact,
  RuntimeEvent,
  StudyRevision,
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

export async function createStudy(workspaceId: string, question: string): Promise<ActionReceipt> {
  return episodeFetch<ActionReceipt>(`/${workspaceId}/actions`, {
    method: "POST",
    body: JSON.stringify({ action: "edit_model", expected_revision: null, model: { question } }),
  });
}

export async function getEpisodeStatus(workspaceId: string): Promise<EpisodeStatus> {
  return episodeFetch(`/${workspaceId}`);
}

export async function getEpisodeTimeline(workspaceId: string): Promise<TimelineResponse> {
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
  commitId: string,
  subroutineId: string,
): Promise<LLMTrace> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${workspaceId}/traces/${commitId}/${encodeURIComponent(subroutineId)}`,
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
 * backend) reports actions_enabled=false; the UI hides action controls.
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
