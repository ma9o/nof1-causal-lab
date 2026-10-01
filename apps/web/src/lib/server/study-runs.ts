import type { ActionReceipt, CapabilitiesResponse, LLMTrace } from "@nof1-causal-lab/api-types";
import { getToolServerUrl } from "@/lib/runtime-urls";

const TOOL_SERVER = getToolServerUrl();

export class StudyRunError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function studyFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${TOOL_SERVER}/api/studies${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new StudyRunError(
      response.status === 409 || response.status === 404 ? response.status : 502,
      `Study API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<T>;
}

export async function createStudy(workspaceId: string, question: string): Promise<ActionReceipt> {
  return studyFetch<ActionReceipt>(`/${workspaceId}/actions`, {
    method: "POST",
    body: JSON.stringify({ action: "edit_model", expected_revision: null, model: { question } }),
  });
}

export async function getStudyTrace(
  workspaceId: string,
  commitId: string,
  subroutineId: string,
): Promise<LLMTrace> {
  const response = await fetch(
    `${TOOL_SERVER}/api/studies/${workspaceId}/traces/${commitId}/${encodeURIComponent(subroutineId)}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new StudyRunError(
      response.status === 404 ? 404 : 502,
      `Trace API error ${response.status}: ${await response.text()}`,
    );
  }
  return response.json() as Promise<LLMTrace>;
}

/**
 * Facade deployment capabilities. A read-only facade (the hosted viewer's
 * backend) reports actions_enabled=false; the UI hides action controls.
 */
export async function getFacadeCapabilities(): Promise<CapabilitiesResponse> {
  const response = await fetch(`${TOOL_SERVER}/api/capabilities`, { cache: "no-store" });
  if (!response.ok) {
    throw new StudyRunError(502, `Capabilities error ${response.status}`);
  }
  return response.json() as Promise<CapabilitiesResponse>;
}
