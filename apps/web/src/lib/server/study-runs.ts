import {
  createModelClient,
  type ActionReceipt,
  type CapabilitiesResponse,
} from "@nof1-causal-lab/api-types";
import { getToolServerUrl } from "@/lib/runtime-urls";

const client = createModelClient({ baseUrl: getToolServerUrl() });

export class StudyRunError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function createStudy(workspaceId: string, question: string): Promise<ActionReceipt> {
  const { data, error, response } = await client.POST("/api/studies/{workspace_id}/actions", {
    params: { path: { workspace_id: workspaceId } },
    body: { action: "edit_model", expected_revision: null, model: { question } },
    cache: "no-store",
  });
  if (data === undefined)
    throw new StudyRunError(
      response.status === 409 || response.status === 404 ? response.status : 502,
      `Study API error ${response.status}: ${JSON.stringify(error)}`,
    );
  return data;
}

/**
 * Facade deployment capabilities. A read-only facade (the hosted viewer's
 * backend) reports actions_enabled=false; the UI hides action controls.
 */
export async function getFacadeCapabilities(): Promise<CapabilitiesResponse> {
  const { data, response } = await client.GET("/api/capabilities", { cache: "no-store" });
  if (data === undefined) throw new StudyRunError(502, `Capabilities error ${response.status}`);
  return data;
}
