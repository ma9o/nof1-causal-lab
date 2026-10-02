import { createModelClient, type WorkspaceList } from "@nof1-causal-lab/api-types";
import { getToolServerUrl } from "@/lib/runtime-urls";
export type { WorkspaceEntry, WorkspaceList } from "@nof1-causal-lab/api-types";
const client = createModelClient({ baseUrl: getToolServerUrl() });
export async function listWorkspaces(): Promise<WorkspaceList> {
  const { data, error, response } = await client.GET("/api/workspaces", { cache: "no-store" });
  if (data === undefined)
    throw new Error(`Workspace facade error ${response.status}: ${JSON.stringify(error)}`);
  return data;
}
