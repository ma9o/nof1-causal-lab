import { createModelClient, type SimulationReport } from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult } from "@/lib/dag/simulation-types";
import type { SimulateFn } from "@/lib/dag/simulate-input";

/** Submit an identified scenario through the same durable simulate action. */
export function createSimulateDispatch(workspaceId: string, modelVersion: number): SimulateFn {
  return async (input): Promise<AnalysisSimulationResult> => {
    const { data, error } = await createModelClient().POST("/api/episodes/{workspace_id}/actions", {
      params: { path: { workspace_id: workspaceId } },
      body: {
        action: "simulate",
        model_version: modelVersion,
        design: { kind: "causal", query: input },
      },
    });
    if (error || !data || data.status !== "applied") throw new Error(JSON.stringify(error ?? data));
    const report = data.diagnostics.report as unknown as SimulationReport;
    if (!report.causal_result)
      throw new Error("Simulation did not return a certified causal result");
    return report.causal_result;
  };
}
