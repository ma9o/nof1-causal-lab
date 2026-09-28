import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import history from "../../../../../data/DEMO/fixture/model_history.json";
import snapshot from "../../../../../data/DEMO/fixture/model_snapshot.json";

/** Validated by the production reader when read fixtures are generated. */
export const demoModelSnapshot = snapshot as unknown as ModelSnapshot;
export const demoModel = demoModelSnapshot.model!.value;
export const demoParameters = demoModel.parameters;
export const demoRawData = demoModelSnapshot.data.raw_data!.value;
export const demoMeasurements = demoModelSnapshot.data.measurements!.value;
export const demoValidationReport = demoModelSnapshot.findings.validation_report!.value;
export const demoModelDiagnostics = demoModelSnapshot.findings.diagnostics!;
export const demoPriorPredictive = demoModelSnapshot.findings.prior_predictive!.value;
export const demoPosterior = demoModelSnapshot.findings.fit!.value.report;

export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === demoModelSnapshot.context.seq) return demoModelSnapshot;
  const revision = (history as unknown as Record<number, ModelSnapshot>)[seq];
  if (!revision) throw new Error(`No committed DEMO revision ${seq}`);
  return revision;
}
