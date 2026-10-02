import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { ModelSnapshot } from "@nof1-causal-lab/api-types";
import history from "../../../../../data/DEMO/fixture/model_history.json";
import snapshot from "../../../../../data/DEMO/fixture/model_snapshot.json";

/** Validated by the production reader when read fixtures are generated. */
export const demoModelSnapshot = snapshot;
export const demoModel = fixtureValue(demoModelSnapshot.model).value;
export const demoParameters = demoModel.parameters;
export const demoRawData = fixtureValue(demoModelSnapshot.data.raw_data).value;
export const demoMeasurements = fixtureValue(demoModelSnapshot.data.measurements).value;
export const demoValidationReport = fixtureValue(
  demoModelSnapshot.findings.validation_report,
).value;
export const demoModelDiagnostics = fixtureValue(demoModelSnapshot.findings.diagnostics);
export const demoPosterior = fixtureValue(demoModelSnapshot.findings.fit).value.report;

export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === demoModelSnapshot.context.seq) return demoModelSnapshot;
  const revision = history[seq];
  if (!revision) throw new Error(`No committed DEMO revision ${seq}`);
  return revision;
}
