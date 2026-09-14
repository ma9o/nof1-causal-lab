import type { ModelSnapshot, ArtifactViews } from "@nof1-causal-lab/api-types";
import history from "../../../../../data/DEMO/fixture/model_history.json";
import views from "../../../../../data/DEMO/fixture/artifact_views.json";
import snapshot from "../../../../../data/DEMO/fixture/model_snapshot.json";

/** Validated by the production reader when read fixtures are generated. */
export const demoModelSnapshot = snapshot as unknown as ModelSnapshot;
const artifactViews = views as unknown as ArtifactViews;
export const demoModel = demoModelSnapshot.model!.value;
export const demoParameters = demoModel.parameters;
export const demoRawData = artifactViews.raw_data!;
export const demoMeasurements = artifactViews.measurements!;
export const demoValidationReport = artifactViews.validation_report!;
export const demoModelDiagnostics = artifactViews.model_diagnostics!;
export const demoPriorPredictive = artifactViews.prior_predictive!;
export const demoPosterior = artifactViews.inference_report!;

export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === demoModelSnapshot.context.seq) return demoModelSnapshot;
  const revision = (history as unknown as Record<number, ModelSnapshot>)[seq];
  if (!revision) throw new Error(`No committed DEMO revision ${seq}`);
  return revision;
}
