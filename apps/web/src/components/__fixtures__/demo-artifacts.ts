import type { ModelSnapshot, ArtifactViews } from "@nof1-causal-lab/api-types";
import history from "../../../../../data/DEMO/fixture/model_history.json";
import views from "../../../../../data/DEMO/fixture/artifact_views.json";
import snapshot from "../../../../../data/DEMO/fixture/model_snapshot.json";

/** Validated by the production snapshot reader during fixture generation. */
export const demoModelSnapshot = snapshot as unknown as ModelSnapshot;
const artifactViews = views as unknown as ArtifactViews;
export const demoParameters = demoModelSnapshot.compiled_parameters!.value;
export const demoRawData = artifactViews.raw_data!;
export const demoLatentStructure = artifactViews.latent_structure!;
export const demoMeasurementStructure = artifactViews.measurement_structure!;
export const demoMeasurements = artifactViews.measurements!;
export const demoValidationReport = artifactViews.validation_report!;
export const demoStatisticalModelSpec = artifactViews.statistical_model_spec!;
export const demoPosterior = artifactViews.posterior!;
export const demoBaselineReport = artifactViews.baseline_report!;

/** Actual backend projections for each committed fixture revision. */
export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === 10) return demoModelSnapshot;
  const revision = (history as unknown as Record<number, ModelSnapshot>)[seq];
  if (!revision) throw new Error(`No committed DEMO revision ${seq}`);
  return revision;
}
