import type { BaselineReportArtifact, LLMTrace } from "@nof1-causal-lab/api-types";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { demoBaselineReport, demoLatentStructure } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";
import {
  buildEdgePosteriors,
  buildPersistencePosteriors,
} from "../../pipeline/output-views/baseline-report-scenarios";
import { deriveConstructStatuses } from "../construct-statuses";
import {
  constructs,
  design,
  edges,
  indicators,
  knownInputs,
  structuralPlan,
} from "./dag-base-fixtures";

export { constructs, edges, indicators, knownInputs };

export const edgePosteriors = buildEdgePosteriors({
  latentStructure: demoLatentStructure,
  estimates: demoModelSnapshot.fit!.value.edge_estimates,
});
export const persistencePosteriors = buildPersistencePosteriors({
  latentStructure: demoLatentStructure,
  estimates: demoModelSnapshot.fit!.value.decay_estimates,
});

const demo = demoBaselineReport as BaselineReportArtifact;
export const identifiableTreatments = demo.intervention_results.map(({ treatment }) => treatment);
export const nodeStatuses = deriveConstructStatuses(design, structuralPlan);

export const demoBaselineTrace: LLMTrace = demoTraces.baseline_report;

/** The complete materialized analysis artifact (rankings, scenarios, and summary). */
export const materializedBaselineReportData: BaselineReportArtifact = {
  intervention_results: demo.intervention_results,
  saved_scenarios: demo.saved_scenarios ?? null,
  final_summary: demo.final_summary ?? null,
};
