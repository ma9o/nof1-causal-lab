import type { BaselineReportArtifact, LLMTrace } from "@nof1-causal-lab/api-types";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { demoBaselineReport, demoModel } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";
import {
  buildEdgePosteriors,
  buildPersistencePosteriors,
} from "../../pipeline/output-views/baseline-report-scenarios";
import { constructStatuses } from "../construct-statuses";
import { constructs, edges, indicators } from "./dag-base-fixtures";

export { constructs, edges, indicators };

export const edgePosteriors = buildEdgePosteriors({
  latentStructure: demoModel,
  estimates: demoModelSnapshot.findings.fit!.value.edge_estimates,
});
export const persistencePosteriors = buildPersistencePosteriors({
  latentStructure: demoModel,
  estimates: demoModelSnapshot.findings.fit!.value.decay_estimates,
});

const demo = demoBaselineReport as BaselineReportArtifact;
export const identifiableTreatments = demo.intervention_results.map(({ treatment }) => treatment);
export const nodeStatuses = constructStatuses(demoModelSnapshot);

export const demoBaselineTrace: LLMTrace = demoTraces.baseline_report;

/** The complete materialized analysis artifact (rankings, scenarios, and summary). */
export const materializedBaselineReportData: BaselineReportArtifact = {
  intervention_results: demo.intervention_results,
  simulation_results: demo.simulation_results,
  final_summary: demo.final_summary ?? null,
};
