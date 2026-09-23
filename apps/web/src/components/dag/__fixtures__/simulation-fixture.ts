import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
import { demoModel } from "../../__fixtures__/demo-artifacts";
import simulationTrace from "./simulation-trace.json";
import { buildEdgePosteriors, buildPersistencePosteriors } from "@/lib/dag/simulation-results";
import { constructStatuses } from "@/lib/dag/construct-statuses";
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

const identified = new Set(
  Object.entries(demoModelSnapshot.findings.identification!.value.treatments)
    .filter(([, finding]) => finding.status === "identified")
    .map(([id]) => id),
);
export const identifiableTreatments = constructs
  .filter((item) => identified.has(item.id))
  .map((item) => item.name);
export const nodeStatuses = constructStatuses(demoModelSnapshot);

export const demoSimulationTrace: LLMTrace = simulationTrace as LLMTrace;
