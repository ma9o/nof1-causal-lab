import { modelConstructs } from "@/lib/model-accessors";
import {
  demoModel,
  demoModelSnapshot,
  demoModelDiagnostics,
  demoAdmissionReport,
} from "../../../__fixtures__/demo-artifacts";
export const model = demoModel;
export const modelSpecData = demoModelSnapshot;
export const observationEquations = demoModelDiagnostics.observation_equations;
export const equations = demoModelDiagnostics.state_equations;
export const confounderEquations = demoModelDiagnostics.confounder_equations;
export const parameters = demoModel.parameters;
export const priorDensities = demoModelDiagnostics.prior_densities;
export const indicators = modelConstructs(demoModel).flatMap((construct) => construct.indicators);
export const likelihoodDiagnostics = demoModelDiagnostics.likelihood_diagnostics;
export const priorPredictiveSamples = demoAdmissionReport.prior_predictive_samples;
export const constructs = modelConstructs(demoModel);
