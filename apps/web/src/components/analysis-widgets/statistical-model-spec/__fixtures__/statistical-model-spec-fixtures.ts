import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import { collectModelSpecUiPriors } from "@/lib/model-spec-data";
import {
  demoMeasurementStructure,
  demoStatisticalModelSpec,
} from "../../../__fixtures__/demo-artifacts";

const measurementStructure = demoMeasurementStructure as MeasurementStructureViewData;

export const modelSpecData = demoStatisticalModelSpec;

export const equations = modelSpecData.state_equations;

export const likelihoods = modelSpecData.statistical_model_spec.likelihoods;
export const parameters = modelSpecData.parameters;
export const priors = collectModelSpecUiPriors(modelSpecData);
export const indicators = measurementStructure.causal_design.measurement.indicators;
export const likelihoodDiagnostics = modelSpecData.likelihood_diagnostics;
export const priorPredictiveSamples = modelSpecData.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;

export const structuralPlan = modelSpecData.structural_plan!;
export const constructs = Object.values(structuralPlan.semantics.constructs);
