import type { InferenceReport, PosteriorPredictiveChecks } from "@nof1-causal-lab/api-types";
import { demoPosterior } from "./demo-artifacts";

export const posterior = demoPosterior as InferenceReport;

import retainedPredictiveChecks from "../../../../../data/DEMO/fixture/predictive_checks.json";

// Retained illustrative measurements; the original simulation draws were not retained.
export const predictiveChecks = retainedPredictiveChecks as PosteriorPredictiveChecks;
