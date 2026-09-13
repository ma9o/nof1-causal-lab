import type { PosteriorArtifact } from "@nof1-causal-lab/api-types";
import { demoPosterior } from "./demo-artifacts";

export const posterior = demoPosterior as PosteriorArtifact;
export const posteriorAuxKalmanMCMC = demoPosterior as PosteriorArtifact;
