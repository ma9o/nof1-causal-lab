import type { ScenarioClamp, SimulationResult } from "@nof1-causal-lab/api-types";

/** A composable analysis scenario result: a start state + a list of timed latent clamps. */
export type AnalysisSimulationResult = SimulationResult;

/** One do-operator clamp within a scenario. */
export type LatentClamp = ScenarioClamp;
