import { modelConstructs } from "@/lib/model-accessors";
import type {
  ConstructId,
  EffectSummary,
  ModelSnapshot,
  ScenarioRequest,
  SimulationResult,
} from "@nof1-causal-lab/api-types";

/** A request offered by the UI, with an optional response held in this session. */
export interface ModelQuery {
  key: string;
  title: string;
  startKind: "baseline" | "abducted";
  horizonDays: number;
  outcome: string;
  posterior: EffectSummary | null;
  modelVersion: number | null;
  simulation: SimulationResult | null;
  request: ScenarioRequest;
  treatmentId: ConstructId;
}

export function interventionQueryKey(treatment: ConstructId, outcome: ConstructId): string {
  return `intervention:${treatment}:${outcome}`;
}

export function buildModelQueries(
  model: ModelSnapshot,
  responses: Record<string, SimulationResult> = {},
): ModelQuery[] {
  const identification = model.findings.identification?.value;
  if (!model.findings.fit || !identification?.outcome) return [];
  const outcomeId = identification.outcome;
  const names = new Map(modelConstructs(model.model?.value).map((item) => [item.id, item.name]));
  const treatments = (Object.keys(identification.treatments) as ConstructId[]).filter(
    (id) => identification.treatments[id].status === "identified",
  );
  return treatments.map<ModelQuery>((treatmentId) => {
    const key = interventionQueryKey(treatmentId, outcomeId);
    const response = responses[key];
    const simulation =
      response?.model.workspace_id === model.context.workspace_id ? response : null;
    // These are query defaults; identification and all effect calculations
    // are performed by the backend.
    const request: ScenarioRequest = {
      start: { kind: "baseline" },
      clamps: [{ target: treatmentId, mode: "shift", amount: 1, from_day: 0 }],
      outcome: outcomeId,
      readout: { estimand: "trajectory", horizon_days: 30, projection: "latent" },
    };
    return {
      key,
      title: `do(${names.get(treatmentId)} +1)`,
      treatmentId,
      startKind: "baseline",
      horizonDays: 30,
      outcome: names.get(outcomeId)!,
      posterior: simulation?.summary ?? null,
      modelVersion: simulation?.model.version ?? null,
      simulation,
      request,
    };
  });
}
