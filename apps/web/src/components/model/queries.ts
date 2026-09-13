import type {
  ConstructId,
  EffectSummary,
  ModelSnapshot,
  ScenarioQuery,
} from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult } from "@/components/dag/intervention-dag-types";

/**
 * One entry of the model's query collection. Analysis queries carry the full simulate
 * result the machine materialized; ranking queries carry the backend effect summary;
 * saved scenarios preserve their query and any materialized result.
 */
export interface ModelQuery {
  key: string;
  title: string;
  origin: "analysis" | "ranking" | "saved";
  startKind: "baseline" | "abducted" | null;
  horizonDays: number | null;
  outcome: string | null;
  posterior: EffectSummary | null;
  posteriorVersion: number | null;
  simulation: AnalysisSimulationResult | null;
  evaluations: AnalysisSimulationResult[];
  prompt?: string;
  blurb?: string;
  treatmentId?: ConstructId;
  savedQuery?: ScenarioQuery;
}

export function rankingQueryKey(treatment: string): string {
  return `ranking:${treatment}`;
}

export function buildModelQueries(model: ModelSnapshot): ModelQuery[] {
  const constructs = model.latent_structure?.value.constructs ?? [];
  const names = new Map(constructs.map((entity) => [entity.id, entity.name]));
  const outcome =
    constructs.find((entity) => entity.id === model.latent_structure?.value.default_outcome?.id)
      ?.name ?? null;
  const reportPosteriorVersion =
    model.state.current.baseline_report?.derived_from.posterior ?? null;
  const ranking = (model.baseline_report?.value.intervention_results ?? []).flatMap((result) => {
    const summary = result.summary;
    if (summary === null) {
      return [];
    }
    return [
      {
        key: rankingQueryKey(result.treatment_id),
        title: `do(${names.get(result.treatment_id)} +1)`,
        treatmentId: result.treatment_id,
        origin: "ranking" as const,
        startKind: "baseline" as const,
        horizonDays: null,
        outcome,
        posterior: summary,
        simulation: null,
        evaluations: [],
        posteriorVersion: reportPosteriorVersion,
      },
    ];
  });
  const saved = (model.saved_scenarios?.value.scenarios ?? []).map((scenario) => {
    const evaluations = scenario.evaluations.map((item) => ({ query: scenario.query, ...item }));
    const selected =
      evaluations.find(
        (item) =>
          item.evaluation.model.id === model.model.id &&
          item.evaluation.posterior.version === model.state.current.posterior?.version,
      ) ?? evaluations.at(-1);
    return {
      key: scenario.query.id,
      title: scenario.label,
      origin: "saved" as const,
      startKind: scenario.query.start.kind,
      horizonDays: scenario.query.readout.horizon_days,
      outcome: selected?.result.outcome_label ?? names.get(scenario.query.outcome.id) ?? null,
      posterior: selected?.result.summary ?? null,
      simulation: selected ?? null,
      evaluations,
      savedQuery: scenario.query,
      posteriorVersion: selected?.evaluation.posterior.version ?? null,
      blurb: scenario.summary ?? undefined,
    };
  });
  return [...ranking, ...saved];
}
