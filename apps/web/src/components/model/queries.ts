import { modelConstructs } from "@/lib/model-accessors";
import type {
  ConstructId,
  EffectSummary,
  ModelSnapshot,
  ScenarioRequest,
} from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult } from "@/components/dag/intervention-dag-types";
import { formatScenarioActionDescription } from "@/components/dag/intervention-dag-semantics";

/** Queries available from the report's ranking and explicitly retained simulations. */
export interface ModelQuery {
  key: string;
  title: string;
  origin: "analysis" | "ranking";
  startKind: "baseline" | "abducted" | null;
  horizonDays: number | null;
  outcome: string | null;
  posterior: EffectSummary | null;
  modelVersion: number | null;
  simulation: AnalysisSimulationResult | null;
  request?: ScenarioRequest;
  treatmentId?: ConstructId;
}

export function rankingQueryKey(treatment: string): string {
  return `ranking:${treatment}`;
}

export function buildModelQueries(model: ModelSnapshot): ModelQuery[] {
  const constructs = modelConstructs(model.model?.value) ?? [];
  const names = new Map(constructs.map((entity) => [entity.id, entity.name]));
  const outcome =
    constructs.find((entity) => entity.id === model.model?.value.default_outcome?.id)?.name ?? null;
  const reportModelVersion =
    model.context.state.current.baseline_report?.derived_from.model ?? null;
  const report = model.findings.baseline_report?.value;
  const ranking: ModelQuery[] = (report?.intervention_results ?? []).flatMap((result) =>
    result.summary === null
      ? []
      : [
          {
            key: rankingQueryKey(result.treatment_id),
            title: `do(${names.get(result.treatment_id)} +1)`,
            treatmentId: result.treatment_id,
            origin: "ranking",
            startKind: "baseline",
            horizonDays: null,
            outcome,
            posterior: result.summary,
            simulation: null,
            modelVersion: reportModelVersion,
          },
        ],
  );
  const retained: ModelQuery[] = (report?.simulation_results ?? []).map((result, index) => ({
    key: `report:${index}`,
    title: formatScenarioActionDescription(result),
    origin: "analysis",
    startKind: result.request.start.kind,
    horizonDays: result.request.readout.horizon_days,
    outcome: result.labels[result.request.outcome.id],
    posterior: result.summary,
    simulation: result,
    request: result.request,
    modelVersion: result.provenance.model.version,
  }));
  return [...ranking, ...retained];
}
