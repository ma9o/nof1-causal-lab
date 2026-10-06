import type { EditModelOutput, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { fixtureValue } from "./fixture-value";

/** Retained model projections shared by isolated action-response fixtures. */
export function modelResult(snapshot: ModelSnapshot): EditModelOutput {
  return {
    model: fixtureValue(snapshot.model),
    can_simulate: snapshot.can_simulate,
    checks: {
      specification: snapshot.specification ?? [],
      question: snapshot.question_checks ?? null,
      predictive: snapshot.predictive ?? null,
      reused: [],
    },
    graph: snapshot.graph,
    identification: snapshot.identification,
    dispositions: snapshot.dispositions,
    entity_failures: snapshot.entity_failures,
    validation_report: snapshot.validation_report,
    specification: snapshot.specification,
    question_checks: snapshot.question_checks,
    predictive: snapshot.predictive,
    predictive_overlays: {},
    confounder_equations: snapshot.confounder_equations,
    state_equations: snapshot.state_equations,
    observation_equations: snapshot.observation_equations,
    likelihood_diagnostics: snapshot.likelihood_diagnostics,
    authoring_prior_densities: snapshot.authoring_prior_densities,
  };
}
