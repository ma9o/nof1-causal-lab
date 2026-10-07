import type { EditModelOutput, FitOutput, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { fixtureValue } from "./fixture-value";

/** Retained model projections shared by isolated action-response fixtures. */
export function modelResult(snapshot: ModelSnapshot): EditModelOutput {
  return {
    model: fixtureValue(snapshot.model),
    arrays: {},
    can_simulate: snapshot.can_simulate,
    checks: {
      specification: snapshot.specification ?? [],
      question: snapshot.question_checks ?? null,
    },
    graph: snapshot.graph,
    identification: snapshot.identification,
    dispositions: snapshot.dispositions,
    entity_failures: snapshot.entity_failures,
    confounder_equations: snapshot.confounder_equations,
    state_equations: snapshot.state_equations,
    observation_equations: snapshot.observation_equations,
    authoring_prior_densities: snapshot.authoring_prior_densities,
  };
}

/** Fit fixtures expose only the values owned by fitting. */
export function fitResult(snapshot: ModelSnapshot): FitOutput {
  return {
    model: fixtureValue(snapshot.model),
    inference: null,
    entity_failures: {},
    validation_report: snapshot.validation_report,
    question_checks: snapshot.question_checks,
    likelihood_diagnostics: snapshot.likelihood_diagnostics,
    edge_estimates: snapshot.fit?.edge_estimates ?? {},
    decay_estimates: snapshot.fit?.decay_estimates ?? {},
    prior_densities: snapshot.fit?.prior_densities ?? {},
    inference_report: snapshot.fit
      ? {
          core: snapshot.fit.report,
          detail: {
            trace_data: [],
            rank_histograms: [],
            pareto_k: [],
            loo_pit: [],
            divergent: null,
            initial_latent_delta: null,
            final_latent_delta: null,
          },
        }
      : null,
    parameter_draws: {
      kind: "unavailable",
      reason: "This illustrative fit retains no joint draws",
    },
    arrays: {},
  };
}
