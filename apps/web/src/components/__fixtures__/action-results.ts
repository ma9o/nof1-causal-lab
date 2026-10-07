import type { EditModelOutput, FitOutput, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { fixtureValue } from "./fixture-value";

/** Retained model projections shared by isolated action-response fixtures. */
export function modelResult(snapshot: ModelSnapshot): EditModelOutput {
  return {
    model: fixtureValue(snapshot.model),
    checks: {
      specification: snapshot.specification ?? [],
      question: snapshot.question_checks ?? null,
    },
    identification: snapshot.identification,
  };
}

/** Fit fixtures expose only the values owned by fitting. */
export function fitResult(snapshot: ModelSnapshot): FitOutput {
  return {
    model: fixtureValue(snapshot.model),
    checks: {
      validation: fixtureValue(snapshot.validation_report),
      question: fixtureValue(snapshot.question_checks),
    },
    inference: {
      run: {
        model: {
          workspace_id: snapshot.workspace_id,
          revision: snapshot.commit_id,
          path: "result.msgpack",
        },
        data: fixtureValue(snapshot.state.data),
        evidence: {
          chain_extra_fields: {},
          observation_log_probs: null,
          observed_rows: null,
          exact_observation_rows: null,
          phase_extra_fields: {},
          warmup_complete_log_posterior_history: null,
          all_complete_log_posterior_history: null,
          initial_latent_delta: null,
          final_latent_delta: null,
        },
      },
      core: fixtureValue(snapshot.fit),
      detail: { trace_data: [], rank_histograms: [], pareto_k: [], loo_pit: [] },
    },
  };
}
