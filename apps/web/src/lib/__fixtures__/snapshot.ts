/** Explicit served projections for unit tests; no saved study or numerical execution. */
import type { FitSummary, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { baseline, decay, diffusion, modelFixture, outcome, treatment } from "./model";

export const modelRef = { workspace_id: "TEST", revision: "1".repeat(40), path: "model.json" };

export const emptySnapshot: ModelSnapshot = {
  workspace_id: "TEST",
  commit_id: "0".repeat(40),
  selected_seq: 0,
  can_simulate: false,
  question: null,
  model: null,
  state: { current: {}, data: null },
  raw_data: null,
  measurements: null,
  metadata: null,
  profile: null,
  identification: null,
  dispositions: null,
  graph: { construct_ids: [], edge_ids: [], dynamic_construct_ids: [], status: {} },
  entity_failures: {},
  validation_report: null,
  confounder_equations: {},
  state_equations: {},
  observation_equations: {},
  likelihood_diagnostics: {},
  authoring_prior_densities: {},
  fit: null,
  specification: null,
  question_checks: null,
  simulation: null,
  predictive: null,
};

const density = { x: [0, 0.5, 1], density: [0, 2, 0] };
export const authoredSnapshot: ModelSnapshot = {
  ...emptySnapshot,
  commit_id: "3".repeat(40),
  selected_seq: 3,
  model: modelFixture,
  question: { text: "How does treatment change the outcome?", outcome: outcome.id, queries: {} },
  state: {
    current: {
      question: {
        artifact_id: "question",
        revision: "2".repeat(40),
        derived_from: {},
        produced_by: "edit_question",
        created_at: "2026-01-01T00:00:00Z",
      },
      model: {
        artifact_id: "model",
        revision: modelRef.revision,
        derived_from: { question: "2".repeat(40) },
        produced_by: "edit_model",
        created_at: "2026-01-01T00:00:00Z",
      },
    },
    data: null,
  },
  dispositions: [
    {
      target: { kind: "construct", id: outcome.id },
      disposition: "retained_state",
      reason: "Observed response.",
    },
  ],
  graph: {
    construct_ids: [baseline.id, treatment.id, outcome.id],
    edge_ids: ["edge:00000000000000000001", "edge:00000000000000000002"],
    dynamic_construct_ids: [treatment.id, outcome.id],
    status: {},
  },
  authoring_prior_densities: { [decay.id]: density, [diffusion.id]: density },
};

const fit: FitSummary = {
  report: {
    time_origin: null,
    inference_metadata: { n_samples: 3, duration_seconds: 0 },
    engine: {
      kind: "not_evaluated",
      subject: "particle_mcmc",
      reason: "STATE_NOT_RECORDED",
      detail: "Recorded presentation data.",
    },
    inference_diagnostics: null,
    sampler_diagnostics: null,
    convergence: { assessments: [], checked: 0, status: "not_evaluated", messages: [] },
    loo_diagnostics: null,
    posterior_marginals: [decay, diffusion].map((parameter) => ({
      parameter: parameter.name,
      subject: {
        parameter_id: parameter.id,
        element_id: `element:${parameter.id.slice("parameter:".length)}` as const,
      },
      density_curve: density,
      mean: 0.5,
      lower: 0.1,
      upper: 0.9,
      interval_kind: "hdi" as const,
      interval_mass: 0.9,
      sd: 0.2,
    })),
  },
  edge_estimates: {},
  decay_estimates: {},
  prior_densities: { [decay.id]: density, [diffusion.id]: density },
};
export const fittedSnapshot: ModelSnapshot = {
  ...authoredSnapshot,
  commit_id: "4".repeat(40),
  selected_seq: 4,
  fit: fit,
};
