import type {
  ActionSuccess,
  EditModelOutput,
  ModelSnapshot,
  PrepareDataOutput,
  RecordDependency,
  TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { questionCall } from "./call-dependencies";

export function callModel(result: ActionSuccess | undefined): EditModelOutput | null {
  return result?.action === "edit_model"
    ? result.body
    : result?.action === "fit"
      ? result.body.model
      : null;
}

interface ViewParts {
  artifacts: ModelSnapshot["state"]["current"];
  question: ModelSnapshot["question"];
  model: EditModelOutput | null;
  prepared: PrepareDataOutput | null;
  data: ModelSnapshot["state"]["data"];
  fit: ModelSnapshot["fit"];
  simulation: ModelSnapshot["simulation"];
}

const empty: ViewParts = {
  artifacts: {},
  question: null,
  model: null,
  prepared: null,
  data: null,
  fit: null,
  simulation: null,
};

/** Select already computed outputs by their recorded roles; never reconstruct science here. */
export function composeCallView(
  workspaceId: string,
  selected: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
  results: ReadonlyMap<number, ActionSuccess>,
): ModelSnapshot {
  const bySeq = new Map(attempts.map((entry) => [entry.record.seq, entry]));
  const views = new Map<number, ViewParts>();

  function inputCall(entry: TimelineRevision, argument: string): TimelineRevision {
    const dependency = dependencies.find(
      (item) => item.seq === entry.record.seq && item.argument === argument,
    );
    if (!dependency) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
    const parent = bySeq.get(dependency.source_seq);
    if (!parent) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
    return parent;
  }

  function input(entry: TimelineRevision, argument: string): ViewParts {
    return parts(inputCall(entry, argument));
  }

  function parts(entry: TimelineRevision): ViewParts {
    const seq = entry.record.seq;
    const cached = views.get(seq);
    if (cached) return cached;
    const result = results.get(seq);
    if (!result) throw new Error(`Missing saved result for call ${seq}`);
    const request = entry.record.attempt.request;
    const outcome = entry.record.attempt.outcome;
    const produced: ModelSnapshot["state"]["current"] = Object.fromEntries(
      outcome.status === "applied"
        ? outcome.effects.produced.map((artifact) => [artifact.artifact_id, artifact])
        : [],
    );
    let view: ViewParts;
    switch (result.action) {
      case "edit_question":
        view = { ...empty, question: result.body.question, artifacts: produced };
        break;
      case "edit_model": {
        const question = parts(questionCall(entry, attempts, dependencies));
        view = {
          ...empty,
          question: question.question,
          artifacts: { ...question.artifacts, ...produced },
          model: result.body,
        };
        break;
      }
      case "prepare_data": {
        const model = input(entry, "model");
        const panel = produced.panel;
        view = {
          ...model,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(model.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
            ...produced,
          },
          prepared: result.body,
          data: panel ? { revision: panel.revision, replicate_index: 0 } : null,
          simulation: null,
        };
        break;
      }
      case "fit": {
        if (request?.action !== "fit") throw new Error("Saved fit is missing its input references");
        const dataCall = inputCall(entry, "data");
        const data = parts(dataCall);
        const model = input(entry, "model");
        view = {
          ...empty,
          question: model.question,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(model.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
            ...(dataCall.record.attempt.action === "prepare_data"
              ? Object.fromEntries(
                  Object.entries(data.artifacts).filter(
                    ([key]) => key === "panel" || key === "raw_data",
                  ),
                )
              : {}),
            ...produced,
          },
          model: result.body.model,
          prepared: dataCall.record.attempt.action === "prepare_data" ? data.prepared : null,
          data: {
            revision: request.input.data_ref,
            replicate_index: request.input.replicate_index,
          },
          fit: result.body.summary,
        };
        break;
      }
      case "simulate": {
        if (request?.action !== "simulate")
          throw new Error("Saved simulation is missing its input references");
        const panel = request.input.panel_ref === null ? empty : input(entry, "panel");
        const model = input(entry, "model");
        view = {
          ...model,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(model.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
            ...Object.fromEntries(
              Object.entries(panel.artifacts).filter(
                ([key]) => key === "panel" || key === "raw_data",
              ),
            ),
          },
          prepared: panel.prepared,
          data: panel.data,
          simulation: result.body.report,
        };
        break;
      }
      case "model_diff":
        view = input(entry, "after");
        break;
      case "data_diff":
        view = input(entry, "right");
        break;
    }
    views.set(seq, view);
    return view;
  }

  const view = parts(selected);
  const model = view.model;
  const prepared = view.prepared;
  return {
    workspace_id: workspaceId,
    commit_id: selected.commit_id,
    selected_seq: selected.record.seq,
    question: view.question,
    state: {
      current: view.artifacts,
      data: view.data,
    },
    model: model?.model ?? null,
    can_simulate: model?.can_simulate ?? false,
    graph: model?.graph ?? {
      construct_ids: [],
      edge_ids: [],
      dynamic_construct_ids: [],
      status: {},
    },
    raw_data: prepared?.raw_data ?? null,
    measurements: prepared?.measurements ?? null,
    metadata: prepared?.metadata ?? null,
    profile: prepared?.profile ?? null,
    identification: model?.identification ?? null,
    dispositions: model?.dispositions ?? null,
    entity_failures: model?.entity_failures ?? {},
    validation_report: model?.validation_report ?? null,
    specification: model?.specification ?? null,
    question_checks: model?.question_checks ?? null,
    predictive: model?.predictive ?? null,
    confounder_equations: model?.confounder_equations ?? {},
    state_equations: model?.state_equations ?? {},
    observation_equations: model?.observation_equations ?? {},
    likelihood_diagnostics: model?.likelihood_diagnostics ?? {},
    authoring_prior_densities: model?.authoring_prior_densities ?? {},
    fit: view.fit,
    simulation: view.simulation,
  };
}
