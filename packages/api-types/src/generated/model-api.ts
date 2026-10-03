/** AUTO-GENERATED from FastAPI OpenAPI. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
    readonly "/api/studies/{workspace_id}/actions": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Execute Scientific Action
         * @description Accept durable work and return its receipt; retrieve results by polling the attempt.
         */
        readonly post: operations["execute_scientific_action_api_studies__workspace_id__actions_post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/actions/{attempt_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Poll Scientific Action
         * @description Read progress or the completed attempt's typed outcome without dispatching work.
         */
        readonly get: operations["poll_scientific_action_api_studies__workspace_id__actions__attempt_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Study
         * @description Current study state: the single read to poll while navigating.
         *
         *     Returns the five scientific action names and per-artifact existence,
         *     freshness and revision from the selected Git branch snapshot, and the
         *     attempt the study's Temporal workflow is executing on any branch, if any.
         */
        readonly get: operations["get_study_api_studies__workspace_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Snapshot
         * @description Batch canonical aggregates in one committed read transaction.
         *
         *     Omit `at` for the selected branch head, or pass an exact Git commit ID.
         *     Use `commit_id` to pin subsequent reads. Failed attempts retain logs without advancing scientific state.
         */
        readonly get: operations["get_model_snapshot_api_studies__workspace_id__model_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/revisions": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Revisions
         * @description List stored model, observation and source revisions for deliberate selection.
         */
        readonly get: operations["get_revisions_api_studies__workspace_id__revisions_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/revisions/model/{revision}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Read Model Revision
         * @description Read a historical definition, including the input to an earlier fit.
         */
        readonly get: operations["read_model_revision_api_studies__workspace_id__revisions_model__revision__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model-diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Diff
         * @description Compare two model artifact revisions or Git checkpoints containing a model.
         *
         *     Returns identity-aligned definition changes, parameter decisions and graph
         *     topology differences. Graph highlights exclude laws and other entity attributes.
         *     Checkpoint selections also include their recorded fit/simulation
         *     evidence; selecting a model tree alone does not infer an associated run.
         */
        readonly get: operations["model_diff"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/data-diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Post Data Diff
         * @description Record a comparison as a read-only leaf off the branch head captured at dispatch.
         *
         *     Each side accepts a data reference or a nonempty array of references. Panel
         *     references select artifact revisions; simulation references select applied
         *     simulation commits and optionally one replicate (otherwise every draw).
         *     Simulation calendar coordinates come from the saved report's origin.
         *     Exact anchors and measurement windows determine which predictive comparisons
         *     are available. Results preserve each history and report incompatible inputs.
         *     Poll the returned attempt_id for the report and its commit_id. The study's
         *     serialized writer saves the report beside the journal record; it never moves
         *     the branch or changes scientific state. Failed comparisons also leave a leaf.
         *     GET /data-diff/{commit_id} reads a saved report without running the comparison.
         */
        readonly post: operations["data_diff"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/data-diff/{commit_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Data Diff
         * @description Read the comparison report retained by its applied outcome.
         */
        readonly get: operations["get_data_diff_api_studies__workspace_id__data_diff__commit_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/revisions/data-profile/{panel_revision}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Read Data Profile
         * @description Read the empirical profile for an observation revision independently of the model.
         */
        readonly get: operations["read_data_profile_api_studies__workspace_id__revisions_data_profile__panel_revision__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/definition": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Definition
         * @description The canonical scientific value selected by this journal revision.
         */
        readonly get: operations["get_model_definition_api_studies__workspace_id__model_definition_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/inference-report": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Inference Report
         * @description Read the fit report associated with the selected model revision.
         */
        readonly get: operations["get_model_inference_report_api_studies__workspace_id__model_inference_report_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/visuals/observations/{indicator_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Observation History
         * @description All prepared observations on their recorded temporal support.
         */
        readonly get: operations["get_observation_history_api_studies__workspace_id__model_visuals_observations__indicator_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/visuals/predictive/{indicator_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Predictive History
         * @description Saved predictive paths on the exact schedule of their pinned inputs.
         */
        readonly get: operations["get_predictive_history_api_studies__workspace_id__model_visuals_predictive__indicator_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/visuals/simulation": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Simulation Paths
         * @description A contiguous page of original simulation draws, without time thinning.
         */
        readonly get: operations["get_simulation_paths_api_studies__workspace_id__model_visuals_simulation_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/visuals/parameters": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Parameter Draws
         * @description All coordinates and all draws of the retained joint posterior.
         */
        readonly get: operations["get_parameter_draws_api_studies__workspace_id__model_visuals_parameters_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/visuals/mechanism": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Get Mechanism Curves
         * @description Read conditional drift curves using the exact model equations; creates no scientific action.
         */
        readonly post: operations["get_mechanism_curves_api_studies__workspace_id__model_visuals_mechanism_post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/constructs": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Constructs
         * @description Authored constructs, using their canonical domain type.
         */
        readonly get: operations["get_model_constructs_api_studies__workspace_id__model_constructs_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/edges": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Edges
         * @description Authored edges, using their canonical domain type.
         */
        readonly get: operations["get_model_edges_api_studies__workspace_id__model_edges_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/indicators": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Indicators
         * @description Authored indicators whose owners survive at the selected revision.
         */
        readonly get: operations["get_model_indicators_api_studies__workspace_id__model_indicators_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model/parameters": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Model Parameters
         * @description Scientific parameter definitions from the selected model, without inference execution.
         */
        readonly get: operations["get_model_parameters_api_studies__workspace_id__model_parameters_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/timeline": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Timeline
         * @description The attempt journal: every action attempt in order.
         *
         *     Each record is `applied` (completed; data_diff leaves do not advance state),
         *     `rejected` (rejected action, state unchanged), or `raised` (the action ran but threw — the record carries the
         *     typed error). Re-running after a `raised`/`rejected` is just proposing the
         *     action again. `dependencies` links each record to the earlier records whose
         *     outputs its request named; `check` marks outputs only its checks read.
         */
        readonly get: operations["get_timeline_api_studies__workspace_id__timeline_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/branches": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /** Get Branches */
        readonly get: operations["get_branches_api_studies__workspace_id__branches_get"];
        readonly put?: never;
        /**
         * Create Branch
         * @description Fork the complete study at a checkpoint; the new branch shares its ancestry.
         */
        readonly post: operations["create_branch_api_studies__workspace_id__branches_post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/logs/{commit_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /** Get Attempt Log */
        readonly get: operations["get_attempt_log_api_studies__workspace_id__logs__commit_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/events": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Events
         * @description Live progress of one attempt: data-preparation step status and extraction fan-out.
         *
         *     Pass the last-seen event cursor as `after` to page forward. Progress is disposable
         *     and never saved with the attempt; its record and traces are authoritative.
         */
        readonly get: operations["get_events_api_studies__workspace_id__events_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/artifacts/{artifact_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Artifact
         * @description One artifact revision: meta + inline JSON payloads.
         *
         *     Defaults to the selected branch's current revision. Binary payload files (parquet, pickle) are listed by name, never
         *     inlined.
         */
        readonly get: operations["get_artifact_api_studies__workspace_id__artifacts__artifact_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/artifacts/{artifact_id}/traces": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Artifact Traces
         * @description Traces of the applied attempt that produced an artifact revision.
         *
         *     Defaults to the study's current revision. The join runs over the
         *     attempt journal, so it works against a published read-only store.
         */
        readonly get: operations["get_artifact_traces_api_studies__workspace_id__artifacts__artifact_id__traces_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/traces/{commit_id}/{subroutine_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Trace
         * @description One trace from the owning attempt's Git commit.
         */
        readonly get: operations["get_trace_api_studies__workspace_id__traces__commit_id___subroutine_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/artifacts/{artifact_id}/files/{filename}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Artifact File
         * @description One declared payload file from an artifact revision.
         *
         *     Defaults to the study's current revision. Unlike the JSON artifact
         *     endpoint, this serves binary files as bytes and refuses undeclared
         *     filenames so callers cannot browse arbitrary workspace paths.
         */
        readonly get: operations["get_artifact_file_api_studies__workspace_id__artifacts__artifact_id__files__filename__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/actions-enabled": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Actions Enabled
         * @description Whether this deployment serves scientific actions.
         *
         *     `actions_enabled` is `false` on the hosted read-only viewer backend, where
         *     every `POST` (scientific actions and study management) returns 403 and only the read
         *     endpoints are live.
         */
        readonly get: operations["get_actions_enabled_api_actions_enabled_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/workspaces": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * List Workspaces
         * @description Published/local workspaces visible through this facade.
         */
        readonly get: operations["list_workspaces_api_workspaces_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/upload": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Upload File
         * @description Stage one raw input file for prepare_data.
         */
        readonly post: operations["upload_file_api_upload_post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/tools/{context_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Tool Schemas
         * @description List a context's validation/query tools — the same tools the in-service LLM loops use.
         *
         *     Each entry is `{name, description, parameters, result}` where `parameters`
         *     and `result` are JSON Schemas. Fetch this first to learn a tool's argument
         *     shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
         *     analysis `simulate` / `get_model_info`, literature `search_literature`.
         */
        readonly get: operations["get_tool_schemas_api_tools__context_id__get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/tools/{context_id}/{tool_name}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Execute Tool
         * @description Execute a context tool against the workspace's current artifact-store versions.
         *
         *     Body is `{"workspace_id": "...", "input": {...}}` where `input` matches the
         *     tool's `parameters` schema from `GET /api/tools/{context_id}`; 422 on a schema
         *     violation. Analysis tools reject stale supporting inputs with 409 before
         *     loading or reusing a fitted context.
         */
        readonly post: operations["execute_tool_api_tools__context_id___tool_name__post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        readonly ActionAttempt: Domain.ActionAttempt;
        /**
         * ActionEffects
         * @description What an executed action did to the store: the workflow installs this.
         */
        readonly ActionEffects: Domain.ActionEffects;
        readonly ActionId: Domain.ActionId;
        /**
         * ActionMessage
         * @description A label emitted by an attempt; measurements belong in its scientific result.
         */
        readonly ActionMessage: Domain.ActionMessage;
        readonly ActionPoll: Domain.ActionPoll;
        /** ActionReceipt */
        readonly ActionReceipt: Domain.ActionReceipt;
        /** Added[ConstructRef] */
        readonly Added_ConstructRef_: Domain.Added<Domain.ConstructRef>;
        /** Added[DataPoint] */
        readonly Added_DataPoint_: Domain.Added<Domain.DataPoint>;
        /** Added[EdgeRef] */
        readonly Added_EdgeRef_: Domain.Added<Domain.EdgeRef>;
        /** Added[ParameterSpec] */
        readonly Added_ParameterSpec_: Domain.Added<Domain.ParameterSpec>;
        /** Applied[DataComparisonResult] */
        readonly Applied_DataComparisonResult_: Domain.Applied<Domain.DataComparisonResult>;
        /** Applied[DataPreparationResult] */
        readonly Applied_DataPreparationResult_: Domain.Applied<Domain.DataPreparationResult>;
        /** Applied[ModelFitResult] */
        readonly Applied_ModelFitResult_: Domain.Applied<Domain.ModelFitResult>;
        /** Applied[ModelSimulationResult] */
        readonly Applied_ModelSimulationResult_: Domain.Applied<Domain.ModelSimulationResult>;
        /** Applied[NoneType] */
        readonly Applied_NoneType_: Domain.Applied<null>;
        /**
         * ArtifactEnvelope
         * @description An artifact envelope delivers a stored payload with its revision and file
         *     list.
         */
        readonly ArtifactEnvelope: Domain.ArtifactEnvelope;
        readonly ArtifactFreshness: Domain.ArtifactFreshness;
        /** @enum {string} */
        readonly ArtifactId: Domain.ArtifactId;
        /**
         * ArtifactRecord
         * @description Artifact revision metadata records how a stored artifact was produced and which inputs it
         *     used.
         *
         *     ``derived_from`` pins the exact input versions the payload was computed
         *     from. For initial model revisions it is empty. ``created_at`` is
         *     stamped by the activity that produced the revision — never inside workflow
         *     code, where wall-clock time is non-deterministic.
         */
        readonly ArtifactRecord: Domain.ArtifactRecord;
        readonly Assessment_ConvergenceAssessmentSubject_NumericCriterionEvidence_: Domain.Assessment<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence>;
        readonly Assessment_IndicatorCheckSubject_NumericCriterionEvidence_: Domain.Assessment<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence>;
        readonly Assessment_str_ParticleMCMCEvidence_: Domain.Assessment<string, Domain.ParticleMCMCEvidence>;
        /**
         * AttemptRecord
         * @description Stored inside the Git object, with no self-referential publication ID.
         */
        readonly AttemptRecord: Domain.AttemptRecord;
        /**
         * AttemptTraceIndex
         * @description Promoted traces identified by their committed execution sequence.
         */
        readonly AttemptTraceIndex: Domain.AttemptTraceIndex;
        /** Attempt[Literal['data_diff'], DataDiffRequest, DataComparisonResult] */
        readonly Attempt_Literal__data_diff___DataDiffRequest_DataComparisonResult_: Domain.Attempt<"data_diff", Domain.DataDiffRequest, Domain.DataComparisonResult>;
        /** Attempt[Literal['edit_model'], EditModelRequest, NoneType] */
        readonly Attempt_Literal__edit_model___EditModelRequest_NoneType_: Domain.Attempt<"edit_model", Domain.EditModelRequest, null>;
        /** Attempt[Literal['fit'], FitRequest, ModelFitResult] */
        readonly Attempt_Literal__fit___FitRequest_ModelFitResult_: Domain.Attempt<"fit", Domain.FitRequest, Domain.ModelFitResult>;
        /** Attempt[Literal['prepare_data'], PrepareDataRequest, DataPreparationResult] */
        readonly Attempt_Literal__prepare_data___PrepareDataRequest_DataPreparationResult_: Domain.Attempt<"prepare_data", Domain.PrepareDataRequest, Domain.DataPreparationResult>;
        /** Attempt[Literal['set_question'], SetQuestionRequest, NoneType] */
        readonly Attempt_Literal__set_question___SetQuestionRequest_NoneType_: Domain.Attempt<"set_question", Domain.SetQuestionRequest, null>;
        /** Attempt[Literal['simulate'], SimulateRequest, ModelSimulationResult] */
        readonly Attempt_Literal__simulate___SimulateRequest_ModelSimulationResult_: Domain.Attempt<"simulate", Domain.SimulateRequest, Domain.ModelSimulationResult>;
        /**
         * AuthoredLawProvenance
         * @description The current laws have authored ancestry without retained fitting.
         */
        readonly AuthoredLawProvenance: Domain.AuthoredLawProvenance;
        /** Available[CausalEffectResult] */
        readonly Available_CausalEffectResult_: Domain.Available<Domain.CausalEffectResult>;
        /** Available[PosteriorPredictiveChecks] */
        readonly Available_PosteriorPredictiveChecks_: Domain.Available<Domain.PosteriorPredictiveChecks>;
        /** Available[tuple[ParameterDrawColumn, ...]] */
        readonly Available_tuple_ParameterDrawColumn__________: Domain.Available<readonly (Domain.ParameterDrawColumn)[]>;
        /** BernoulliLogitsLawSpec[Expression] */
        readonly "BernoulliLogitsLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliLogits";
            readonly logits: components["schemas"]["Expression-Input"];
        };
        /** BernoulliLogitsLawSpec[Expression] */
        readonly "BernoulliLogitsLawSpec_Expression_-Output": Domain.BernoulliLogitsLawSpec<Domain.Expression>;
        /** BernoulliProbsLawSpec[Expression] */
        readonly "BernoulliProbsLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliProbs";
            readonly probs: components["schemas"]["Expression-Input"];
        };
        /** BernoulliProbsLawSpec[Expression] */
        readonly "BernoulliProbsLawSpec_Expression_-Output": Domain.BernoulliProbsLawSpec<Domain.Expression>;
        /** BetaLawSpec[Expression] */
        readonly "BetaLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Beta";
            readonly concentration1: components["schemas"]["Expression-Input"];
            readonly concentration0: components["schemas"]["Expression-Input"];
        };
        /** BetaLawSpec[Expression] */
        readonly "BetaLawSpec_Expression_-Output": Domain.BetaLawSpec<Domain.Expression>;
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        readonly "BinaryExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "binary";
            readonly operator: components["schemas"]["BinaryOperator"];
            readonly left: components["schemas"]["Expression-Input"];
            readonly right: components["schemas"]["Expression-Input"];
        };
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        readonly "BinaryExpression-Output": Domain.BinaryExpression;
        /** @enum {string} */
        readonly BinaryOperator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
        /** Body_upload_file_api_upload_post */
        readonly Body_upload_file_api_upload_post: {
            /** File */
            readonly file: Blob;
            /** Workspaceid */
            readonly workspaceId: string;
        };
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        readonly "CallExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "call";
            readonly function: components["schemas"]["ExpressionFunction"];
            /** Arguments */
            readonly arguments: readonly components["schemas"]["Expression-Input"][];
        };
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        readonly "CallExpression-Output": Domain.CallExpression;
        /** CategoricalLawSpec[Expression] */
        readonly "CategoricalLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Categorical";
            readonly logits: components["schemas"]["Expression-Input"];
        };
        /** CategoricalLawSpec[Expression] */
        readonly "CategoricalLawSpec_Expression_-Output": Domain.CategoricalLawSpec<Domain.Expression>;
        /**
         * CategoryProbabilitySummary
         * @description Predictive probabilities for each declared level; unobserved anchors are null.
         */
        readonly CategoryProbabilitySummary: Domain.CategoryProbabilitySummary;
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        readonly "CausalEdgeSpec-Input": {
            /** @description Persistent identity. Preserve when revising the same edge. */
            readonly id: components["schemas"]["EdgeId-Input"];
            /**
             * Mechanisms
             * @default []
             */
            readonly mechanisms?: readonly components["schemas"]["DriftMechanismSpec-Input"][];
            /** @description Cause construct; shared endpoints have one identity. */
            readonly cause: components["schemas"]["ConstructSpec-Input"] | components["schemas"]["ConstructRef-Input"];
            /** @description Effect construct; shared endpoints have one identity. */
            readonly effect: components["schemas"]["ConstructSpec-Input"] | components["schemas"]["ConstructRef-Input"];
            /**
             * Description
             * @description Theoretical justification for this causal link
             */
            readonly description: string;
            /**
             * Sources
             * @description Literature sources supporting this causal link
             */
            readonly sources?: readonly components["schemas"]["LiteratureSource-Input"][];
        };
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        readonly "CausalEdgeSpec-Output": Domain.CausalEdgeSpec;
        /**
         * CausalEffectResult
         * @description Causal effects and realized trajectories under the enclosing report's design.
         */
        readonly CausalEffectResult: Domain.CausalEffectResult;
        /**
         * ChainDiagnostics
         * @description Compact retained-chain measurements; plot series compose the report detail.
         */
        readonly ChainDiagnostics: Domain.ChainDiagnostics;
        readonly Change_ConstructRef_: Domain.Change<Domain.ConstructRef>;
        readonly Change_DataPoint_: Domain.Change<Domain.DataPoint>;
        readonly Change_EdgeRef_: Domain.Change<Domain.EdgeRef>;
        readonly Change_ParameterSpec_: Domain.Change<Domain.ParameterSpec>;
        /** @enum {string} */
        readonly CheckGroup: Domain.CheckGroup;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        readonly "CoefficientExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "coefficient";
            readonly role: components["schemas"]["CoefficientRole"];
            /**
             * Value
             * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
             */
            readonly value?: number | components["schemas"]["ParameterId-Input"] | null;
            /**
             * Construct Ids
             * @description Additional constructs participating in this coefficient use.
             * @default []
             */
            readonly construct_ids?: readonly components["schemas"]["ConstructId-Input"][];
        };
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        readonly "CoefficientExpression-Output": Domain.CoefficientExpression;
        /** @enum {string} */
        readonly CoefficientRole: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
        /**
         * CompletedExtractionWorker
         * @description Retained measurements from a completed worker; no failure field exists.
         */
        readonly CompletedExtractionWorker: Domain.CompletedExtractionWorker;
        /** CompletedPoll */
        readonly CompletedPoll: Domain.CompletedPoll;
        /**
         * ComputedExtractionSpec
         * @description Compute a deterministic support-window measurement from source columns.
         */
        readonly "ComputedExtractionSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "computed";
            /**
             * How To Measure
             * @description Description of the deterministic measurement.
             */
            readonly how_to_measure: string;
            /** Source Columns */
            readonly source_columns: readonly string[];
            /** @description Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation. */
            readonly computed_rule?: components["schemas"]["WindowExpression"] | null;
            /**
             * Fill Null
             * @description Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
             */
            readonly fill_null?: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
            /**
             * Fill Null Limit
             * @description Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
             */
            readonly fill_null_limit?: number | null;
        };
        /**
         * ComputedExtractionSpec
         * @description Compute a deterministic support-window measurement from source columns.
         */
        readonly "ComputedExtractionSpec-Output": Domain.ComputedExtractionSpec;
        readonly "ConstructId-Input": string;
        readonly "ConstructId-Output": Domain.ConstructId;
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        readonly "ConstructRef-Input": {
            /**
             * Kind
             * @default construct
             * @constant
             */
            readonly kind?: "construct";
            readonly id: components["schemas"]["ConstructId-Input"];
        };
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        readonly "ConstructRef-Output": Domain.ConstructRef;
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        readonly "ConstructSpec-Input": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["ConstructId-Input"];
            /**
             * Name
             * @description Construct name (e.g., 'stress', 'sleep_quality')
             */
            readonly name: string;
            /**
             * Description
             * @description What this theoretical construct represents
             */
            readonly description: string;
            /**
             * Indicators
             * @default []
             */
            readonly indicators?: readonly components["schemas"]["IndicatorSpec-Input"][];
            /**
             * Dynamics
             * @default []
             */
            readonly dynamics?: readonly components["schemas"]["DynamicsMechanismSpec-Input"][];
            /**
             * Coefficients
             * @default []
             */
            readonly coefficients?: readonly components["schemas"]["CoefficientExpression-Input"][];
            /**
             * Innovation Family
             * @default gaussian
             * @enum {string}
             */
            readonly innovation_family?: "gaussian" | "student_t";
            /** @description Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points. */
            readonly distribution?: components["schemas"]["DistributionId-Input"] | null;
            /** @description 'endogenous' means modeled, with or without parents; 'exogenous' means given through direct exact readings, with no law. Unmeasured constructs are endogenous. */
            readonly role: components["schemas"]["Role"];
            /** @description 'time_varying' (changes over time) or 'time_invariant' (fixed) */
            readonly temporal_status: components["schemas"]["TemporalStatus"];
        };
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        readonly "ConstructSpec-Output": Domain.ConstructSpec;
        readonly ConvergenceAssessmentSubject: Domain.ConvergenceAssessmentSubject;
        /**
         * ConvergenceCriterion
         * @enum {string}
         */
        readonly ConvergenceCriterion: Domain.ConvergenceCriterion;
        /**
         * ConvergenceSubject
         * @description A convergence criterion on one stable scientific scalar.
         */
        readonly ConvergenceSubject: Domain.ConvergenceSubject;
        /** CreateBranchBody */
        readonly CreateBranchBody: {
            /** Name */
            readonly name: string;
            readonly at: components["schemas"]["GitOid-Input"];
        };
        /** @enum {string} */
        readonly DSMCLeafProposal: Domain.DSMCLeafProposal;
        /**
         * DataComparisonResult
         * @description A retained data comparison; it never installs scientific artifacts.
         */
        readonly DataComparisonResult: Domain.DataComparisonResult;
        /**
         * DataDiffReport
         * @description Comparisons of existing data, preserving each history's immutable source reference.
         */
        readonly DataDiffReport: Domain.DataDiffReport;
        /**
         * DataDiffRequest
         * @description Compare two immutable data selections, each containing one or more histories.
         */
        readonly "DataDiffRequest-Input": {
            /**
             * Action
             * @default data_diff
             * @constant
             */
            readonly action?: "data_diff";
            readonly left: components["schemas"]["DataSelection-Input"];
            readonly right: components["schemas"]["DataSelection-Input"];
        };
        /**
         * DataDiffRequest
         * @description Compare two immutable data selections, each containing one or more histories.
         */
        readonly "DataDiffRequest-Output": Domain.DataDiffRequest;
        /**
         * DataPoint
         * @description An observed anchor and support; dates are synthetic for a calendar-free series.
         */
        readonly DataPoint: Domain.DataPoint;
        /**
         * DataPreparationResult
         * @description Preparation artifacts and the measurements actually retained by extraction.
         */
        readonly DataPreparationResult: Domain.DataPreparationResult;
        /**
         * DataPreparationSpec
         * @description A versioned data definition supplied directly to prepare_data.
         */
        readonly "DataPreparationSpec-Input": {
            /** Default Window */
            readonly default_window: string;
            /** Variables */
            readonly variables: readonly components["schemas"]["DataVariableSpec-Input"][];
            /**
             * Context
             * @description Optional context for interpreting the source data.
             * @default
             */
            readonly context?: string;
        };
        /**
         * DataPreparationSpec
         * @description A versioned data definition supplied directly to prepare_data.
         */
        readonly "DataPreparationSpec-Output": Domain.DataPreparationSpec;
        /**
         * DataProfileArtifact
         * @description Model-independent empirical measurements and data-quality findings.
         */
        readonly DataProfileArtifact: Domain.DataProfileArtifact;
        readonly "DataRef-Input": components["schemas"]["PanelRef-Input"] | components["schemas"]["SimulationRef-Input"];
        readonly "DataRef-Output": Domain.DataRef;
        /** @description A data selection identifies one or more saved observation histories. */
        readonly "DataSelection-Input": components["schemas"]["DataRef-Input"] | readonly components["schemas"]["DataRef-Input"][];
        /** @description A data selection identifies one or more saved observation histories. */
        readonly "DataSelection-Output": Domain.DataSelection;
        /**
         * DataSeries
         * @description One variable's recorded measurements in one history; no pooling across replicas.
         */
        readonly DataSeries: Domain.DataSeries;
        /** @description Uploaded sources or one recorded simulation replicate. */
        readonly DataSourceRef: Domain.DataSourceRef;
        /** @enum {string} */
        readonly DataStatistic: Domain.DataStatistic;
        /**
         * DataStatisticComparison
         * @description The same descriptive statistic measured independently in every selected history.
         */
        readonly DataStatisticComparison: Domain.DataStatisticComparison;
        /**
         * DataVariableDiff
         * @description Definitions, histories and comparisons for one persistent observation identity.
         */
        readonly DataVariableDiff: Domain.DataVariableDiff;
        /**
         * DataVariableSpec
         * @description Compose an observed variable with its data-owned extraction instructions.
         */
        readonly "DataVariableSpec-Input": {
            readonly observation: components["schemas"]["ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Input"];
            readonly extraction: components["schemas"]["ExtractionSpec-Input"];
        };
        /**
         * DataVariableSpec
         * @description Compose an observed variable with its data-owned extraction instructions.
         */
        readonly "DataVariableSpec-Output": Domain.DataVariableSpec;
        /** DeltaLawSpec[Expression] */
        readonly "DeltaLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Delta";
            readonly v: components["schemas"]["Expression-Input"];
        };
        /** DeltaLawSpec[Expression] */
        readonly "DeltaLawSpec_Expression_-Output": Domain.DeltaLawSpec<Domain.Expression>;
        /**
         * DensityCurve
         * @description Aligned density ordinates; the owning field distinguishes PDF samples from histogram heights.
         */
        readonly DensityCurve: Domain.DensityCurve;
        readonly "DistributionId-Input": string;
        readonly "DistributionId-Output": Domain.DistributionId;
        /**
         * DriftMechanismSpec
         * @description An additive drift contribution on a construct or directed edge.
         */
        readonly "DriftMechanismSpec-Input": {
            readonly id: components["schemas"]["MechanismId-Input"];
            readonly expression: components["schemas"]["Expression-Input"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "drift";
        };
        /**
         * DriftMechanismSpec
         * @description An additive drift contribution on a construct or directed edge.
         */
        readonly "DriftMechanismSpec-Output": Domain.DriftMechanismSpec;
        readonly "DynamicsMechanismSpec-Input": components["schemas"]["DriftMechanismSpec-Input"] | components["schemas"]["PotentialMechanismSpec-Input"];
        readonly "DynamicsMechanismSpec-Output": Domain.DynamicsMechanismSpec;
        readonly "EdgeId-Input": string;
        readonly "EdgeId-Output": Domain.EdgeId;
        /**
         * EdgeRef
         * @description An edge reference identifies a causal relationship independently of edits to its
         *     definition.
         */
        readonly EdgeRef: Domain.EdgeRef;
        /**
         * EditModelRequest
         * @description Replace one named base revision with a validated scientific definition.
         */
        readonly "EditModelRequest-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_model";
            readonly expected_revision: components["schemas"]["GitOid-Input"] | null;
            /** @description Endogenous constructs are modeled, with or without parents, and include every latent construct. Exogenous constructs are given by direct exact Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law. */
            readonly model: components["schemas"]["ModelSpec-Input"];
        };
        /**
         * EditModelRequest
         * @description Replace one named base revision with a validated scientific definition.
         */
        readonly "EditModelRequest-Output": Domain.EditModelRequest;
        /**
         * EffectSummary
         * @description An effect summary reports posterior location, uncertainty, and sign probability.
         */
        readonly EffectSummary: Domain.EffectSummary;
        /** EmpiricalPoint */
        readonly EmpiricalPoint: Domain.EmpiricalPoint;
        /**
         * EnergyDiagnostics
         * @description Energy distributions and the producer's chain-specific BFMI values.
         */
        readonly EnergyDiagnostics: Domain.EnergyDiagnostics;
        readonly EntityRef: Domain.EntityRef;
        /**
         * EvaluatedPredictiveChecks
         * @description An evaluated run may retain failed and partially unavailable scientific evidence.
         */
        readonly EvaluatedPredictiveChecks: Domain.EvaluatedPredictiveChecks;
        /** Evaluated[ConvergenceAssessmentSubject, NumericCriterionEvidence] */
        readonly Evaluated_ConvergenceAssessmentSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence>;
        /** Evaluated[IndicatorCheckSubject, NumericCriterionEvidence] */
        readonly Evaluated_IndicatorCheckSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence>;
        /** Evaluated[PredictiveSubject, tuple[NumericCriterionEvidence, ...]] */
        readonly Evaluated_PredictiveSubject_tuple_NumericCriterionEvidence__________: Domain.Evaluated<Domain.PredictiveSubject, readonly (Domain.NumericCriterionEvidence)[]>;
        /** Evaluated[QuestionSubject, str] */
        readonly Evaluated_QuestionSubject_str_: Domain.Evaluated<Domain.QuestionSubject, string>;
        /** Evaluated[str, ParticleMCMCEvidence] */
        readonly Evaluated_str_ParticleMCMCEvidence_: Domain.Evaluated<string, Domain.ParticleMCMCEvidence>;
        /** Evaluated[str, str] */
        readonly Evaluated_str_str_: Domain.Evaluated<string, string>;
        readonly Evaluation_CausalEffectResult_: Domain.Evaluation<Domain.CausalEffectResult>;
        readonly Evaluation_PosteriorPredictiveChecks_: Domain.Evaluation<Domain.PosteriorPredictiveChecks>;
        readonly "Expression-Input": components["schemas"]["LiteralExpression-Input"] | components["schemas"]["StateExpression-Input"] | components["schemas"]["CoefficientExpression-Input"] | components["schemas"]["BinaryExpression-Input"] | components["schemas"]["CallExpression-Input"];
        readonly "Expression-Output": Domain.Expression;
        /** @enum {string} */
        readonly ExpressionFunction: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
        /**
         * ExtractionPlanEvent
         * @description The extraction fan-out plan.
         */
        readonly ExtractionPlanEvent: Domain.ExtractionPlanEvent;
        /**
         * ExtractionSnapshotEvent
         * @description Aggregate extraction worker counts.
         */
        readonly ExtractionSnapshotEvent: Domain.ExtractionSnapshotEvent;
        readonly "ExtractionSpec-Input": components["schemas"]["ComputedExtractionSpec-Input"] | components["schemas"]["SemanticExtractionSpec-Input"];
        readonly "ExtractionSpec-Output": Domain.ExtractionSpec;
        /**
         * ExtractionWorkerEvent
         * @description One extraction worker's state; a worker reports its LLM calls when it finishes.
         */
        readonly ExtractionWorkerEvent: Domain.ExtractionWorkerEvent;
        readonly ExtractionWorkerResult: Domain.ExtractionWorkerResult;
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact revision and records its freshness.
         */
        readonly FactSource: Domain.FactSource;
        /**
         * FailedExtractionChunk
         * @description An extraction failure with its error and no usable result-file reference.
         */
        readonly FailedExtractionChunk: Domain.FailedExtractionChunk;
        /**
         * FilePreparationSpec
         * @description Uploaded sources and the complete recipe for preparing their observations.
         */
        readonly "FilePreparationSpec-Input": {
            readonly source: components["schemas"]["FileSourceRef-Input"];
            readonly definition: components["schemas"]["DataPreparationSpec-Input"];
        };
        /**
         * FilePreparationSpec
         * @description Uploaded sources and the complete recipe for preparing their observations.
         */
        readonly "FilePreparationSpec-Output": Domain.FilePreparationSpec;
        /**
         * FileSourceRef
         * @description Explicit uploaded filenames, relative to this study's input directory.
         */
        readonly "FileSourceRef-Input": {
            /** Files */
            readonly files: readonly string[];
            /**
             * Start
             * @description Inclusive UTC source-coverage date.
             */
            readonly start?: string | null;
            /**
             * End
             * @description Exclusive UTC source-coverage date.
             */
            readonly end?: string | null;
        };
        /**
         * FileSourceRef
         * @description Explicit uploaded filenames, relative to this study's input directory.
         */
        readonly "FileSourceRef-Output": Domain.FileSourceRef;
        /** @enum {string} */
        readonly FitReliability: Domain.FitReliability;
        /**
         * FitRequest
         * @description Condition explicitly selected model and observation revisions.
         */
        readonly "FitRequest-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "fit";
            readonly model_revision: components["schemas"]["GitOid-Input"];
            readonly panel_revision: components["schemas"]["GitOid-Input"];
            readonly settings?: components["schemas"]["FitSettingsSpec-Input"];
        };
        /**
         * FitRequest
         * @description Condition explicitly selected model and observation revisions.
         */
        readonly "FitRequest-Output": Domain.FitRequest;
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured particle sampler.
         */
        readonly "FitSettingsSpec-Input": {
            /** Num Samples */
            readonly num_samples?: number | null;
            /** Num Warmup */
            readonly num_warmup?: number | null;
            /** Num Chains */
            readonly num_chains?: number | null;
            /** N Particles */
            readonly n_particles?: number | null;
            /** Seed */
            readonly seed?: number | null;
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured particle sampler.
         */
        readonly "FitSettingsSpec-Output": Domain.FitSettingsSpec;
        /**
         * FitSummary
         * @description A fit read contains the inference report summary and server-composed display findings.
         *
         *     Per-draw diagnostics load separately from the inference report endpoint.
         */
        readonly FitSummary: Domain.FitSummary;
        /**
         * FittedLawProvenance
         * @description All current laws retain one committed fit's model and observation panel.
         */
        readonly FittedLawProvenance: Domain.FittedLawProvenance;
        /** GammaLawSpec[Expression] */
        readonly "GammaLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Gamma";
            readonly concentration: components["schemas"]["Expression-Input"];
            readonly rate: components["schemas"]["Expression-Input"];
        };
        /** GammaLawSpec[Expression] */
        readonly "GammaLawSpec_Expression_-Output": Domain.GammaLawSpec<Domain.Expression>;
        readonly "GitOid-Input": string;
        readonly "GitOid-Output": Domain.GitOid;
        /**
         * GitRef
         * @description An exact file in a study's Git object database: repository, object, and path.
         */
        readonly GitRef: Domain.GitRef;
        /** HTTPValidationError */
        readonly HTTPValidationError: {
            /** Detail */
            readonly detail?: readonly components["schemas"]["ValidationError"][];
        };
        /**
         * HistogramBin
         * @description A histogram bin gives its interval, center, and number of posterior draws.
         */
        readonly HistogramBin: Domain.HistogramBin;
        /**
         * IdentificationReport
         * @description Positive and negative causal identification findings for the study question's outcome.
         */
        readonly IdentificationReport: Domain.IdentificationReport;
        /**
         * IdentifiedTreatmentStatus
         * @description Details on how a treatment effect is identified.
         */
        readonly IdentifiedTreatmentStatus: Domain.IdentifiedTreatmentStatus;
        /**
         * IdentityTransformSpec
         * @description Keep the authored probability law on the scientific quantity's native scale.
         */
        readonly "IdentityTransformSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "identity";
        };
        /**
         * IdentityTransformSpec
         * @description Keep the authored probability law on the scientific quantity's native scale.
         */
        readonly "IdentityTransformSpec-Output": Domain.IdentityTransformSpec;
        /**
         * IndicatorAudit
         * @description An indicator audit combines its empirical data profile with the results of validation
         *     checks.
         */
        readonly IndicatorAudit: Domain.IndicatorAudit;
        /**
         * IndicatorCheckSubject
         * @description The indicator and criterion remain present when evaluation is unavailable.
         */
        readonly IndicatorCheckSubject: Domain.IndicatorCheckSubject;
        /**
         * IndicatorEmpiricalProfile
         * @description An empirical profile summarizes an indicator's observed values, coverage, and data-
         *     quality signals.
         */
        readonly IndicatorEmpiricalProfile: Domain.IndicatorEmpiricalProfile;
        readonly "IndicatorId-Input": string;
        readonly "IndicatorId-Output": Domain.IndicatorId;
        /**
         * IndicatorPolarity
         * @description Indicator polarity states whether a measurement increases or decreases with its
         *     construct.
         * @enum {string}
         */
        readonly IndicatorPolarity: "positive" | "negative";
        /**
         * IndicatorRef
         * @description An indicator reference identifies a measurement definition independently of its name or
         *     revision.
         */
        readonly IndicatorRef: Domain.IndicatorRef;
        /**
         * IndicatorSpec
         * @description Bind an observed-variable ID to a construct and an emission likelihood.
         *
         *     Extraction instructions belong to DataPreparationSpec. The shared observation
         *     schema also permits generative models before any observations have been collected.
         */
        readonly "IndicatorSpec-Input": {
            readonly observation: components["schemas"]["ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Input"];
            readonly likelihood?: components["schemas"]["LikelihoodSpec-Input"] | null;
            /** @description Whether higher values move with (positive) or against (negative) the construct. */
            readonly construct_polarity: components["schemas"]["IndicatorPolarity"];
        };
        /**
         * IndicatorSpec
         * @description Bind an observed-variable ID to a construct and an emission likelihood.
         *
         *     Extraction instructions belong to DataPreparationSpec. The shared observation
         *     schema also permits generative models before any observations have been collected.
         */
        readonly "IndicatorSpec-Output": Domain.IndicatorSpec;
        /**
         * InferenceMetadata
         * @description Inference metadata records the sampling method, sample count, and run duration.
         */
        readonly InferenceMetadata: Domain.InferenceMetadata;
        /**
         * InferenceReport
         * @description The compact core composed with retained detail, without filtering or re-parsing.
         */
        readonly InferenceReport: Domain.InferenceReport;
        /**
         * InferenceReportCore
         * @description Compact scientific report shared by snapshots and the full report.
         */
        readonly InferenceReportCore: Domain.InferenceReportCore;
        /**
         * InferenceReportDetail
         * @description Retained plot series served in full by the report endpoint.
         */
        readonly InferenceReportDetail: Domain.InferenceReportDetail;
        /**
         * InitialCorrelationTransformSpec
         * @description Constrain an initial-state correlation to its scientific support [-1, 1].
         */
        readonly "InitialCorrelationTransformSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "initial_state_correlation";
        };
        /**
         * InitialCorrelationTransformSpec
         * @description Constrain an initial-state correlation to its scientific support [-1, 1].
         */
        readonly "InitialCorrelationTransformSpec-Output": Domain.InitialCorrelationTransformSpec;
        /**
         * IntervalEffectTransformSpec
         * @description Divide an interval effect by its explicit duration in days.
         */
        readonly "IntervalEffectTransformSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_effect_to_ct_rate";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
        /**
         * IntervalEffectTransformSpec
         * @description Divide an interval effect by its explicit duration in days.
         */
        readonly "IntervalEffectTransformSpec-Output": Domain.IntervalEffectTransformSpec;
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Input": {
            readonly target: components["schemas"]["ConstructId-Input"];
            /**
             * After
             * @description Offset from the design's start; omitted means at the start.
             */
            readonly after?: string | null;
            /** Value */
            readonly value: number;
        };
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Output": Domain.InterventionSpec;
        readonly "JsonArray-Input": Domain.JsonArray;
        readonly "JsonArray-Output": Domain.JsonArray;
        readonly "JsonObject-Input": Domain.JsonObject;
        readonly "JsonObject-Output": Domain.JsonObject;
        readonly JsonScalar: boolean | number | string | null;
        readonly "JsonValue-Input": Domain.JsonValue;
        readonly "JsonValue-Output": Domain.JsonValue;
        /**
         * LLMTrace
         * @description An LLM trace records a conversation, its model, elapsed time, and token usage.
         */
        readonly LLMTrace: Domain.LLMTrace;
        /**
         * LOODiagnostics
         * @description Exact-emission leave-one-measurement-row-out interpolation diagnostics.
         */
        readonly LOODiagnostics: Domain.LOODiagnostics;
        /**
         * LOOPITPoint
         * @description A retained PIT value and its empirical and reference cumulative probabilities.
         */
        readonly LOOPITPoint: Domain.LOOPITPoint;
        /**
         * LikelihoodSpec
         * @description An indicator's conditional probability law and its scientific justification.
         */
        readonly "LikelihoodSpec-Input": {
            readonly law: components["schemas"]["ObservationLawSpec-Input"];
            /**
             * Standardized
             * @description Whether observations are mean-centered and scaled before fitting.
             * @default false
             */
            readonly standardized?: boolean;
            /**
             * Reasoning
             * @description Why this conditional law was chosen for the indicator
             */
            readonly reasoning: string;
            /**
             * Sources
             * @default []
             */
            readonly sources?: readonly components["schemas"]["LiteratureSource-Input"][];
        };
        /**
         * LikelihoodSpec
         * @description An indicator's conditional probability law and its scientific justification.
         */
        readonly "LikelihoodSpec-Output": Domain.LikelihoodSpec;
        /**
         * LiteralExpression
         * @description A finite scalar constant in a model equation.
         */
        readonly "LiteralExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "literal";
            /** Value */
            readonly value: number;
        };
        /**
         * LiteralExpression
         * @description A finite scalar constant in a model equation.
         */
        readonly "LiteralExpression-Output": Domain.LiteralExpression;
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        readonly "LiteratureSource-Input": {
            /**
             * Title
             * @description Title of the source (paper, meta-analysis, textbook, etc.)
             */
            readonly title: string;
            /**
             * Url
             * @description URL of the source if available
             */
            readonly url?: string | null;
            /**
             * Snippet
             * @description Relevant excerpt or paraphrase from the source
             */
            readonly snippet: string;
        };
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        readonly "LiteratureSource-Output": Domain.LiteratureSource;
        /**
         * MarginalParticleGibbsSpec
         * @description Marginalized Particle Gibbs inference settings.
         */
        readonly MarginalParticleGibbsSpec: Domain.MarginalParticleGibbsSpec;
        /** @enum {string} */
        readonly MeasurementDtype: "continuous" | "binary" | "count" | "ordinal" | "categorical";
        /**
         * MeasurementsData
         * @description Counts and representative observations read directly from one panel revision.
         */
        readonly MeasurementsData: Domain.MeasurementsData;
        /**
         * MechanismCurves
         * @description Exact conditional drift contributions, not marginal or total causal effects.
         */
        readonly MechanismCurves: Domain.MechanismCurves;
        readonly "MechanismId-Input": string;
        readonly "MechanismId-Output": Domain.MechanismId;
        /**
         * MechanismRef
         * @description A particular additive term, independently of its position or coefficient values.
         */
        readonly MechanismRef: Domain.MechanismRef;
        /** MechanismViewRequest */
        readonly MechanismViewRequest: {
            /** Owner Id */
            readonly owner_id: string;
            readonly axis?: components["schemas"]["ConstructId-Input"] | null;
            /**
             * Lower
             * @default -3
             */
            readonly lower?: number;
            /**
             * Upper
             * @default 3
             */
            readonly upper?: number;
            /** Held */
            readonly held?: {
                readonly [key: string]: number;
            };
            readonly moderator?: components["schemas"]["ConstructId-Input"] | null;
            /**
             * Levels
             * @default [
             *       -1,
             *       0,
             *       1
             *     ]
             */
            readonly levels?: readonly number[];
            /**
             * Start
             * @default 0
             */
            readonly start?: number;
            /**
             * Count
             * @default 24
             */
            readonly count?: number;
            /**
             * Points
             * @default 201
             */
            readonly points?: number;
        };
        /**
         * Missing
         * @description An artifact absent from the selected state.
         */
        readonly Missing: Domain.Missing;
        /**
         * MixedLawProvenance
         * @description Some laws retain a committed fit and others have different ancestry.
         */
        readonly MixedLawProvenance: Domain.MixedLawProvenance;
        /**
         * ModelCheckReport
         * @description Checks selected by their consumed inputs, retained with the study snapshot.
         */
        readonly ModelCheckReport: Domain.ModelCheckReport;
        /**
         * ModelDiffReport
         * @description A model diff joins typed entity comparisons and evidence at two model revisions or checkpoints.
         */
        readonly ModelDiffReport: Domain.ModelDiffReport;
        /**
         * ModelFitResult
         * @description One retained fit report, with the exact inputs and truthful retention state.
         */
        readonly ModelFitResult: Domain.ModelFitResult;
        /**
         * ModelGraphView
         * @description Scientific entity identities selected for the graph at this authoring checkpoint.
         */
        readonly ModelGraphView: Domain.ModelGraphView;
        readonly ModelPredictiveEvaluation: Domain.ModelPredictiveEvaluation;
        /**
         * ModelPredictiveReport
         * @description One automatic, reproducible battery over the full model's current laws.
         */
        readonly ModelPredictiveReport: Domain.ModelPredictiveReport;
        /**
         * ModelSimulationResult
         * @description The report owns its model reference; the selected panel is separately pinned.
         */
        readonly ModelSimulationResult: Domain.ModelSimulationResult;
        /**
         * ModelSnapshot
         * @description The canonical scientific definition with independently sourced inputs and findings.
         */
        readonly ModelSnapshot: Domain.ModelSnapshot;
        /**
         * ModelSpec
         * @description A connected causal graph with owned scientific detail, built to answer the study question.
         */
        readonly "ModelSpec-Input": {
            /**
             * Edges
             * @default []
             */
            readonly edges?: readonly components["schemas"]["CausalEdgeSpec-Input"][];
            /**
             * Parameters
             * @default []
             */
            readonly parameters?: readonly components["schemas"]["ParameterSpec-Input"][];
            /**
             * Distributions
             * @description All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements.
             */
            readonly distributions?: {
                readonly [key: string]: components["schemas"]["NumPyroDistribution-Input"];
            };
            /**
             * Time Points
             * @default []
             */
            readonly time_points?: readonly number[];
            /** Measurement Clock */
            readonly measurement_clock?: string | null;
        };
        /**
         * ModelSpec
         * @description A connected causal graph with owned scientific detail, built to answer the study question.
         */
        readonly "ModelSpec-Output": Domain.ModelSpec;
        /** NegativeBinomial2LawSpec[Expression] */
        readonly "NegativeBinomial2LawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "NegativeBinomial2";
            readonly mean: components["schemas"]["Expression-Input"];
            readonly concentration: components["schemas"]["Expression-Input"];
        };
        /** NegativeBinomial2LawSpec[Expression] */
        readonly "NegativeBinomial2LawSpec_Expression_-Output": Domain.NegativeBinomial2LawSpec<Domain.Expression>;
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        readonly NonIdentifiableTreatmentStatus: Domain.NonIdentifiableTreatmentStatus;
        /** NormalLawSpec[Expression] */
        readonly "NormalLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Normal";
            readonly loc: components["schemas"]["Expression-Input"];
            readonly scale: components["schemas"]["Expression-Input"];
        };
        /** NormalLawSpec[Expression] */
        readonly "NormalLawSpec_Expression_-Output": Domain.NormalLawSpec<Domain.Expression>;
        /**
         * NotApplicable
         * @description The selected operation does not call for this result.
         */
        readonly NotApplicable: Domain.NotApplicable;
        readonly NotEvaluatedReason: Domain.NotEvaluatedReason;
        /** NotEvaluated[ConvergenceAssessmentSubject] */
        readonly NotEvaluated_ConvergenceAssessmentSubject_: Domain.NotEvaluated<Domain.ConvergenceAssessmentSubject>;
        /** NotEvaluated[IndicatorCheckSubject] */
        readonly NotEvaluated_IndicatorCheckSubject_: Domain.NotEvaluated<Domain.IndicatorCheckSubject>;
        /** NotEvaluated[PredictiveSubject] */
        readonly NotEvaluated_PredictiveSubject_: Domain.NotEvaluated<Domain.PredictiveSubject>;
        /** NotEvaluated[QuestionSubject] */
        readonly NotEvaluated_QuestionSubject_: Domain.NotEvaluated<Domain.QuestionSubject>;
        /** NotEvaluated[str] */
        readonly NotEvaluated_str_: Domain.NotEvaluated<string>;
        readonly "NumPyroDistribution-Input": {
            /** Distribution */
            readonly distribution: string;
            /** Params */
            readonly params: {
                readonly [key: string]: components["schemas"]["JsonValue-Input"];
            };
        };
        readonly "NumPyroDistribution-Output": Domain.NumPyroDistribution;
        /**
         * NumericCriterionEvidence
         * @description A measured scalar and the producer's numerical acceptance region.
         */
        readonly NumericCriterionEvidence: Domain.NumericCriterionEvidence;
        /**
         * ObservationHistory
         * @description All prepared observations, their true anchors and their measurement support.
         */
        readonly ObservationHistory: Domain.ObservationHistory;
        readonly "ObservationLawSpec-Input": components["schemas"]["DeltaLawSpec_Expression_-Input"] | components["schemas"]["NormalLawSpec_Expression_-Input"] | components["schemas"]["StudentTLawSpec_Expression_-Input"] | components["schemas"]["PoissonLawSpec_Expression_-Input"] | components["schemas"]["GammaLawSpec_Expression_-Input"] | components["schemas"]["BernoulliLogitsLawSpec_Expression_-Input"] | components["schemas"]["BernoulliProbsLawSpec_Expression_-Input"] | components["schemas"]["NegativeBinomial2LawSpec_Expression_-Input"] | components["schemas"]["BetaLawSpec_Expression_-Input"] | components["schemas"]["OrderedLogisticLawSpec_Expression_-Input"] | components["schemas"]["CategoricalLawSpec_Expression_-Input"];
        readonly "ObservationLawSpec-Output": Domain.ObservationLawSpec;
        /**
         * ObservationRecord
         * @description Canonical serialized extraction observation row.
         */
        readonly ObservationRecord: Domain.ObservationRecord;
        /** ObservationSpec[Annotated[Union[Duration, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]] */
        readonly "ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Input": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["IndicatorId-Input"];
            /**
             * Name
             * @description Indicator name (e.g., 'hrv', 'self_reported_stress')
             */
            readonly name: string;
            /** @description 'continuous', 'binary', 'count', 'ordinal', 'categorical' */
            readonly measurement_dtype: components["schemas"]["MeasurementDtype"];
            /** @description Aggregation function applied when bucketing raw extractions within the indicator support window. Supported operators: first, last, sum, count, mean, std. A computed_rule must produce this same summary. */
            readonly aggregation: components["schemas"]["SummaryOperator"];
            /**
             * Observation Window
             * @description Optional duration string describing the support window summarized by this indicator, in positive fixed units s, m, h, d or w (for example '2w'). Resolved by the preparation window or the generative model clock.
             */
            readonly observation_window?: string | null;
            /**
             * Ordinal Levels
             * @description Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
             */
            readonly ordinal_levels?: readonly string[] | null;
            /**
             * Categorical Levels
             * @description Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
             */
            readonly categorical_levels?: readonly string[] | null;
        };
        /** ObservationSpec[Annotated[Union[Duration, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]] */
        readonly "ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Output": Domain.ObservationSpec<string | null>;
        /** ObservationSpec[Duration] */
        readonly ObservationSpec_Duration_: Domain.ObservationSpec<string>;
        /** OrderedLogisticLawSpec[Expression] */
        readonly "OrderedLogisticLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "OrderedLogistic";
            readonly predictor: components["schemas"]["Expression-Input"];
            readonly cutpoints: components["schemas"]["Expression-Input"];
        };
        /** OrderedLogisticLawSpec[Expression] */
        readonly "OrderedLogisticLawSpec_Expression_-Output": Domain.OrderedLogisticLawSpec<Domain.Expression>;
        /**
         * OutcomeSubject
         * @description Whether the model defines the question's outcome as a measured, modeled course.
         */
        readonly OutcomeSubject: Domain.OutcomeSubject;
        /**
         * PPCOverlay
         * @description A predictive overlay sets one indicator's observed values against simulated ones.
         *
         *     It carries the predictive median and a few individual replicated series, the
         *     spaghetti plot of a visual predictive check.
         */
        readonly PPCOverlay: Domain.PPCOverlay;
        /**
         * PPCTestStat
         * @description A predictive test statistic compares an observed summary with its distribution under
         *     replicated data.
         *
         *     Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
         *     with a vertical line at T(y_observed).
         */
        readonly PPCTestStat: Domain.PPCTestStat;
        /**
         * PanelRef
         * @description An immutable observed panel with its own calendar history.
         */
        readonly "PanelRef-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "panel";
            readonly revision: components["schemas"]["GitOid-Input"];
        };
        /**
         * PanelRef
         * @description An immutable observed panel with its own calendar history.
         */
        readonly "PanelRef-Output": Domain.PanelRef;
        /**
         * ParameterConvergenceReport
         * @description Recorded-chain criteria cover parameters, not latent-path mixing.
         */
        readonly ParameterConvergenceReport: Domain.ParameterConvergenceReport;
        /**
         * ParameterDiagnostics
         * @description Measurements on one scientifically identified retained scalar chain.
         */
        readonly ParameterDiagnostics: Domain.ParameterDiagnostics;
        /** ParameterDrawColumn */
        readonly ParameterDrawColumn: Domain.ParameterDrawColumn;
        readonly ParameterDraws: Domain.ParameterDraws;
        readonly ParameterElementId: Domain.ParameterElementId;
        readonly "ParameterId-Input": string;
        readonly "ParameterId-Output": Domain.ParameterId;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        readonly ParameterRef: Domain.ParameterRef;
        /**
         * ParameterSpec
         * @description A named uncertain quantity; fixed coefficients are literals in component slots.
         */
        readonly "ParameterSpec-Input": {
            readonly id: components["schemas"]["ParameterId-Input"];
            /**
             * Name
             * @description Authored parameter label; relationships use its persistent ID
             */
            readonly name: string;
            /**
             * Description
             * @description Human-readable description of what this parameter represents
             */
            readonly description: string;
            readonly transform?: components["schemas"]["ParameterTransformSpec-Input"];
            /** @description Membership in a native law in ModelSpec.distributions; may be joint. None means the law has not been assigned yet. */
            readonly distribution?: components["schemas"]["DistributionId-Input"] | null;
            /**
             * Reasoning
             * @description Why the authored prior law fits this quantity, and where its values come from.
             */
            readonly reasoning?: string | null;
            /**
             * Sources
             * @description Evidence behind the authored prior law.
             * @default []
             */
            readonly sources?: readonly components["schemas"]["LiteratureSource-Input"][];
        };
        /**
         * ParameterSpec
         * @description A named uncertain quantity; fixed coefficients are literals in component slots.
         */
        readonly "ParameterSpec-Output": Domain.ParameterSpec;
        readonly "ParameterTransformSpec-Input": components["schemas"]["IdentityTransformSpec-Input"] | components["schemas"]["PersistenceTransformSpec-Input"] | components["schemas"]["IntervalEffectTransformSpec-Input"] | components["schemas"]["InitialCorrelationTransformSpec-Input"];
        readonly "ParameterTransformSpec-Output": Domain.ParameterTransformSpec;
        /**
         * ParameterWarmupDiagnostics
         * @description Realized initialization and preconditioning, with the complete Pathfinder evidence once.
         */
        readonly ParameterWarmupDiagnostics: Domain.ParameterWarmupDiagnostics;
        /**
         * ParetoKPoint
         * @description One PSIS influence measurement with its original row and scientific class.
         */
        readonly ParetoKPoint: Domain.ParetoKPoint;
        /**
         * ParticleMCMCEvidence
         * @description The production particle-MCMC target and its exact latent transition.
         */
        readonly ParticleMCMCEvidence: Domain.ParticleMCMCEvidence;
        /**
         * ParticleSamplerDiagnostics
         * @description Typed exact-sampler settings and transition telemetry from the native producer.
         */
        readonly ParticleSamplerDiagnostics: Domain.ParticleSamplerDiagnostics;
        /** PathSeries */
        readonly PathSeries: Domain.PathSeries;
        /**
         * PathfinderDiagnostics
         * @description Retained native initialization measurements; never posterior evidence.
         */
        readonly PathfinderDiagnostics: Domain.PathfinderDiagnostics;
        /**
         * PathfinderStartDiagnostics
         * @description Retained native initialization measurements; never posterior evidence.
         */
        readonly PathfinderStartDiagnostics: Domain.PathfinderStartDiagnostics;
        /**
         * PersistenceTransformSpec
         * @description Map persistence p to -log(p) divided by its explicit interval in days.
         */
        readonly "PersistenceTransformSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_persistence_to_ct_decay";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
        /**
         * PersistenceTransformSpec
         * @description Map persistence p to -log(p) divided by its explicit interval in days.
         */
        readonly "PersistenceTransformSpec-Output": Domain.PersistenceTransformSpec;
        /** PoissonLawSpec[Expression] */
        readonly "PoissonLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Poisson";
            readonly rate: components["schemas"]["Expression-Input"];
        };
        /** PoissonLawSpec[Expression] */
        readonly "PoissonLawSpec_Expression_-Output": Domain.PoissonLawSpec<Domain.Expression>;
        /**
         * PosteriorMarginal
         * @description One parameter's posterior interval, scale and density plot.
         */
        readonly PosteriorMarginal: Domain.PosteriorMarginal;
        /**
         * PosteriorPredictiveChecks
         * @description Posterior predictive checks report exact-model checks and their supporting plot data.
         */
        readonly PosteriorPredictiveChecks: Domain.PosteriorPredictiveChecks;
        /**
         * PotentialMechanismSpec
         * @description A construct potential whose negative gradient contributes to its drift.
         */
        readonly "PotentialMechanismSpec-Input": {
            readonly id: components["schemas"]["MechanismId-Input"];
            readonly expression: components["schemas"]["Expression-Input"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "potential";
        };
        /**
         * PotentialMechanismSpec
         * @description A construct potential whose negative gradient contributes to its drift.
         */
        readonly "PotentialMechanismSpec-Output": Domain.PotentialMechanismSpec;
        readonly PredictiveAssessment: Domain.PredictiveAssessment;
        /** @enum {string} */
        readonly PredictiveCheckReason: Domain.PredictiveCheckReason;
        /**
         * PredictiveComparison
         * @description A selected reference history retains its role even when checks are unavailable.
         */
        readonly PredictiveComparison: Domain.PredictiveComparison;
        readonly PredictiveComparisonResult: Domain.PredictiveComparisonResult;
        readonly PredictiveLawProvenance: Domain.PredictiveLawProvenance;
        /**
         * PredictiveSubject
         * @description One named check and its stable target in a construct's scientific context.
         */
        readonly PredictiveSubject: Domain.PredictiveSubject;
        /**
         * PrepareDataRequest
         * @description Prepare uploaded sources or a simulation replicate without a model.
         */
        readonly "PrepareDataRequest-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** Input */
            readonly input: components["schemas"]["FilePreparationSpec-Input"] | components["schemas"]["SimulationReplicateRef-Input"];
        };
        /**
         * PrepareDataRequest
         * @description Prepare uploaded sources or a simulation replicate without a model.
         */
        readonly "PrepareDataRequest-Output": Domain.PrepareDataRequest;
        /**
         * PreparedDataMetadata
         * @description Self-contained semantics and provenance of one prepared observation table.
         */
        readonly PreparedDataMetadata: Domain.PreparedDataMetadata;
        /**
         * Present
         * @description The selected artifact record and its input validity.
         */
        readonly Present: Domain.Present;
        readonly ProgressEvent: Domain.ProgressEvent;
        /** @enum {string} */
        readonly ProgressStep: Domain.ProgressStep;
        readonly QueryName: string;
        /**
         * QueryTargetSubject
         * @description One query's intervention target: defined, identified, or set inside the record.
         */
        readonly QueryTargetSubject: Domain.QueryTargetSubject;
        /**
         * QueryWindowSubject
         * @description Whether the record supports one query's window.
         */
        readonly QueryWindowSubject: Domain.QueryWindowSubject;
        readonly QuestionAssessment: Domain.QuestionAssessment;
        /**
         * QuestionCheckReport
         * @description The study question checked against the model and, once prepared, the record.
         */
        readonly QuestionCheckReport: Domain.QuestionCheckReport;
        /**
         * QuestionSpec
         * @description What the study asks: the user's words, the outcome, and named contrasts.
         *
         *     Each query is a contrast of its interventions against the recorded course.
         *     Constructs are named by identity before a model defines them.
         */
        readonly "QuestionSpec-Input": {
            /**
             * Text
             * @description The user's question in their own words.
             */
            readonly text: string;
            /** @description The construct whose course answers the question. */
            readonly outcome?: components["schemas"]["ConstructId-Input"] | null;
            /**
             * Queries
             * @description Named contrasts against the recorded course, each with at least one intervention.
             */
            readonly queries?: {
                readonly [key: string]: components["schemas"]["SimulationSpec-Input"];
            };
        };
        /**
         * QuestionSpec
         * @description What the study asks: the user's words, the outcome, and named contrasts.
         *
         *     Each query is a contrast of its interventions against the recorded course.
         *     Constructs are named by identity before a model defines them.
         */
        readonly "QuestionSpec-Output": Domain.QuestionSpec;
        readonly QuestionSubject: Domain.QuestionSubject;
        /** Raised */
        readonly Raised: Domain.Raised;
        /**
         * RankHistogram
         * @description Pooled-rank bin counts, grouped in original chain order.
         */
        readonly RankHistogram: Domain.RankHistogram;
        /**
         * RawDataColumnDescription
         * @description A stored column's physical type and authored interpretation.
         */
        readonly RawDataColumnDescription: Domain.RawDataColumnDescription;
        /**
         * RawDataData
         * @description Profile and representative rows from one uploaded table revision.
         */
        readonly RawDataData: Domain.RawDataData;
        /**
         * RawDataDateRange
         * @description Observed date bounds of the uploaded table, when it contains a date column.
         */
        readonly RawDataDateRange: Domain.RawDataDateRange;
        /** RecordDependency */
        readonly RecordDependency: Domain.RecordDependency;
        /** RecordedPath */
        readonly RecordedPath: Domain.RecordedPath;
        /** Rejected */
        readonly Rejected: Domain.Rejected;
        /** @enum {string} */
        readonly RejectionReason: Domain.RejectionReason;
        /** Removed[ConstructRef] */
        readonly Removed_ConstructRef_: Domain.Removed<Domain.ConstructRef>;
        /** Removed[DataPoint] */
        readonly Removed_DataPoint_: Domain.Removed<Domain.DataPoint>;
        /** Removed[EdgeRef] */
        readonly Removed_EdgeRef_: Domain.Removed<Domain.EdgeRef>;
        /** Removed[ParameterSpec] */
        readonly Removed_ParameterSpec_: Domain.Removed<Domain.ParameterSpec>;
        /** ResponseCurve */
        readonly ResponseCurve: Domain.ResponseCurve;
        /**
         * RetractedArtifact
         * @description A current artifact removed by an action, with the finding that caused it.
         */
        readonly RetractedArtifact: Domain.RetractedArtifact;
        /** Revised[ConstructRef] */
        readonly Revised_ConstructRef_: Domain.Revised<Domain.ConstructRef>;
        /** Revised[DataPoint] */
        readonly Revised_DataPoint_: Domain.Revised<Domain.DataPoint>;
        /** Revised[EdgeRef] */
        readonly Revised_EdgeRef_: Domain.Revised<Domain.EdgeRef>;
        /** Revised[ParameterSpec] */
        readonly Revised_ParameterSpec_: Domain.Revised<Domain.ParameterSpec>;
        /**
         * Role
         * @description A construct role states whether the variable is modeled as endogenous or treated as
         *     exogenous.
         * @enum {string}
         */
        readonly Role: "endogenous" | "exogenous";
        /** RunningAction */
        readonly RunningAction: Domain.RunningAction;
        /** RunningPoll */
        readonly RunningPoll: Domain.RunningPoll;
        /**
         * SamplerSpec
         * @description Fully resolved controls for the exact particle sampler.
         */
        readonly SamplerSpec: Domain.SamplerSpec;
        /** @enum {string} */
        readonly ScientificActionId: Domain.ScientificActionId;
        /**
         * SemanticExtractionSpec
         * @description Interpret source records using an explicit measurement rubric.
         */
        readonly "SemanticExtractionSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "semantic";
            /**
             * How To Measure
             * @description Scoring rubric and extraction instructions.
             */
            readonly how_to_measure: string;
            /**
             * Source Columns
             * @description Source columns exposed to the extraction worker.
             * @default []
             */
            readonly source_columns?: readonly string[];
        };
        /**
         * SemanticExtractionSpec
         * @description Interpret source records using an explicit measurement rubric.
         */
        readonly "SemanticExtractionSpec-Output": Domain.SemanticExtractionSpec;
        /**
         * SetQuestionRequest
         * @description Set the study question; it is the first action of every study.
         */
        readonly "SetQuestionRequest-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "set_question";
            readonly question: components["schemas"]["QuestionSpec-Input"];
        };
        /**
         * SetQuestionRequest
         * @description Set the study question; it is the first action of every study.
         */
        readonly "SetQuestionRequest-Output": Domain.SetQuestionRequest;
        /**
         * SimulateRequest
         * @description Generate a dated window with optional interventions; compare saved data with data_diff.
         */
        readonly "SimulateRequest-Input": {
            /**
             * Start
             * Format: date
             * @description Calendar day the window starts, at 00:00 UTC.
             */
            readonly start: string;
            /**
             * Horizon
             * @description How long the window lasts, such as 9w or 61d.
             */
            readonly horizon: string;
            /**
             * Interventions
             * @default []
             */
            readonly interventions?: readonly components["schemas"]["InterventionSpec-Input"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "simulate";
            readonly model_revision: components["schemas"]["GitOid-Input"];
        };
        /**
         * SimulateRequest
         * @description Generate a dated window with optional interventions; compare saved data with data_diff.
         */
        readonly "SimulateRequest-Output": Domain.SimulateRequest;
        /**
         * SimulationObservationLayout
         * @description Saved observation semantics and coordinates; generation truths remain separate.
         */
        readonly SimulationObservationLayout: Domain.SimulationObservationLayout;
        /**
         * SimulationPaths
         * @description Contiguous pages of original draws, with every recorded time point intact.
         */
        readonly SimulationPaths: Domain.SimulationPaths;
        /**
         * SimulationRef
         * @description A saved simulation; a null replicate selects all its recorded draws.
         */
        readonly "SimulationRef-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "simulation";
            readonly revision: components["schemas"]["GitOid-Input"];
            /** Replicate */
            readonly replicate?: number | null;
        };
        /**
         * SimulationRef
         * @description A saved simulation; a null replicate selects all its recorded draws.
         */
        readonly "SimulationRef-Output": Domain.SimulationRef;
        /**
         * SimulationReplicateRef
         * @description One replicate from a recorded, applied simulation in this study.
         */
        readonly "SimulationReplicateRef-Input": {
            readonly revision: components["schemas"]["GitOid-Input"];
            /** Replicate */
            readonly replicate: number;
        };
        /**
         * SimulationReplicateRef
         * @description One replicate from a recorded, applied simulation in this study.
         */
        readonly "SimulationReplicateRef-Output": Domain.SimulationReplicateRef;
        /**
         * SimulationReport
         * @description Generated histories and derived findings with their resolved execution coordinates.
         */
        readonly SimulationReport: Domain.SimulationReport;
        /**
         * SimulationSpec
         * @description Generate from a calendar day over a horizon, with interventions placed after the start.
         *
         *     The start is the only absolute time. A record's origin places it in model days;
         *     without a record, the start is model day zero.
         */
        readonly "SimulationSpec-Input": {
            /**
             * Start
             * Format: date
             * @description Calendar day the window starts, at 00:00 UTC.
             */
            readonly start: string;
            /**
             * Horizon
             * @description How long the window lasts, such as 9w or 61d.
             */
            readonly horizon: string;
            /**
             * Interventions
             * @default []
             */
            readonly interventions?: readonly components["schemas"]["InterventionSpec-Input"][];
        };
        /**
         * SimulationSpec
         * @description Generate from a calendar day over a horizon, with interventions placed after the start.
         *
         *     The start is the only absolute time. A record's origin places it in model days;
         *     without a record, the start is model day zero.
         */
        readonly "SimulationSpec-Output": Domain.SimulationSpec;
        /**
         * SourceValidity
         * @description Whether a selected artifact still matches its pinned inputs.
         * @enum {string}
         */
        readonly SourceValidity: Domain.SourceValidity;
        /** Sourced[DataProfileArtifact] */
        readonly Sourced_DataProfileArtifact_: Domain.Sourced<Domain.DataProfileArtifact>;
        /** Sourced[FitSummary] */
        readonly Sourced_FitSummary_: Domain.Sourced<Domain.FitSummary>;
        /** Sourced[IdentificationReport] */
        readonly Sourced_IdentificationReport_: Domain.Sourced<Domain.IdentificationReport>;
        /** Sourced[InferenceReport] */
        readonly Sourced_InferenceReport_: {
            readonly value: components["schemas"]["InferenceReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /** Sourced[MeasurementsData] */
        readonly Sourced_MeasurementsData_: Domain.Sourced<Domain.MeasurementsData>;
        /** Sourced[ModelPredictiveReport] */
        readonly Sourced_ModelPredictiveReport_: Domain.Sourced<Domain.ModelPredictiveReport>;
        /** Sourced[ModelSpec] */
        readonly Sourced_ModelSpec_: Domain.Sourced<Domain.ModelSpec>;
        /** Sourced[PreparedDataMetadata] */
        readonly Sourced_PreparedDataMetadata_: Domain.Sourced<Domain.PreparedDataMetadata>;
        /** Sourced[QuestionCheckReport] */
        readonly Sourced_QuestionCheckReport_: Domain.Sourced<Domain.QuestionCheckReport>;
        /** Sourced[QuestionSpec] */
        readonly Sourced_QuestionSpec_: Domain.Sourced<Domain.QuestionSpec>;
        /** Sourced[RawDataData] */
        readonly Sourced_RawDataData_: Domain.Sourced<Domain.RawDataData>;
        /** Sourced[SimulationReport] */
        readonly Sourced_SimulationReport_: Domain.Sourced<Domain.SimulationReport>;
        /** Sourced[ValidationReportArtifact] */
        readonly Sourced_ValidationReportArtifact_: Domain.Sourced<Domain.ValidationReportArtifact>;
        /** Sourced[tuple[SpecificationAssessment, ...]] */
        readonly Sourced_tuple_SpecificationAssessment__________: Domain.Sourced<readonly (Domain.SpecificationAssessment)[]>;
        /** Sourced[tuple[StructuralItemDisposition, ...]] */
        readonly Sourced_tuple_StructuralItemDisposition__________: Domain.Sourced<readonly (Domain.StructuralItemDisposition)[]>;
        readonly SpecificationAssessment: Domain.SpecificationAssessment;
        /**
         * StateAssignment
         * @description A state set at one model time: a resolved intervention or a replayed input reading.
         */
        readonly StateAssignment: Domain.StateAssignment;
        /**
         * StateExpression
         * @description A construct's state or declared known input, referenced by identity.
         */
        readonly "StateExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "state";
            readonly construct_id: components["schemas"]["ConstructId-Input"];
        };
        /**
         * StateExpression
         * @description A construct's state or declared known input, referenced by identity.
         */
        readonly "StateExpression-Output": Domain.StateExpression;
        /**
         * StepError
         * @description The error type and message of a failed step.
         */
        readonly StepError: Domain.StepError;
        /**
         * StepEvent
         * @description A data-preparation step changed status.
         */
        readonly StepEvent: Domain.StepEvent;
        /** @enum {string} */
        readonly StepStatus: Domain.StepStatus;
        /**
         * StructuralDisposition
         * @description A structural disposition classifies how compilation uses or excludes an authored model
         *     entity.
         * @enum {string}
         */
        readonly StructuralDisposition: Domain.StructuralDisposition;
        /**
         * StructuralItemDisposition
         * @description An item disposition explains the compilation decision for one identified authored
         *     entity.
         */
        readonly StructuralItemDisposition: Domain.StructuralItemDisposition;
        /** StudentTLawSpec[Expression] */
        readonly "StudentTLawSpec_Expression_-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "StudentT";
            readonly df: components["schemas"]["Expression-Input"];
            readonly loc: components["schemas"]["Expression-Input"];
            readonly scale: components["schemas"]["Expression-Input"];
        };
        /** StudentTLawSpec[Expression] */
        readonly "StudentTLawSpec_Expression_-Output": Domain.StudentTLawSpec<Domain.Expression>;
        /**
         * StudyRevision
         * @description Git publication wraps its already-owned record, without copying its fields.
         */
        readonly StudyRevision: Domain.StudyRevision;
        /**
         * StudyState
         * @description Study state projects the artifact trees selected by one Git commit.
         *
         *     ``current`` maps artifact id → the revision info that is *current* for the
         *     study. Absent key = the artifact does not exist (either never produced,
         *     or produced-when-nonempty semantics withheld it).
         */
        readonly StudyState: Domain.StudyState;
        /**
         * StudyStatus
         * @description Study status reports committed artifacts, their freshness, actions, and any running one.
         */
        readonly StudyStatus: Domain.StudyStatus;
        /**
         * SummaryOperator
         * @description A summary operator specifies how values within a measurement window produce one
         *     observation.
         * @enum {string}
         */
        readonly SummaryOperator: "first" | "last" | "sum" | "count" | "mean" | "std";
        /**
         * TemperingDiagnostics
         * @description Retained tempering telemetry, separate from evidence of a production engine.
         */
        readonly TemperingDiagnostics: Domain.TemperingDiagnostics;
        /**
         * TemporalStatus
         * @description Temporal status states whether a construct varies within the individual over time.
         * @enum {string}
         */
        readonly TemporalStatus: "time_varying" | "time_invariant";
        /**
         * TimelineResponse
         * @description Typed attempt journal returned by the study read plane.
         */
        readonly TimelineResponse: Domain.TimelineResponse;
        /** ToolCallRequest */
        readonly ToolCallRequest: {
            /**
             * Branch
             * @default main
             */
            readonly branch?: string;
            readonly expected_head?: components["schemas"]["GitOid-Input"] | null;
            /** Workspace Id */
            readonly workspace_id: string;
            readonly input: components["schemas"]["JsonObject-Input"];
        };
        /** ToolResult */
        readonly ToolResult: {
            readonly result: components["schemas"]["JsonValue-Output"];
            readonly context_output?: components["schemas"]["JsonObject-Output"] | null;
        };
        /** ToolSchema */
        readonly ToolSchema: {
            /** Name */
            readonly name: string;
            /** Description */
            readonly description: string;
            /** Parameters */
            readonly parameters: {
                readonly [key: string]: unknown;
            };
            /** Result */
            readonly result: {
                readonly [key: string]: unknown;
            } | null;
        };
        /**
         * TraceMessage
         * @description A trace message records one conversational step, including any reasoning or tool
         *     interaction.
         */
        readonly TraceMessage: Domain.TraceMessage;
        /**
         * TraceSeries
         * @description Every retained draw, grouped in original chain order.
         */
        readonly TraceSeries: Domain.TraceSeries;
        /**
         * TraceToolCall
         * @description A function invocation with the call identity used to match its result.
         */
        readonly TraceToolCall: Domain.TraceToolCall;
        /**
         * TraceUsage
         * @description Trace usage records the input, output, and reasoning tokens consumed by a conversation.
         */
        readonly TraceUsage: Domain.TraceUsage;
        /**
         * Unavailable
         * @description An applicable result could not be produced, for an explicit reason.
         */
        readonly Unavailable: Domain.Unavailable;
        /**
         * UnavailablePredictiveChecks
         * @description The run could not evaluate its scientific battery.
         */
        readonly UnavailablePredictiveChecks: Domain.UnavailablePredictiveChecks;
        /** Unchanged[ConstructRef] */
        readonly Unchanged_ConstructRef_: Domain.Unchanged<Domain.ConstructRef>;
        /** Unchanged[EdgeRef] */
        readonly Unchanged_EdgeRef_: Domain.Unchanged<Domain.EdgeRef>;
        /**
         * UnknownLawProvenance
         * @description Imported laws do not establish a conditioning history.
         */
        readonly UnknownLawProvenance: Domain.UnknownLawProvenance;
        /** ValidationError */
        readonly ValidationError: {
            /** Location */
            readonly loc: readonly (string | number)[];
            /** Message */
            readonly msg: string;
            /** Error Type */
            readonly type: string;
            /** Input */
            readonly input?: unknown;
            /** Context */
            readonly ctx?: Record<string, never>;
        };
        /**
         * ValidationIssue
         * @description A validation issue explains a data problem and its severity for an indicator or the
         *     dataset.
         */
        readonly ValidationIssue: Domain.ValidationIssue;
        /**
         * ValidationReportArtifact
         * @description Data findings composed with model-dependent execution checks.
         */
        readonly ValidationReportArtifact: Domain.ValidationReportArtifact;
        /** @description Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values. */
        readonly WindowExpression: string;
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    readonly execute_scientific_action_api_studies__workspace_id__actions_post: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly expected_head?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["SetQuestionRequest-Input"] | components["schemas"]["EditModelRequest-Input"] | components["schemas"]["PrepareDataRequest-Input"] | components["schemas"]["FitRequest-Input"] | components["schemas"]["SimulateRequest-Input"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 202: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ActionReceipt"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly poll_scientific_action_api_studies__workspace_id__actions__attempt_id__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly attempt_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_study_api_studies__workspace_id__get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["StudyStatus"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_snapshot_api_studies__workspace_id__model_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ModelSnapshot"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_revisions_api_studies__workspace_id__revisions_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["ArtifactRecord"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly read_model_revision_api_studies__workspace_id__revisions_model__revision__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly revision: components["schemas"]["GitOid-Input"];
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ModelSpec-Output"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly model_diff: {
        readonly parameters: {
            readonly query: {
                readonly before: components["schemas"]["GitOid-Input"];
                readonly after: components["schemas"]["GitOid-Input"];
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ModelDiffReport"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly data_diff: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["DataDiffRequest-Input"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 202: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ActionReceipt"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_data_diff_api_studies__workspace_id__data_diff__commit_id__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly commit_id: components["schemas"]["GitOid-Input"];
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["DataDiffReport"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly read_data_profile_api_studies__workspace_id__revisions_data_profile__panel_revision__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly panel_revision: components["schemas"]["GitOid-Input"];
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["DataProfileArtifact"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_definition_api_studies__workspace_id__model_definition_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["Sourced_ModelSpec_"] | null;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_inference_report_api_studies__workspace_id__model_inference_report_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["Sourced_InferenceReport_"] | null;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_observation_history_api_studies__workspace_id__model_visuals_observations__indicator_id__get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly indicator_id: components["schemas"]["IndicatorId-Input"];
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ObservationHistory"] | null;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_predictive_history_api_studies__workspace_id__model_visuals_predictive__indicator_id__get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly indicator_id: components["schemas"]["IndicatorId-Input"];
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["PPCOverlay"] | null;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_simulation_paths_api_studies__workspace_id__model_visuals_simulation_get: {
        readonly parameters: {
            readonly query?: {
                readonly start?: number;
                readonly count?: number;
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["SimulationPaths"] | null;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_parameter_draws_api_studies__workspace_id__model_visuals_parameters_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ParameterDraws"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_mechanism_curves_api_studies__workspace_id__model_visuals_mechanism_post: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["MechanismViewRequest"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["MechanismCurves"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_constructs_api_studies__workspace_id__model_constructs_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["ConstructSpec-Output"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_edges_api_studies__workspace_id__model_edges_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["CausalEdgeSpec-Output"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_indicators_api_studies__workspace_id__model_indicators_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["IndicatorSpec-Output"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_model_parameters_api_studies__workspace_id__model_parameters_get: {
        readonly parameters: {
            readonly query?: {
                readonly branch?: string;
                readonly at?: components["schemas"]["GitOid-Input"] | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["ParameterSpec-Output"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_timeline_api_studies__workspace_id__timeline_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["TimelineResponse"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_branches_api_studies__workspace_id__branches_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": {
                        readonly [key: string]: components["schemas"]["GitOid-Output"];
                    };
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly create_branch_api_studies__workspace_id__branches_post: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["CreateBranchBody"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["GitOid-Output"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_attempt_log_api_studies__workspace_id__logs__commit_id__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly commit_id: components["schemas"]["GitOid-Input"];
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["StudyRevision"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_events_api_studies__workspace_id__events_get: {
        readonly parameters: {
            readonly query: {
                readonly attempt_id: string;
                readonly after?: string | null;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["ProgressEvent"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_artifact_api_studies__workspace_id__artifacts__artifact_id__get: {
        readonly parameters: {
            readonly query?: {
                readonly revision?: components["schemas"]["GitOid-Input"] | null;
                readonly branch?: string;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly artifact_id: "question" | "raw_data" | "model" | "identification_report" | "panel" | "data_profile" | "validation_report";
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ArtifactEnvelope"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_artifact_traces_api_studies__workspace_id__artifacts__artifact_id__traces_get: {
        readonly parameters: {
            readonly query?: {
                readonly revision?: components["schemas"]["GitOid-Input"] | null;
                readonly branch?: string;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly artifact_id: "question" | "raw_data" | "model" | "identification_report" | "panel" | "data_profile" | "validation_report";
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["AttemptTraceIndex"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_trace_api_studies__workspace_id__traces__commit_id___subroutine_id__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly commit_id: components["schemas"]["GitOid-Input"];
                readonly subroutine_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["LLMTrace"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_artifact_file_api_studies__workspace_id__artifacts__artifact_id__files__filename__get: {
        readonly parameters: {
            readonly query?: {
                readonly revision?: components["schemas"]["GitOid-Input"] | null;
                readonly branch?: string;
            };
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly artifact_id: "question" | "raw_data" | "model" | "identification_report" | "panel" | "data_profile" | "validation_report";
                readonly filename: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": unknown;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_actions_enabled_api_actions_enabled_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": boolean;
                };
            };
        };
    };
    readonly list_workspaces_api_workspaces_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": {
                        readonly [key: string]: string | null;
                    };
                };
            };
        };
    };
    readonly upload_file_api_upload_post: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "multipart/form-data": components["schemas"]["Body_upload_file_api_upload_post"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": string;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_tool_schemas_api_tools__context_id__get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly context_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": readonly components["schemas"]["ToolSchema"][];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly execute_tool_api_tools__context_id___tool_name__post: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly context_id: string;
                readonly tool_name: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["ToolCallRequest"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["ToolResult"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
