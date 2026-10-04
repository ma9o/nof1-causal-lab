/** AUTO-GENERATED from Python's OpenAPI graph. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
    readonly "/api/studies/{workspace_id}/set_question": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Set Question
         * @description Set the immutable study question first. Repeat parsed arguments to read saved results or running progress; failed calls may be retried.
         */
        readonly post: operations["set_question"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/edit_model": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Edit Model
         * @description Save a model naming its base expected_revision and optional panel_revision. No head conflict check. The complete result includes findings, draws, histories, artifacts and traces; use model_diff separately for changes.
         */
        readonly post: operations["edit_model"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/prepare_data": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Prepare Data
         * @description Prepare named uploaded files or a saved simulation replicate. Capture each named file's SHA-256 at call time. Repeat the retained source.hashes to read saved results without uploaded bytes. Running results include step/extraction events; completed results include all observations, profiles and traces.
         */
        readonly post: operations["prepare_data"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/fit": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Fit
         * @description Condition the named model_revision on panel_revision. Returns the saved complete inference result, including joint posterior arrays, diagnostics and every observation history. Repeat the same arguments to read progress or the saved completion.
         */
        readonly post: operations["fit"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/simulate": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Simulate
         * @description Simulate the named model_revision and optional panel_revision. Fitted laws retain their fit origin. Returns all paths and arrays without paging, their summaries, causal evidence and traces. Repeat the same arguments to read progress or saved completion.
         */
        readonly post: operations["simulate"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/data_diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Data Diff
         * @description Compare immutable left/right data selections and retain a comparison leaf. Identical applied calls reuse the complete comparison without another attempt; identical running calls return that attempt's progress. Failures stay in the timeline and can be retried.
         */
        readonly post: operations["data_diff"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model_diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Model Diff
         * @description Compare named before/after model trees or checkpoints, including their definitions and evidence. Cached by parsed arguments. Never creates an attempt, leaf, or timeline entry, and works on the read-only facade.
         */
        readonly post: operations["model_diff"];
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
         * @description Slim call log: replayable arguments, status, messages, errors, trace ids and dependencies. No inline results and no branches. Selecting an applied node repeats its action; never repeat failed or unknown nodes from the viewer.
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
    readonly "/api/workspaces": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * List Workspaces
         * @description Available workspaces and their immutable study questions.
         *
         *     X-Actions-Enabled preserves the landing page's deployment capability without a separate endpoint.
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
         * @description Stage one named raw input file for prepare_data; that call captures its SHA-256.
         */
        readonly post: operations["upload_file_api_upload_post"];
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
        /** @description A closed action attempt pairs its request with only that action's successful result or failure outcome. */
        readonly ActionAttempt: Domain.Attempt<"set_question", Domain.SetQuestionRequest, null> | Domain.Attempt<"edit_model", Domain.EditModelRequest, null> | Domain.Attempt<"prepare_data", Domain.PrepareDataRequest, Domain.DataPreparationResult> | Domain.Attempt<"fit", Domain.FitRequest, Domain.ModelFitResult | null> | Domain.Attempt<"simulate", Domain.SimulateRequest, Domain.ModelSimulationResult> | Domain.Attempt<"data_diff", Domain.DataDiffRequest, null>;
        /**
         * ActionEffects
         * @description What an executed action did to the store: the workflow installs this.
         */
        readonly ActionEffects: {
            /** Produced */
            readonly produced: readonly components["schemas"]["ArtifactRecord"][];
            /** Retracted */
            readonly retracted: readonly components["schemas"]["RetractedArtifact"][];
        };
        readonly ActionId: components["schemas"]["ScientificActionId"] | "data_diff";
        /**
         * ActionMessage
         * @description A label emitted by an attempt; measurements belong in its scientific result.
         */
        readonly ActionMessage: {
            /**
             * Timestamp
             * Format: date-time
             */
            readonly timestamp: string;
            /**
             * Level
             * @enum {string}
             */
            readonly level: "debug" | "info" | "warn" | "error";
            /** Label */
            readonly label: string;
        };
        /** @description A call returns running arguments, messages and progress, or its complete saved outcome and scientific views. */
        readonly ActionPoll: components["schemas"]["RunningPoll"] | components["schemas"]["CompletedPoll"];
        /** Added[ConstructRef] */
        readonly Added_ConstructRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "added";
            readonly after: components["schemas"]["ConstructRef-Output"];
        };
        /** Added[DataPoint] */
        readonly Added_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "added";
            readonly after: components["schemas"]["DataPoint"];
        };
        /** Added[EdgeRef] */
        readonly Added_EdgeRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "added";
            readonly after: components["schemas"]["EdgeRef"];
        };
        /** Added[ParameterSpec] */
        readonly Added_ParameterSpec_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "added";
            readonly after: components["schemas"]["ParameterSpec-Output"];
        };
        /** Applied[DataPreparationResult] */
        readonly Applied_DataPreparationResult_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "applied";
            readonly result: components["schemas"]["DataPreparationResult"];
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /** Applied[ModelSimulationResult] */
        readonly Applied_ModelSimulationResult_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "applied";
            readonly result: components["schemas"]["ModelSimulationResult"];
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /** Applied[NoneType] */
        readonly Applied_NoneType_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "applied";
            /** Result */
            readonly result: null;
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /** Applied[Union[ModelFitResult, NoneType]] */
        readonly Applied_Union_ModelFitResult__NoneType__: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "applied";
            readonly result: components["schemas"]["ModelFitResult"] | null;
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /**
         * @description An artifact identity selects one node in the study's artifact graph.
         * @enum {string}
         */
        readonly ArtifactId: "question" | "raw_data" | "model" | "panel";
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
        readonly ArtifactRecord: {
            readonly artifact_id: components["schemas"]["ArtifactId"];
            readonly revision: components["schemas"]["GitOid-Output"];
            /** Derived From */
            readonly derived_from: Readonly<Partial<Record<components["schemas"]["ArtifactId"], components["schemas"]["GitOid-Output"]>>>;
            /**
             * Produced By
             * @default null
             */
            readonly produced_by: string | null;
            /**
             * Created At
             * @default
             */
            readonly created_at: string;
        };
        readonly Assessment_ConvergenceAssessmentSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence> | Domain.NotEvaluated<Domain.ConvergenceAssessmentSubject>;
        readonly Assessment_IndicatorCheckSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence> | Domain.NotEvaluated<Domain.IndicatorCheckSubject>;
        readonly Assessment_str_ParticleMCMCEvidence_: Domain.Evaluated<string, Domain.ParticleMCMCEvidence> | Domain.NotEvaluated<string>;
        /**
         * AttemptRecord
         * @description Stored inside the Git object, with no self-referential publication ID.
         */
        readonly AttemptRecord: {
            /** Seq */
            readonly seq: number;
            /**
             * Attempt Id
             * @default null
             */
            readonly attempt_id: string | null;
            /** Ts */
            readonly ts: string;
            /**
             * Messages
             * @default []
             */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /**
             * Trace Ids
             * @default []
             */
            readonly trace_ids: readonly string[];
            readonly attempt: components["schemas"]["ActionAttempt"];
        };
        /** Attempt[ActionId, Union[ScientificActionRequest, DataDiffRequest], NoneType] */
        readonly Attempt_ActionId_Union_ScientificActionRequest__DataDiffRequest__NoneType_: {
            readonly action: components["schemas"]["ActionId"];
            /**
             * Request
             * @description Parsed arguments, or null for a historical attempt whose arguments were not retained
             */
            readonly request: components["schemas"]["ScientificActionRequest"] | components["schemas"]["DataDiffRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['data_diff'], DataDiffRequest, NoneType] */
        readonly Attempt_Literal__data_diff___DataDiffRequest_NoneType_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "data_diff";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["DataDiffRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['edit_model'], EditModelRequest, NoneType] */
        readonly Attempt_Literal__edit_model___EditModelRequest_NoneType_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_model";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["EditModelRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['fit'], FitRequest, Union[ModelFitResult, NoneType]] */
        readonly Attempt_Literal__fit___FitRequest_Union_ModelFitResult__NoneType__: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "fit";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["FitRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.ModelFitResult | null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['prepare_data'], PrepareDataRequest, DataPreparationResult] */
        readonly Attempt_Literal__prepare_data___PrepareDataRequest_DataPreparationResult_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["PrepareDataRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.DataPreparationResult> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['set_question'], SetQuestionRequest, NoneType] */
        readonly Attempt_Literal__set_question___SetQuestionRequest_NoneType_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "set_question";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["SetQuestionRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /** Attempt[Literal['simulate'], SimulateRequest, ModelSimulationResult] */
        readonly Attempt_Literal__simulate___SimulateRequest_ModelSimulationResult_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "simulate";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["SimulateRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.ModelSimulationResult> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * AuthoredLawProvenance
         * @description The current laws have authored ancestry without retained fitting.
         */
        readonly AuthoredLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "authored";
            /**
             * Interpretation
             * @constant
             */
            readonly interpretation: "prior_predictive";
        };
        /** Available[CausalEffectResult] */
        readonly Available_CausalEffectResult_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "available";
            readonly value: components["schemas"]["CausalEffectResult"];
        };
        /** Available[PosteriorPredictiveChecks] */
        readonly Available_PosteriorPredictiveChecks_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "available";
            readonly value: components["schemas"]["PosteriorPredictiveChecks"];
        };
        /** Available[tuple[ParameterDrawColumn, ...]] */
        readonly Available_tuple_ParameterDrawColumn__________: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "available";
            /** Value */
            readonly value: readonly components["schemas"]["ParameterDrawColumn"][];
        };
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
        readonly "BernoulliLogitsLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliLogits";
            readonly logits: components["schemas"]["Expression-Output"];
        };
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
        readonly "BernoulliProbsLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliProbs";
            readonly probs: components["schemas"]["Expression-Output"];
        };
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
        readonly "BetaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Beta";
            readonly concentration1: components["schemas"]["Expression-Output"];
            readonly concentration0: components["schemas"]["Expression-Output"];
        };
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
        readonly "BinaryExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "binary";
            readonly operator: components["schemas"]["BinaryOperator"];
            readonly left: components["schemas"]["Expression-Output"];
            readonly right: components["schemas"]["Expression-Output"];
        };
        /**
         * @description A binary operator combines two scalar expression operands.
         * @enum {string}
         */
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
        readonly "CallExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "call";
            readonly function: components["schemas"]["ExpressionFunction"];
            /** Arguments */
            readonly arguments: readonly components["schemas"]["Expression-Output"][];
        };
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
        readonly "CategoricalLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Categorical";
            readonly logits: components["schemas"]["Expression-Output"];
        };
        /**
         * CategoryProbabilitySummary
         * @description Predictive probabilities for each declared level; unobserved anchors are null.
         */
        readonly CategoryProbabilitySummary: {
            /** Probabilities */
            readonly probabilities: {
                readonly [key: string]: readonly (number | null)[];
            };
            /** N Draws */
            readonly n_draws: readonly number[];
        };
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
        readonly "CausalEdgeSpec-Output": {
            /** @description Persistent identity. Preserve when revising the same edge. */
            readonly id: components["schemas"]["EdgeId-Output"];
            /**
             * Mechanisms
             * @default []
             */
            readonly mechanisms: readonly components["schemas"]["DriftMechanismSpec-Output"][];
            /** @description Cause construct; shared endpoints have one identity. */
            readonly cause: components["schemas"]["ConstructSpec-Output"] | components["schemas"]["ConstructRef-Output"];
            /** @description Effect construct; shared endpoints have one identity. */
            readonly effect: components["schemas"]["ConstructSpec-Output"] | components["schemas"]["ConstructRef-Output"];
            /**
             * Description
             * @description Theoretical justification for this causal link
             */
            readonly description: string;
            /**
             * Sources
             * @description Literature sources supporting this causal link
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
        /**
         * CausalEffectResult
         * @description Causal effects and realized trajectories under the enclosing report's design.
         */
        readonly CausalEffectResult: {
            readonly outcome: components["schemas"]["ConstructId-Output"];
            /** Labels */
            readonly labels: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], string>>>;
            /** Warnings */
            readonly warnings: readonly string[];
        };
        /**
         * ChainDiagnostics
         * @description Compact retained-chain measurements; plot series compose the report detail.
         */
        readonly ChainDiagnostics: {
            /** Num Chains */
            readonly num_chains: number;
            /** Num Samples */
            readonly num_samples: number;
            /** Per Parameter */
            readonly per_parameter: readonly components["schemas"]["ParameterDiagnostics"][];
            /**
             * Num Divergences
             * @default null
             */
            readonly num_divergences: number | null;
            /**
             * Divergence Rate
             * @default null
             */
            readonly divergence_rate: number | null;
            /**
             * Tree Depth Mean
             * @default null
             */
            readonly tree_depth_mean: number | null;
            /**
             * Tree Depth Max
             * @default null
             */
            readonly tree_depth_max: number | null;
            /**
             * Accept Prob Mean
             * @default null
             */
            readonly accept_prob_mean: number | null;
            /**
             * Latent Accept Prob Mean
             * @default null
             */
            readonly latent_accept_prob_mean: number | null;
            /**
             * Parameter Accept Prob Mean
             * @default null
             */
            readonly parameter_accept_prob_mean: number | null;
            /** @default null */
            readonly energy: components["schemas"]["EnergyDiagnostics"] | null;
        };
        readonly Change_ConstructRef_: Domain.Added<Domain.ConstructRef> | Domain.Removed<Domain.ConstructRef> | Domain.Revised<Domain.ConstructRef>;
        readonly Change_DataPoint_: Domain.Added<Domain.DataPoint> | Domain.Removed<Domain.DataPoint> | Domain.Revised<Domain.DataPoint>;
        readonly Change_EdgeRef_: Domain.Added<Domain.EdgeRef> | Domain.Removed<Domain.EdgeRef> | Domain.Revised<Domain.EdgeRef>;
        readonly Change_ParameterSpec_: Domain.Added<Domain.ParameterSpec> | Domain.Removed<Domain.ParameterSpec> | Domain.Revised<Domain.ParameterSpec>;
        /**
         * @description A group of model checks is selected by the inputs it consumes.
         * @enum {string}
         */
        readonly CheckGroup: "specification" | "identification" | "compatibility" | "question";
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
             * @default null
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
        readonly "CoefficientExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "coefficient";
            readonly role: components["schemas"]["CoefficientRole"];
            /**
             * Value
             * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
             * @default null
             */
            readonly value: number | components["schemas"]["ParameterId-Output"] | null;
            /**
             * Construct Ids
             * @description Additional constructs participating in this coefficient use.
             * @default []
             */
            readonly construct_ids: readonly components["schemas"]["ConstructId-Output"][];
        };
        /**
         * @description A coefficient role identifies an expression operand’s scientific quantity and support.
         * @enum {string}
         */
        readonly CoefficientRole: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
        /**
         * CompletedExtractionWorker
         * @description Retained measurements from a completed worker; no failure field exists.
         */
        readonly CompletedExtractionWorker: {
            /** Worker Id */
            readonly worker_id: number;
            /** N Extractions */
            readonly n_extractions: number;
            /** N Windows */
            readonly n_windows: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "completed";
        };
        /** CompletedPoll */
        readonly CompletedPoll: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "completed";
            readonly commit_id: components["schemas"]["GitOid-Output"] | null;
            readonly attempt: components["schemas"]["ActionAttempt"];
            /**
             * Messages
             * @default []
             */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /** @default null */
            readonly snapshot: components["schemas"]["ModelSnapshot"] | null;
            /** @default null */
            readonly inference_report: Domain.Sourced<Domain.InferenceReport> | null;
            /** @default null */
            readonly data_comparison: components["schemas"]["DataDiffReport"] | null;
            /** @default null */
            readonly checks: components["schemas"]["ModelCheckReport"] | null;
            /** Observation Histories */
            readonly observation_histories: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["ObservationHistory"]>>>;
            /** Predictive Overlays */
            readonly predictive_overlays: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["PPCOverlay"]>>>;
            /** @default null */
            readonly simulation_paths: components["schemas"]["SimulationPaths"] | null;
            /** @default null */
            readonly parameter_draws: components["schemas"]["ParameterDraws"] | null;
            /** Traces */
            readonly traces: {
                readonly [key: string]: components["schemas"]["LLMTrace"];
            };
            /** Artifacts */
            readonly artifacts: {
                readonly [key: string]: Domain.JsonObject;
            };
            /** Arrays */
            readonly arrays: {
                readonly [key: string]: Domain.JsonValue;
            };
        };
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
            readonly source_columns: readonly [
                string,
                ...string[]
            ];
            /**
             * @description Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation.
             * @default null
             */
            readonly computed_rule?: components["schemas"]["WindowExpression"] | null;
            /**
             * Fill Null
             * @description Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
             * @default null
             */
            readonly fill_null?: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
            /**
             * Fill Null Limit
             * @description Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
             * @default null
             */
            readonly fill_null_limit?: number | null;
        };
        /**
         * ComputedExtractionSpec
         * @description Compute a deterministic support-window measurement from source columns.
         */
        readonly "ComputedExtractionSpec-Output": {
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
            readonly source_columns: readonly [
                string,
                ...string[]
            ];
            /**
             * @description Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation.
             * @default null
             */
            readonly computed_rule: components["schemas"]["WindowExpression"] | null;
            /**
             * Fill Null
             * @description Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
             * @default null
             */
            readonly fill_null: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
            /**
             * Fill Null Limit
             * @description Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
             * @default null
             */
            readonly fill_null_limit: number | null;
        };
        /** @description A persistent construct identity survives changes to its display name. */
        readonly "ConstructId-Input": `construct:${string}`;
        /** @description A persistent construct identity survives changes to its display name. */
        readonly "ConstructId-Output": `construct:${string}`;
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
        readonly "ConstructRef-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "construct";
            readonly id: components["schemas"]["ConstructId-Output"];
        };
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
            /**
             * @description Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points.
             * @default null
             */
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
        readonly "ConstructSpec-Output": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["ConstructId-Output"];
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
            readonly indicators: readonly components["schemas"]["IndicatorSpec-Output"][];
            /**
             * Dynamics
             * @default []
             */
            readonly dynamics: readonly components["schemas"]["DynamicsMechanismSpec-Output"][];
            /**
             * Coefficients
             * @default []
             */
            readonly coefficients: readonly components["schemas"]["CoefficientExpression-Output"][];
            /**
             * Innovation Family
             * @default gaussian
             * @enum {string}
             */
            readonly innovation_family: "gaussian" | "student_t";
            /**
             * @description Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points.
             * @default null
             */
            readonly distribution: components["schemas"]["DistributionId-Output"] | null;
            /** @description 'endogenous' means modeled, with or without parents; 'exogenous' means given through direct exact readings, with no law. Unmeasured constructs are endogenous. */
            readonly role: components["schemas"]["Role"];
            /** @description 'time_varying' (changes over time) or 'time_invariant' (fixed) */
            readonly temporal_status: components["schemas"]["TemporalStatus"];
        };
        readonly ConvergenceAssessmentSubject: components["schemas"]["ConvergenceSubject"] | "recorded_parameter_chains";
        /**
         * ConvergenceCriterion
         * @enum {string}
         */
        readonly ConvergenceCriterion: "r_hat" | "ess_bulk" | "ess_tail";
        /**
         * ConvergenceSubject
         * @description A convergence criterion on one stable scientific scalar.
         */
        readonly ConvergenceSubject: {
            readonly parameter: components["schemas"]["ParameterRef"];
            readonly criterion: components["schemas"]["ConvergenceCriterion"];
            /** Label */
            readonly label: string;
        };
        /** @enum {string} */
        readonly DSMCLeafProposal: "amala_exact" | "paid_mix";
        /**
         * DataDiffReport
         * @description Comparisons of existing data, preserving each history's immutable source reference.
         */
        readonly DataDiffReport: {
            /** Left */
            readonly left: readonly components["schemas"]["DataRef-Output"][];
            /** Right */
            readonly right: readonly components["schemas"]["DataRef-Output"][];
            /** Variables */
            readonly variables: readonly components["schemas"]["DataVariableDiff"][];
        };
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
        readonly "DataDiffRequest-Output": {
            /**
             * Action
             * @default data_diff
             * @constant
             */
            readonly action: "data_diff";
            readonly left: components["schemas"]["DataSelection-Output"];
            readonly right: components["schemas"]["DataSelection-Output"];
        };
        /**
         * DataPoint
         * @description An observed anchor and support; dates are synthetic for a calendar-free series.
         */
        readonly DataPoint: {
            /**
             * Anchor Time
             * Format: date-time
             */
            readonly anchor_time: string;
            /** Support Start */
            readonly support_start: string | null;
            /** Support End */
            readonly support_end: string | null;
            /** Value */
            readonly value: number | null;
        };
        /**
         * DataPreparationResult
         * @description Preparation artifacts and the measurements actually retained by extraction.
         */
        readonly DataPreparationResult: {
            /**
             * Workers
             * @default []
             */
            readonly workers: readonly components["schemas"]["ExtractionWorkerResult"][];
            /**
             * Ingestion Reused
             * @default null
             */
            readonly ingestion_reused: boolean | null;
            /**
             * Extraction Reused
             * @default null
             */
            readonly extraction_reused: number | null;
        };
        /**
         * DataPreparationSpec
         * @description A versioned data definition supplied directly to prepare_data.
         */
        readonly "DataPreparationSpec-Input": {
            /** Default Window */
            readonly default_window: string;
            /** Variables */
            readonly variables: readonly [
                components["schemas"]["DataVariableSpec-Input"],
                ...components["schemas"]["DataVariableSpec-Input"][]
            ];
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
        readonly "DataPreparationSpec-Output": {
            /** Default Window */
            readonly default_window: string;
            /** Variables */
            readonly variables: readonly [
                components["schemas"]["DataVariableSpec-Output"],
                ...components["schemas"]["DataVariableSpec-Output"][]
            ];
            /**
             * Context
             * @description Optional context for interpreting the source data.
             * @default
             */
            readonly context: string;
        };
        /**
         * DataProfileArtifact
         * @description Model-independent empirical measurements and data-quality findings.
         */
        readonly DataProfileArtifact: {
            /** Indicators */
            readonly indicators: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["IndicatorAudit"]>>>;
            /** Dataset Issues */
            readonly dataset_issues: readonly components["schemas"]["ValidationIssue"][];
            /**
             * Is Valid
             * @description Whether the data findings contain no errors, independent of any model.
             */
            readonly is_valid: boolean;
        };
        readonly "DataRef-Input": components["schemas"]["PanelRef-Input"] | components["schemas"]["SimulationRef-Input"];
        readonly "DataRef-Output": components["schemas"]["PanelRef-Output"] | components["schemas"]["SimulationRef-Output"];
        /** @description A data selection identifies one or more saved observation histories. */
        readonly "DataSelection-Input": components["schemas"]["DataRef-Input"] | readonly components["schemas"]["DataRef-Input"][];
        /** @description A data selection identifies one or more saved observation histories. */
        readonly "DataSelection-Output": components["schemas"]["DataRef-Output"] | readonly components["schemas"]["DataRef-Output"][];
        /**
         * DataSeries
         * @description One variable's recorded measurements in one history; no pooling across replicas.
         */
        readonly DataSeries: {
            readonly variable: Domain.ObservationSpec<string> | null;
            /**
             * Time Origin
             * @description Recorded calendar binding; null means the point dates are serialization coordinates, not real dates.
             */
            readonly time_origin: string | null;
            /** Points */
            readonly points: readonly components["schemas"]["DataPoint"][];
        };
        /** @enum {string} */
        readonly DataStatistic: "observed_count" | "missing_count" | "mean" | "sd" | "min" | "max" | "proportion";
        /**
         * DataStatisticComparison
         * @description The same descriptive statistic measured independently in every selected history.
         */
        readonly DataStatisticComparison: {
            readonly statistic: components["schemas"]["DataStatistic"];
            /**
             * Level
             * @default null
             */
            readonly level: string | null;
            /** Left */
            readonly left: readonly (number | null)[];
            /** Right */
            readonly right: readonly (number | null)[];
            /** Left Histogram */
            readonly left_histogram: readonly components["schemas"]["HistogramBin"][];
            /** Right Histogram */
            readonly right_histogram: readonly components["schemas"]["HistogramBin"][];
        };
        /**
         * DataVariableDiff
         * @description Definitions, histories and comparisons for one persistent observation identity.
         */
        readonly DataVariableDiff: {
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /** Left */
            readonly left: readonly components["schemas"]["DataSeries"][];
            /** Right */
            readonly right: readonly components["schemas"]["DataSeries"][];
            /** Changes */
            readonly changes: readonly Domain.Change<Domain.DataPoint>[];
            /** Statistics */
            readonly statistics: readonly components["schemas"]["DataStatisticComparison"][];
            /** Comparison Issues */
            readonly comparison_issues: readonly string[];
            readonly predictive: components["schemas"]["PredictiveComparisonResult"];
        };
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
        readonly "DataVariableSpec-Output": {
            readonly observation: Domain.ObservationSpec<string | null>;
            readonly extraction: components["schemas"]["ExtractionSpec-Output"];
        };
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
        readonly "DeltaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Delta";
            readonly v: components["schemas"]["Expression-Output"];
        };
        /**
         * DensityCurve
         * @description Aligned density ordinates; the owning field distinguishes PDF samples from histogram heights.
         */
        readonly DensityCurve: {
            /**
             * X
             * @default []
             */
            readonly x: readonly number[];
            /**
             * Density
             * @default []
             */
            readonly density: readonly number[];
        };
        readonly "DistributionId-Input": `distribution:${string}`;
        readonly "DistributionId-Output": `distribution:${string}`;
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
        readonly "DriftMechanismSpec-Output": {
            readonly id: components["schemas"]["MechanismId-Output"];
            readonly expression: components["schemas"]["Expression-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "drift";
        };
        /** @description A dynamics mechanism declares one contribution to continuous-time drift. */
        readonly "DynamicsMechanismSpec-Input": components["schemas"]["DriftMechanismSpec-Input"] | components["schemas"]["PotentialMechanismSpec-Input"];
        /** @description A dynamics mechanism declares one contribution to continuous-time drift. */
        readonly "DynamicsMechanismSpec-Output": components["schemas"]["DriftMechanismSpec-Output"] | components["schemas"]["PotentialMechanismSpec-Output"];
        /** @description A persistent edge identity identifies one authored causal relationship. */
        readonly "EdgeId-Input": `edge:${string}`;
        /** @description A persistent edge identity identifies one authored causal relationship. */
        readonly "EdgeId-Output": `edge:${string}`;
        /**
         * EdgeRef
         * @description An edge reference identifies a causal relationship independently of edits to its
         *     definition.
         */
        readonly EdgeRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "edge";
            readonly id: components["schemas"]["EdgeId-Output"];
        };
        /**
         * EditModelRequest
         * @description Replace one named base revision with a validated scientific definition.
         */
        readonly "EditModelRequest-Input": {
            /**
             * Action
             * @default edit_model
             * @constant
             */
            readonly action?: "edit_model";
            readonly expected_revision: components["schemas"]["GitOid-Input"] | null;
            /**
             * @description Exact observations used by this edit's checks; omitted runs no predictive check.
             * @default null
             */
            readonly panel_revision?: components["schemas"]["GitOid-Input"] | null;
            /** @description Endogenous constructs are modeled, with or without parents, and include every latent construct. Exogenous constructs are given by direct exact Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law. */
            readonly model: components["schemas"]["ModelSpec-Input"];
        };
        /**
         * EditModelRequest
         * @description Replace one named base revision with a validated scientific definition.
         */
        readonly "EditModelRequest-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_model";
            readonly expected_revision: components["schemas"]["GitOid-Output"] | null;
            /**
             * @description Exact observations used by this edit's checks; omitted runs no predictive check.
             * @default null
             */
            readonly panel_revision: components["schemas"]["GitOid-Output"] | null;
            /** @description Endogenous constructs are modeled, with or without parents, and include every latent construct. Exogenous constructs are given by direct exact Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law. */
            readonly model: components["schemas"]["ModelSpec-Output"];
        };
        /**
         * EffectSummary
         * @description An effect summary reports posterior location, uncertainty, and sign probability.
         */
        readonly EffectSummary: {
            /** Mean */
            readonly mean: number;
            /** Median */
            readonly median: number;
            /** Lower 95 */
            readonly lower_95: number;
            /** Upper 95 */
            readonly upper_95: number;
            /** Prob Positive */
            readonly prob_positive: number;
        };
        /** EmpiricalPoint */
        readonly EmpiricalPoint: {
            /** Value */
            readonly value: number;
            /** Probability */
            readonly probability: number;
        };
        /**
         * EnergyDiagnostics
         * @description Energy distributions and the producer's chain-specific BFMI values.
         */
        readonly EnergyDiagnostics: {
            readonly energy_hist: components["schemas"]["DensityCurve"];
            readonly energy_transition_hist: components["schemas"]["DensityCurve"];
            /** Bfmi */
            readonly bfmi: readonly number[];
        };
        /** @description An entity reference identifies a construct, edge, indicator, or mechanism by its persistent identity. */
        readonly EntityRef: components["schemas"]["ConstructRef-Output"] | components["schemas"]["EdgeRef"] | components["schemas"]["IndicatorRef"] | components["schemas"]["MechanismRef"];
        /**
         * EvaluatedPredictiveChecks
         * @description An evaluated run may retain failed and partially unavailable scientific evidence.
         */
        readonly EvaluatedPredictiveChecks: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Findings */
            readonly findings: readonly components["schemas"]["PredictiveAssessment"][];
            /** @default null */
            readonly predictive_checks: components["schemas"]["PosteriorPredictiveChecks"] | null;
        };
        /** Evaluated[ConvergenceAssessmentSubject, NumericCriterionEvidence] */
        readonly Evaluated_ConvergenceAssessmentSubject_NumericCriterionEvidence_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            readonly subject: components["schemas"]["ConvergenceAssessmentSubject"];
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            readonly evidence: components["schemas"]["NumericCriterionEvidence"];
        };
        /** Evaluated[IndicatorCheckSubject, NumericCriterionEvidence] */
        readonly Evaluated_IndicatorCheckSubject_NumericCriterionEvidence_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            readonly subject: components["schemas"]["IndicatorCheckSubject"];
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            readonly evidence: components["schemas"]["NumericCriterionEvidence"];
        };
        /** Evaluated[PredictiveSubject, tuple[NumericCriterionEvidence, ...]] */
        readonly Evaluated_PredictiveSubject_tuple_NumericCriterionEvidence__________: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            readonly subject: components["schemas"]["PredictiveSubject"];
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            /** Evidence */
            readonly evidence: readonly components["schemas"]["NumericCriterionEvidence"][];
        };
        /** Evaluated[QuestionSubject, str] */
        readonly Evaluated_QuestionSubject_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            readonly subject: components["schemas"]["QuestionSubject"];
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            /** Evidence */
            readonly evidence: string;
        };
        /** Evaluated[str, ParticleMCMCEvidence] */
        readonly Evaluated_str_ParticleMCMCEvidence_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Subject */
            readonly subject: string;
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            readonly evidence: components["schemas"]["ParticleMCMCEvidence"];
        };
        /** Evaluated[str, str] */
        readonly Evaluated_str_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Subject */
            readonly subject: string;
            /**
             * Outcome
             * @enum {string}
             */
            readonly outcome: "passed" | "failed" | "warning" | "error";
            /** Evidence */
            readonly evidence: string;
        };
        readonly Evaluation_CausalEffectResult_: Domain.Available<Domain.CausalEffectResult> | components["schemas"]["Unavailable"] | components["schemas"]["NotApplicable"];
        readonly Evaluation_PosteriorPredictiveChecks_: Domain.Available<Domain.PosteriorPredictiveChecks> | components["schemas"]["Unavailable"] | components["schemas"]["NotApplicable"];
        /** @description A scalar expression composes supported arithmetic with scientific state and coefficient references. */
        readonly "Expression-Input": components["schemas"]["LiteralExpression-Input"] | components["schemas"]["StateExpression-Input"] | components["schemas"]["CoefficientExpression-Input"] | components["schemas"]["BinaryExpression-Input"] | components["schemas"]["CallExpression-Input"];
        /** @description A scalar expression composes supported arithmetic with scientific state and coefficient references. */
        readonly "Expression-Output": components["schemas"]["LiteralExpression-Output"] | components["schemas"]["StateExpression-Output"] | components["schemas"]["CoefficientExpression-Output"] | components["schemas"]["BinaryExpression-Output"] | components["schemas"]["CallExpression-Output"];
        /**
         * @description An expression function transforms scalar operands or constructs structured observation arguments.
         * @enum {string}
         */
        readonly ExpressionFunction: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
        /**
         * ExtractionPlanEvent
         * @description The extraction fan-out plan.
         */
        readonly ExtractionPlanEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "nof1-causal-lab.extraction.plan";
            /** Total Workers */
            readonly total_workers: number;
            /**
             * Max Concurrent Workers
             * @default null
             */
            readonly max_concurrent_workers: number | null;
        };
        /**
         * ExtractionSnapshotEvent
         * @description Aggregate extraction worker counts.
         */
        readonly ExtractionSnapshotEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "nof1-causal-lab.extraction.snapshot";
            /** Total Workers */
            readonly total_workers: number;
            /** Pending Workers */
            readonly pending_workers: number;
            /** Running Workers */
            readonly running_workers: number;
            /** Completed Workers */
            readonly completed_workers: number;
            /** Failed Workers */
            readonly failed_workers: number;
        };
        readonly "ExtractionSpec-Input": components["schemas"]["ComputedExtractionSpec-Input"] | components["schemas"]["SemanticExtractionSpec-Input"];
        readonly "ExtractionSpec-Output": components["schemas"]["ComputedExtractionSpec-Output"] | components["schemas"]["SemanticExtractionSpec-Output"];
        /**
         * ExtractionWorkerEvent
         * @description One extraction worker's state; a worker reports its LLM calls when it finishes.
         */
        readonly ExtractionWorkerEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "nof1-causal-lab.extraction.worker";
            /** Worker Id */
            readonly worker_id: number;
            /**
             * State
             * @enum {string}
             */
            readonly state: "pending" | "running" | "completed" | "failed";
            /** N Windows */
            readonly n_windows: number;
            /**
             * N Extractions
             * @default null
             */
            readonly n_extractions: number | null;
            /**
             * N Llm Calls
             * @default null
             */
            readonly n_llm_calls: number | null;
            /**
             * Error
             * @default null
             */
            readonly error: string | null;
        };
        readonly ExtractionWorkerResult: components["schemas"]["CompletedExtractionWorker"] | components["schemas"]["FailedExtractionChunk"];
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact revision and records its freshness.
         */
        readonly FactSource: {
            readonly ref: components["schemas"]["GitRef"];
            /** Pointer */
            readonly pointer: string;
            readonly validity: components["schemas"]["SourceValidity"];
        };
        /**
         * FailedExtractionChunk
         * @description An extraction failure with its error and no usable result-file reference.
         */
        readonly FailedExtractionChunk: {
            /** Worker Id */
            readonly worker_id: number;
            /** N Extractions */
            readonly n_extractions: number;
            /** N Windows */
            readonly n_windows: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "failed";
            /** Error */
            readonly error: string;
            /**
             * N Llm Calls
             * @default 0
             */
            readonly n_llm_calls: number | null;
            /**
             * Reused
             * @default false
             */
            readonly reused: boolean | null;
        };
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
        readonly "FilePreparationSpec-Output": {
            readonly source: components["schemas"]["FileSourceRef-Output"];
            readonly definition: components["schemas"]["DataPreparationSpec-Output"];
        };
        /**
         * FilePreparedDataMetadata
         * @description An uploaded panel's recipe owns its resolved observation schema.
         */
        readonly FilePreparedDataMetadata: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "file";
            readonly source: components["schemas"]["FileSourceRef-Output"];
            readonly preparation: components["schemas"]["DataPreparationSpec-Output"];
            /**
             * Time Origin
             * @description Calendar instant of model day zero; null denotes a calendar-free history.
             */
            readonly time_origin: string | null;
            /** Variables */
            readonly variables: readonly Domain.ObservationSpec<string>[];
        };
        /**
         * FileSourceRef
         * @description Explicit uploaded filenames, relative to this study's input directory.
         */
        readonly "FileSourceRef-Input": {
            /** Files */
            readonly files: readonly [
                string,
                ...string[]
            ];
            /**
             * Hashes
             * @description Call-time SHA-256 of every named file. The edge fills these for new calls; saved calls can be repeated from these hashes without uploaded bytes.
             */
            readonly hashes?: {
                readonly [key: string]: string;
            };
            /**
             * Start
             * @description Inclusive UTC source-coverage date.
             * @default null
             */
            readonly start?: string | null;
            /**
             * End
             * @description Exclusive UTC source-coverage date.
             * @default null
             */
            readonly end?: string | null;
        };
        /**
         * FileSourceRef
         * @description Explicit uploaded filenames, relative to this study's input directory.
         */
        readonly "FileSourceRef-Output": {
            /** Files */
            readonly files: readonly [
                string,
                ...string[]
            ];
            /**
             * Hashes
             * @description Call-time SHA-256 of every named file. The edge fills these for new calls; saved calls can be repeated from these hashes without uploaded bytes.
             */
            readonly hashes: {
                readonly [key: string]: string;
            };
            /**
             * Start
             * @description Inclusive UTC source-coverage date.
             * @default null
             */
            readonly start: string | null;
            /**
             * End
             * @description Exclusive UTC source-coverage date.
             * @default null
             */
            readonly end: string | null;
        };
        /** @enum {string} */
        readonly FitReliability: "not_fitted" | "converged" | "unconverged" | "unknown";
        /**
         * FitRequest
         * @description Condition explicitly selected model and observation revisions.
         */
        readonly "FitRequest-Input": {
            /**
             * Action
             * @default fit
             * @constant
             */
            readonly action?: "fit";
            readonly model_revision: components["schemas"]["GitOid-Input"];
            readonly panel_revision: components["schemas"]["GitOid-Input"];
            readonly settings?: components["schemas"]["FitSettingsSpec-Input"];
        };
        /**
         * FitRequest
         * @description Condition explicitly selected model and observation revisions.
         */
        readonly "FitRequest-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "fit";
            readonly model_revision: components["schemas"]["GitOid-Output"];
            readonly panel_revision: components["schemas"]["GitOid-Output"];
            readonly settings: components["schemas"]["FitSettingsSpec-Output"];
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured particle sampler.
         */
        readonly "FitSettingsSpec-Input": {
            /**
             * Num Samples
             * @default null
             */
            readonly num_samples?: number | null;
            /**
             * Num Warmup
             * @default null
             */
            readonly num_warmup?: number | null;
            /**
             * Num Chains
             * @default null
             */
            readonly num_chains?: number | null;
            /**
             * N Particles
             * @default null
             */
            readonly n_particles?: number | null;
            /**
             * Seed
             * @default null
             */
            readonly seed?: number | null;
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured particle sampler.
         */
        readonly "FitSettingsSpec-Output": {
            /**
             * Num Samples
             * @default null
             */
            readonly num_samples: number | null;
            /**
             * Num Warmup
             * @default null
             */
            readonly num_warmup: number | null;
            /**
             * Num Chains
             * @default null
             */
            readonly num_chains: number | null;
            /**
             * N Particles
             * @default null
             */
            readonly n_particles: number | null;
            /**
             * Seed
             * @default null
             */
            readonly seed: number | null;
        };
        /**
         * FitSummary
         * @description A fit read contains the inference report summary and server-composed display findings.
         *
         *     The completed action also carries the full inference report and per-draw diagnostics.
         */
        readonly FitSummary: {
            readonly report: components["schemas"]["InferenceReportCore"];
            /** Edge Estimates */
            readonly edge_estimates: Readonly<Partial<Record<components["schemas"]["EdgeId-Output"], components["schemas"]["ParameterRef"]>>>;
            /** Decay Estimates */
            readonly decay_estimates: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], components["schemas"]["ParameterRef"]>>>;
            /**
             * Prior Densities
             * @description Conditioned input laws of the fitted parameters, on their posterior marginals' quantity scale; absent where the current compiler cannot place the input model.
             */
            readonly prior_densities: Readonly<Partial<Record<components["schemas"]["ParameterId-Output"], components["schemas"]["DensityCurve"]>>>;
        };
        /**
         * FittedLawProvenance
         * @description All current laws retain one committed fit's model and observation panel.
         */
        readonly FittedLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fitted";
            readonly fitted_panel_revision: components["schemas"]["GitOid-Output"];
            readonly fitted_model_revision: components["schemas"]["GitOid-Output"];
            /**
             * Interpretation
             * @enum {string}
             */
            readonly interpretation: "in_sample_posterior_predictive" | "posterior_predictive";
        };
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
        readonly "GammaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Gamma";
            readonly concentration: components["schemas"]["Expression-Output"];
            readonly rate: components["schemas"]["Expression-Output"];
        };
        /** @description A native Git object identity for an immutable tree or commit. */
        readonly "GitOid-Input": string;
        /** @description A native Git object identity for an immutable tree or commit. */
        readonly "GitOid-Output": string;
        /**
         * GitRef
         * @description An exact file in a study's Git object database: repository, object, and path.
         */
        readonly GitRef: {
            /** Workspace Id */
            readonly workspace_id: string;
            readonly revision: components["schemas"]["GitOid-Output"];
            /** Path */
            readonly path: string;
        };
        /** HTTPValidationError */
        readonly HTTPValidationError: {
            /** Detail */
            readonly detail?: readonly components["schemas"]["ValidationError"][];
        };
        /**
         * HistogramBin
         * @description A histogram bin gives its interval, center, and number of posterior draws.
         */
        readonly HistogramBin: {
            /** Bin Center */
            readonly bin_center: number;
            /** Bin Start */
            readonly bin_start: number;
            /** Bin End */
            readonly bin_end: number;
            /** Count */
            readonly count: number;
        };
        /**
         * IdentificationReport
         * @description Positive and negative causal identification findings for the study question's outcome.
         */
        readonly IdentificationReport: {
            readonly outcome: components["schemas"]["ConstructId-Output"] | null;
            /**
             * Treatments
             * @description One tagged identification result per treatment, including its supporting evidence
             */
            readonly treatments: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], components["schemas"]["IdentifiedTreatmentStatus"] | components["schemas"]["NonIdentifiableTreatmentStatus"]>>>;
        };
        /**
         * IdentifiedTreatmentStatus
         * @description Details on how a treatment effect is identified.
         */
        readonly IdentifiedTreatmentStatus: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "identified";
            /**
             * Estimand
             * @description Nonparametric estimand returned by do-calculus
             */
            readonly estimand: string;
            /**
             * Marginalized Confounders
             * @description Unobserved confounders the estimand integrates out
             */
            readonly marginalized_confounders: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Instruments
             * @description Instrument constructs appearing in the nonparametric identification argument
             */
            readonly instruments: readonly components["schemas"]["ConstructId-Output"][];
        };
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
        readonly "IdentityTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "identity";
        };
        /**
         * IndicatorAudit
         * @description An indicator audit combines its empirical data profile with the results of validation
         *     checks.
         */
        readonly IndicatorAudit: {
            /** @default null */
            readonly profile: components["schemas"]["IndicatorEmpiricalProfile"] | null;
            /** Issues */
            readonly issues: readonly components["schemas"]["ValidationIssue"][];
            /** Checks */
            readonly checks: {
                readonly [key: string]: "ok" | "warning" | "error" | "not_evaluated";
            };
        };
        /** @enum {string} */
        readonly IndicatorCheck: "calibration" | "autocorrelation" | "variance";
        /**
         * IndicatorCheckSubject
         * @description The indicator and criterion remain present when evaluation is unavailable.
         */
        readonly IndicatorCheckSubject: {
            readonly target: components["schemas"]["IndicatorRef"];
            readonly check: components["schemas"]["IndicatorCheck"];
        };
        /**
         * IndicatorEmpiricalProfile
         * @description Retained count of usable observations for one indicator.
         */
        readonly IndicatorEmpiricalProfile: {
            /** N Obs */
            readonly n_obs: number;
        };
        /** @description A persistent indicator identity survives changes to its measurement label. */
        readonly "IndicatorId-Input": `indicator:${string}`;
        /** @description A persistent indicator identity survives changes to its measurement label. */
        readonly "IndicatorId-Output": `indicator:${string}`;
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
        readonly IndicatorRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "indicator";
            readonly id: components["schemas"]["IndicatorId-Output"];
        };
        /**
         * IndicatorSpec
         * @description Bind an observed-variable ID to a construct and an emission likelihood.
         *
         *     Extraction instructions belong to DataPreparationSpec. The shared observation
         *     schema also permits generative models before any observations have been collected.
         */
        readonly "IndicatorSpec-Input": {
            readonly observation: components["schemas"]["ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Input"];
            /** @default null */
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
        readonly "IndicatorSpec-Output": {
            readonly observation: Domain.ObservationSpec<string | null>;
            /** @default null */
            readonly likelihood: components["schemas"]["LikelihoodSpec-Output"] | null;
            /** @description Whether higher values move with (positive) or against (negative) the construct. */
            readonly construct_polarity: components["schemas"]["IndicatorPolarity"];
        };
        /**
         * InferenceEvidence
         * @description Native execution telemetry; posterior atoms and coordinates belong to the model.
         */
        readonly InferenceEvidence: {
            /** Time Origin */
            readonly time_origin: string | null;
            /** Duration Seconds */
            readonly duration_seconds: number;
            /**
             * Num Chains
             * @default null
             */
            readonly num_chains: number | null;
            /** Chain Extra Fields */
            readonly chain_extra_fields: {
                readonly [key: string]: string;
            };
            /**
             * Observation Log Probs
             * @default null
             */
            readonly observation_log_probs: string | null;
            /**
             * Observed Rows
             * @default null
             */
            readonly observed_rows: string | null;
            /**
             * Exact Observation Rows
             * @default null
             */
            readonly exact_observation_rows: string | null;
            /** @default null */
            readonly sampler_diagnostics: components["schemas"]["ParticleSamplerDiagnostics"] | null;
            /** Phase Extra Fields */
            readonly phase_extra_fields: {
                readonly [key: string]: {
                    readonly [key: string]: string;
                };
            };
            /**
             * Warmup Complete Log Posterior History
             * @default null
             */
            readonly warmup_complete_log_posterior_history: string | null;
            /**
             * All Complete Log Posterior History
             * @default null
             */
            readonly all_complete_log_posterior_history: string | null;
            /**
             * Initial Latent Delta
             * @default null
             */
            readonly initial_latent_delta: string | null;
            /**
             * Final Latent Delta
             * @default null
             */
            readonly final_latent_delta: string | null;
        };
        /**
         * InferenceMetadata
         * @description Run measurements for the production particle sampler.
         */
        readonly InferenceMetadata: {
            /** N Samples */
            readonly n_samples: number;
            /** Duration Seconds */
            readonly duration_seconds: number;
        };
        /**
         * InferenceReport
         * @description The compact core composed with retained detail, without filtering or re-parsing.
         */
        readonly InferenceReport: {
            readonly core: components["schemas"]["InferenceReportCore"];
            readonly detail: components["schemas"]["InferenceReportDetail"];
        };
        /**
         * InferenceReportCore
         * @description Compact scientific report shared by snapshots and the full report.
         */
        readonly InferenceReportCore: {
            /** Time Origin */
            readonly time_origin: string | null;
            readonly inference_metadata: components["schemas"]["InferenceMetadata"];
            readonly engine: Domain.Assessment<string, Domain.ParticleMCMCEvidence>;
            readonly inference_diagnostics: components["schemas"]["ChainDiagnostics"] | null;
            readonly sampler_diagnostics: components["schemas"]["ParticleSamplerDiagnostics"] | null;
            readonly convergence: components["schemas"]["ParameterConvergenceReport"];
            /** @default null */
            readonly loo_diagnostics: components["schemas"]["LOODiagnostics"] | null;
            /**
             * Posterior Marginals
             * @default null
             */
            readonly posterior_marginals: readonly components["schemas"]["PosteriorMarginal"][] | null;
        };
        /**
         * InferenceReportDetail
         * @description Retained plot series served in full by the action result.
         */
        readonly InferenceReportDetail: {
            /**
             * Trace Data
             * @default []
             */
            readonly trace_data: readonly components["schemas"]["TraceSeries"][];
            /**
             * Rank Histograms
             * @default []
             */
            readonly rank_histograms: readonly components["schemas"]["RankHistogram"][];
            /**
             * Pareto K
             * @default []
             */
            readonly pareto_k: readonly components["schemas"]["ParetoKPoint"][];
            /**
             * Loo Pit
             * @default []
             */
            readonly loo_pit: readonly components["schemas"]["LOOPITPoint"][];
            /**
             * Divergent
             * @default null
             */
            readonly divergent: readonly boolean[] | null;
            /**
             * Initial Latent Delta
             * @default null
             */
            readonly initial_latent_delta: readonly (readonly number[])[] | null;
            /**
             * Final Latent Delta
             * @default null
             */
            readonly final_latent_delta: readonly (readonly number[])[] | null;
        };
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
        readonly "InitialCorrelationTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "initial_state_correlation";
        };
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
        readonly "IntervalEffectTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_effect_to_ct_rate";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Input": {
            readonly target: components["schemas"]["ConstructId-Input"];
            /**
             * After
             * @description Offset from the design's start; omitted means at the start.
             * @default null
             */
            readonly after?: string | null;
            /** Value */
            readonly value: number;
        };
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Output": {
            readonly target: components["schemas"]["ConstructId-Output"];
            /**
             * After
             * @description Offset from the design's start; omitted means at the start.
             * @default null
             */
            readonly after: string | null;
            /** Value */
            readonly value: number;
        };
        /**
         * JointLawLayout
         * @description An explicit law membership ordered independently of native tensor positions.
         *
         *     Parameters sort by identity, then by scientific element identity. Trajectories
         *     follow in construct-identity order, retaining the supplied time-point order.
         *     Element identities include their categorical or covariance basis.
         */
        readonly "JointLawLayout-Input": {
            /** Parameters */
            readonly parameters: readonly (readonly [
                components["schemas"]["ParameterId-Input"],
                readonly components["schemas"]["ParameterElementId-Input"][]
            ])[];
            /** Constructs */
            readonly constructs: readonly components["schemas"]["ConstructId-Input"][];
            /** Time Points */
            readonly time_points: readonly number[];
            /** Labels */
            readonly labels: Readonly<Partial<Record<components["schemas"]["ParameterElementId-Input"], string>>>;
        };
        /**
         * JointLawLayout
         * @description An explicit law membership ordered independently of native tensor positions.
         *
         *     Parameters sort by identity, then by scientific element identity. Trajectories
         *     follow in construct-identity order, retaining the supplied time-point order.
         *     Element identities include their categorical or covariance basis.
         */
        readonly "JointLawLayout-Output": {
            /** Parameters */
            readonly parameters: readonly (readonly [
                components["schemas"]["ParameterId-Output"],
                readonly components["schemas"]["ParameterElementId-Output"][]
            ])[];
            /** Constructs */
            readonly constructs: readonly components["schemas"]["ConstructId-Output"][];
            /** Time Points */
            readonly time_points: readonly number[];
            /** Labels */
            readonly labels: Readonly<Partial<Record<components["schemas"]["ParameterElementId-Output"], string>>>;
        };
        /** @description A JSON array transports an ordered collection of recursively typed values. */
        readonly "JsonArray-Input": readonly Domain.JsonValue[];
        /** @description A JSON array transports an ordered collection of recursively typed values. */
        readonly "JsonArray-Output": readonly Domain.JsonValue[];
        /** @description A JSON object transports string-keyed recursively typed values. */
        readonly "JsonObject-Input": {
            readonly [key: string]: Domain.JsonValue;
        };
        /** @description A JSON object transports string-keyed recursively typed values. */
        readonly "JsonObject-Output": {
            readonly [key: string]: Domain.JsonValue;
        };
        /** @description A JSON scalar transports a string, number, boolean, or null. */
        readonly JsonScalar: boolean | number | string | null;
        /** @description A JSON value transports a scalar or a recursive array or object. */
        readonly "JsonValue-Input": Domain.JsonScalar | Domain.JsonArray | Domain.JsonObject;
        /** @description A JSON value transports a scalar or a recursive array or object. */
        readonly "JsonValue-Output": Domain.JsonScalar | Domain.JsonArray | Domain.JsonObject;
        /**
         * LLMTrace
         * @description An LLM trace records a conversation, its model, elapsed time, and token usage.
         */
        readonly LLMTrace: {
            /** Messages */
            readonly messages: readonly components["schemas"]["TraceMessage"][];
            /**
             * Model
             * @default
             */
            readonly model: string;
            /**
             * Total Time Seconds
             * @default 0
             */
            readonly total_time_seconds: number;
            readonly usage: components["schemas"]["TraceUsage"];
        };
        /**
         * LOODiagnostics
         * @description Exact-emission leave-one-measurement-row-out interpolation diagnostics.
         */
        readonly LOODiagnostics: {
            /** Elpd Loo */
            readonly elpd_loo: number;
            /** P Loo */
            readonly p_loo: number;
            /** Se */
            readonly se: number;
            /** N Data Points */
            readonly n_data_points: number;
            /**
             * N Bad K
             * @default null
             */
            readonly n_bad_k: number | null;
            /**
             * N Warn K
             * @default null
             */
            readonly n_warn_k: number | null;
            /**
             * Pareto Warning Limit
             * @default 0.5
             */
            readonly pareto_warning_limit: number;
            /**
             * Pareto Failure Limit
             * @default 0.7
             */
            readonly pareto_failure_limit: number;
        };
        /**
         * LOOPITPoint
         * @description A retained PIT value and its empirical and reference cumulative probabilities.
         */
        readonly LOOPITPoint: {
            /** Pit */
            readonly pit: number;
            /** Ecdf */
            readonly ecdf: number;
            /** Uniform */
            readonly uniform: number;
        };
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
        readonly "LikelihoodSpec-Output": {
            readonly law: components["schemas"]["ObservationLawSpec-Output"];
            /**
             * Standardized
             * @description Whether observations are mean-centered and scaled before fitting.
             * @default false
             */
            readonly standardized: boolean;
            /**
             * Reasoning
             * @description Why this conditional law was chosen for the indicator
             */
            readonly reasoning: string;
            /**
             * Sources
             * @default []
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
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
        readonly "LiteralExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "literal";
            /** Value */
            readonly value: number;
        };
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
             * @default null
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
        readonly "LiteratureSource-Output": {
            /**
             * Title
             * @description Title of the source (paper, meta-analysis, textbook, etc.)
             */
            readonly title: string;
            /**
             * Url
             * @description URL of the source if available
             * @default null
             */
            readonly url: string | null;
            /**
             * Snippet
             * @description Relevant excerpt or paraphrase from the source
             */
            readonly snippet: string;
        };
        /**
         * MarginalParticleGibbsSpec
         * @description Marginalized Particle Gibbs inference settings.
         */
        readonly MarginalParticleGibbsSpec: {
            /**
             * N Parameter Particles
             * @default 2
             */
            readonly n_parameter_particles: number;
            /**
             * Latent Delta
             * @default 0.2
             */
            readonly latent_delta: number;
            /**
             * Parameter Proposal
             * @default pseudo_langevin
             * @enum {string}
             */
            readonly parameter_proposal: "random_walk" | "pseudo_langevin";
            /**
             * Amala Delta Init
             * @default 0.01
             */
            readonly amala_delta_init: number;
            /**
             * Amala Delta Min
             * @default 0.00001
             */
            readonly amala_delta_min: number;
            /**
             * Amala Delta Max
             * @default 10
             */
            readonly amala_delta_max: number;
            /**
             * Amala Target Accept
             * @default 0.75
             */
            readonly amala_target_accept: number;
            /**
             * Amala Adaptation Window
             * @default 100
             */
            readonly amala_adaptation_window: number;
            /**
             * Amala Adaptation Tolerance
             * @default 0.05
             */
            readonly amala_adaptation_tolerance: number;
            /**
             * Amala Adaptation Rho
             * @default 0.5
             */
            readonly amala_adaptation_rho: number;
            /**
             * Amala Adaptation Rho Min
             * @default 0.001
             */
            readonly amala_adaptation_rho_min: number;
            /**
             * Amala Adaptation Gamma
             * @default -0.5
             */
            readonly amala_adaptation_gamma: number;
            /**
             * Amala Kappa
             * @default 0.75
             */
            readonly amala_kappa: number;
            /** Amala Grad Clip */
            readonly amala_grad_clip: number | "infinity";
            /** @default amala_exact */
            readonly dsmc_leaf_proposal: components["schemas"]["DSMCLeafProposal"];
            /**
             * Latent Block Coords
             * @default null
             */
            readonly latent_block_coords: number | null;
            /**
             * Paid Mix Z Weight
             * @default 0.85
             */
            readonly paid_mix_z_weight: number;
            /**
             * Paid Mix Pilot Weight
             * @default 0.1
             */
            readonly paid_mix_pilot_weight: number;
            /**
             * Paid Mix Pilot Var Scale
             * @default 0.25
             */
            readonly paid_mix_pilot_var_scale: number;
            /**
             * Paid Mix Wide Mult
             * @default 4
             */
            readonly paid_mix_wide_mult: number;
            /**
             * Diagnostic Metrics All
             * @default false
             */
            readonly diagnostic_metrics_all: boolean;
            /**
             * Diagnostic Metrics
             * @default []
             */
            readonly diagnostic_metrics: readonly string[];
            /**
             * Param Step Size
             * @default 0.02
             */
            readonly param_step_size: number;
            /**
             * Param Step Size Min
             * @default 0.000001
             */
            readonly param_step_size_min: number;
            /**
             * Param Step Size Max
             * @default 1000
             */
            readonly param_step_size_max: number;
            /**
             * Param Target Accept
             * @default 0.35
             */
            readonly param_target_accept: number;
            /**
             * Adaptation Rate
             * @default 0.05
             */
            readonly adaptation_rate: number;
            /**
             * Adaptation Scheme
             * @default simple
             * @enum {string}
             */
            readonly adaptation_scheme: "simple" | "dual_averaging";
            /**
             * Init Method
             * @default pathfinder
             * @enum {string}
             */
            readonly init_method: "random" | "pathfinder";
            /**
             * Pathfinder Num Elbo Samples
             * @default 20
             */
            readonly pathfinder_num_elbo_samples: number;
            /**
             * Pathfinder Maxiter
             * @default 20
             */
            readonly pathfinder_maxiter: number;
            /**
             * N Pathfinder Starts
             * @default 8
             */
            readonly n_pathfinder_starts: number;
            /**
             * Pathfinder Parallel Workers
             * @default null
             */
            readonly pathfinder_parallel_workers: number | null;
            /**
             * Pathfinder Init Scale
             * @default 0.1
             */
            readonly pathfinder_init_scale: number | null;
            /**
             * Auto Preconditioner Method
             * @default pathfinder
             * @enum {string}
             */
            readonly auto_preconditioner_method: "map" | "none" | "pathfinder";
            /**
             * Auto Preconditioner Maxiter
             * @default 200
             */
            readonly auto_preconditioner_maxiter: number;
            /**
             * Init Scale
             * @default 0.05
             */
            readonly init_scale: number;
            /**
             * Compute Latent Posterior Summary
             * @default true
             */
            readonly compute_latent_posterior_summary: boolean;
            /**
             * N Ieks Iters
             * @default 6
             */
            readonly n_ieks_iters: number;
        };
        /**
         * @description A measurement dtype defines the observed value domain of an indicator.
         * @enum {string}
         */
        readonly MeasurementDtype: "continuous" | "binary" | "count" | "ordinal" | "categorical";
        /**
         * MeasurementsData
         * @description Counts and representative observations read directly from one panel revision.
         */
        readonly MeasurementsData: {
            /** N Observations */
            readonly n_observations: number;
            /** Per Indicator Counts */
            readonly per_indicator_counts: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], number>>>;
            /** Combined Extractions Sample */
            readonly combined_extractions_sample: readonly components["schemas"]["ObservationRecord"][];
        };
        /** @description A persistent mechanism identity distinguishes additive terms through reordering and revision. */
        readonly "MechanismId-Input": `mechanism:${string}`;
        /** @description A persistent mechanism identity distinguishes additive terms through reordering and revision. */
        readonly "MechanismId-Output": `mechanism:${string}`;
        /**
         * MechanismRef
         * @description A particular additive term, independently of its position or coefficient values.
         */
        readonly MechanismRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "mechanism";
            readonly id: components["schemas"]["MechanismId-Output"];
        };
        /**
         * MixedLawProvenance
         * @description Some laws retain a committed fit and others have different ancestry.
         */
        readonly MixedLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "mixed";
            readonly fitted_panel_revision: components["schemas"]["GitOid-Output"];
            readonly fitted_model_revision: components["schemas"]["GitOid-Output"];
            /**
             * Interpretation
             * @constant
             */
            readonly interpretation: "mixed";
        };
        /**
         * ModelCheckReport
         * @description Current-code findings selected by their consumed scientific inputs.
         */
        readonly ModelCheckReport: {
            /** Specification */
            readonly specification: readonly components["schemas"]["SpecificationAssessment"][];
            /** @default null */
            readonly question: components["schemas"]["QuestionCheckReport"] | null;
            /** @default null */
            readonly predictive: components["schemas"]["ModelPredictiveReport"] | null;
            /**
             * Reused
             * @default []
             */
            readonly reused: readonly (components["schemas"]["CheckGroup"] | "predictive")[];
        };
        /**
         * ModelDiffReport
         * @description A model diff joins typed entity comparisons and evidence at two model revisions or checkpoints.
         */
        readonly ModelDiffReport: {
            readonly before: components["schemas"]["GitRef"] | null;
            readonly after: components["schemas"]["GitRef"] | null;
            readonly before_model: components["schemas"]["ModelSpec-Output"] | null;
            readonly after_model: components["schemas"]["ModelSpec-Output"] | null;
            /** Parameters */
            readonly parameters: readonly Domain.Change<Domain.ParameterSpec>[];
            /** Constructs */
            readonly constructs: readonly (Domain.Change<Domain.ConstructRef> | Domain.Unchanged<Domain.ConstructRef>)[];
            /** Edges */
            readonly edges: readonly (Domain.Change<Domain.EdgeRef> | Domain.Unchanged<Domain.EdgeRef>)[];
            /** Before Dispositions */
            readonly before_dispositions: readonly components["schemas"]["StructuralItemDisposition"][];
            /** After Dispositions */
            readonly after_dispositions: readonly components["schemas"]["StructuralItemDisposition"][];
            /** Before Dynamic Construct Ids */
            readonly before_dynamic_construct_ids: readonly components["schemas"]["ConstructId-Output"][];
            /** After Dynamic Construct Ids */
            readonly after_dynamic_construct_ids: readonly components["schemas"]["ConstructId-Output"][];
            /** Changed Inputs */
            readonly changed_inputs: readonly string[];
            /** Before Checks */
            readonly before_checks: readonly components["schemas"]["SpecificationAssessment"][];
            /** After Checks */
            readonly after_checks: readonly components["schemas"]["SpecificationAssessment"][];
            readonly before_fit: components["schemas"]["InferenceReportCore"] | null;
            readonly after_fit: components["schemas"]["InferenceReportCore"] | null;
            readonly before_simulation: components["schemas"]["SimulationReport"] | null;
            readonly after_simulation: components["schemas"]["SimulationReport"] | null;
        };
        /**
         * ModelDiffRequest
         * @description An immutable comparison read; it creates no attempt or journal record.
         */
        readonly "ModelDiffRequest-Input": {
            readonly before: components["schemas"]["GitOid-Input"];
            readonly after: components["schemas"]["GitOid-Input"];
        };
        /**
         * ModelDiffRequest
         * @description An immutable comparison read; it creates no attempt or journal record.
         */
        readonly "ModelDiffRequest-Output": {
            readonly before: components["schemas"]["GitOid-Output"];
            readonly after: components["schemas"]["GitOid-Output"];
        };
        /**
         * ModelFitResult
         * @description Fit inputs and native telemetry; current reports are derived from its atoms.
         */
        readonly ModelFitResult: {
            readonly model: components["schemas"]["GitRef"];
            readonly panel: components["schemas"]["GitRef"];
            readonly evidence: components["schemas"]["InferenceEvidence"];
        };
        /**
         * ModelGraphView
         * @description Scientific entity identities selected for the graph at this authoring checkpoint.
         */
        readonly ModelGraphView: {
            /**
             * Construct Ids
             * @default []
             */
            readonly construct_ids: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Edge Ids
             * @default []
             */
            readonly edge_ids: readonly components["schemas"]["EdgeId-Output"][];
            /**
             * Dynamic Construct Ids
             * @default []
             */
            readonly dynamic_construct_ids: readonly components["schemas"]["ConstructId-Output"][];
            /** Status */
            readonly status: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], "observed" | "marginalized" | "blocking">>>;
        };
        readonly ModelPredictiveEvaluation: components["schemas"]["EvaluatedPredictiveChecks"] | components["schemas"]["UnavailablePredictiveChecks"];
        /**
         * ModelPredictiveReport
         * @description One automatic, reproducible battery over the full model's current laws.
         */
        readonly ModelPredictiveReport: {
            readonly model_revision: components["schemas"]["GitOid-Output"];
            readonly panel_revision: components["schemas"]["GitOid-Output"] | null;
            /** Draws */
            readonly draws: number;
            /** Seed */
            readonly seed: number;
            readonly law: components["schemas"]["PredictiveLawProvenance"];
            readonly evaluation: components["schemas"]["ModelPredictiveEvaluation"];
            /**
             * Status
             * @enum {string}
             */
            readonly status: "passed" | "failed" | "not_evaluated";
        };
        /**
         * ModelSimulationResult
         * @description The exact generated histories retained by one simulation.
         */
        readonly ModelSimulationResult: {
            readonly evidence: components["schemas"]["SimulationEvidence"];
        };
        /**
         * ModelSnapshot
         * @description The canonical scientific definition with independently sourced inputs and findings.
         */
        readonly ModelSnapshot: {
            /** @default null */
            readonly question: Domain.Sourced<Domain.QuestionSpec> | null;
            /** @default null */
            readonly model: Domain.Sourced<Domain.ModelSpec> | null;
            /** Workspace Id */
            readonly workspace_id: string;
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Selected Seq */
            readonly selected_seq: number;
            /**
             * Can Simulate
             * @default false
             */
            readonly can_simulate: boolean;
            readonly state: components["schemas"]["StudyState"];
            /** @default null */
            readonly raw_data: Domain.Sourced<Domain.RawDataData> | null;
            /** @default null */
            readonly measurements: Domain.Sourced<Domain.MeasurementsData> | null;
            /** @default null */
            readonly metadata: Domain.Sourced<Domain.PreparedDataMetadata> | null;
            /** @default null */
            readonly profile: Domain.Sourced<Domain.DataProfileArtifact> | null;
            /** @default null */
            readonly identification: Domain.Sourced<Domain.IdentificationReport> | null;
            /** @default null */
            readonly dispositions: Domain.Sourced<readonly (Domain.StructuralItemDisposition)[]> | null;
            readonly graph: components["schemas"]["ModelGraphView"];
            /** Entity Failures */
            readonly entity_failures: {
                readonly [key: string]: readonly string[];
            };
            /** @default null */
            readonly validation_report: Domain.Sourced<Domain.ValidationReportArtifact> | null;
            /** Confounder Equations */
            readonly confounder_equations: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], string>>>;
            /** State Equations */
            readonly state_equations: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], string>>>;
            /** Observation Equations */
            readonly observation_equations: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], string>>>;
            /** Likelihood Diagnostics */
            readonly likelihood_diagnostics: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], readonly components["schemas"]["HistogramBin"][]>>>;
            /** Authoring Prior Densities */
            readonly authoring_prior_densities: Readonly<Partial<Record<components["schemas"]["ParameterId-Output"], components["schemas"]["DensityCurve"]>>>;
            /** @default null */
            readonly fit: Domain.Sourced<Domain.FitSummary> | null;
            /** @default null */
            readonly specification: Domain.Sourced<readonly (Domain.SpecificationAssessment)[]> | null;
            /** @default null */
            readonly question_checks: Domain.Sourced<Domain.QuestionCheckReport> | null;
            /** @default null */
            readonly simulation: Domain.Sourced<Domain.SimulationReport> | null;
            /** @default null */
            readonly predictive: Domain.Sourced<Domain.ModelPredictiveReport> | null;
        };
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
            readonly distributions?: Readonly<Partial<Record<components["schemas"]["DistributionId-Input"], components["schemas"]["NumPyroDistribution-Input"]>>>;
            /**
             * Law Layouts
             * @description Scientific coordinates and production labels of each joint law, beside its native atoms.
             */
            readonly law_layouts?: Readonly<Partial<Record<components["schemas"]["DistributionId-Input"], components["schemas"]["JointLawLayout-Input"]>>>;
            /**
             * Measurement Clock
             * @default null
             */
            readonly measurement_clock?: string | null;
        };
        /**
         * ModelSpec
         * @description A connected causal graph with owned scientific detail, built to answer the study question.
         */
        readonly "ModelSpec-Output": {
            /**
             * Edges
             * @default []
             */
            readonly edges: readonly components["schemas"]["CausalEdgeSpec-Output"][];
            /**
             * Parameters
             * @default []
             */
            readonly parameters: readonly components["schemas"]["ParameterSpec-Output"][];
            /**
             * Distributions
             * @description All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements.
             */
            readonly distributions: Readonly<Partial<Record<components["schemas"]["DistributionId-Output"], components["schemas"]["NumPyroDistribution-Output"]>>>;
            /**
             * Law Layouts
             * @description Scientific coordinates and production labels of each joint law, beside its native atoms.
             */
            readonly law_layouts: Readonly<Partial<Record<components["schemas"]["DistributionId-Output"], components["schemas"]["JointLawLayout-Output"]>>>;
            /**
             * Measurement Clock
             * @default null
             */
            readonly measurement_clock: string | null;
            /**
             * Time Points
             * @description Joint trajectory coordinates own the model's retained time grid.
             */
            readonly time_points: readonly number[];
        };
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
        readonly "NegativeBinomial2LawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "NegativeBinomial2";
            readonly mean: components["schemas"]["Expression-Output"];
            readonly concentration: components["schemas"]["Expression-Output"];
        };
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        readonly NonIdentifiableTreatmentStatus: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "not_identified";
            /**
             * Confounders
             * @description Unobserved constructs blocking identification
             */
            readonly confounders: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Notes
             * @description Optional explanation if confounders cannot be enumerated
             * @default null
             */
            readonly notes: string | null;
        };
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
        readonly "NormalLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Normal";
            readonly loc: components["schemas"]["Expression-Output"];
            readonly scale: components["schemas"]["Expression-Output"];
        };
        /**
         * NotApplicable
         * @description The selected operation does not call for this result.
         */
        readonly NotApplicable: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_applicable";
            /** Reason */
            readonly reason: string;
        };
        readonly NotEvaluatedReason: components["schemas"]["PredictiveCheckReason"] | ("NONFINITE_EMISSION_MEAN" | "INSUFFICIENT_TIMES" | "NO_RELAXATION_TERM" | "EDGE_CONTRASTS_EXPLICIT" | "NO_OBSERVATION_SUPPORT" | "NO_OBSERVATIONS" | "STATIC_CONSTRUCT" | "INSUFFICIENT_OBSERVATIONS" | "ZERO_RESIDUAL_VARIANCE" | "ZERO_OBSERVED_VARIANCE" | "NONFINITE_PATHS" | "NONFINITE_SIGNAL" | "COMPARISON_INPUTS_MISSING" | "INSUFFICIENT_CHAIN_SAMPLES" | "NO_RETAINED_CHAINS" | "ARCHIVED_ENGINE_NOT_RETAINED" | "NO_OUTCOME" | "CONSTRUCT_UNDEFINED" | "NO_PANEL" | "STATE_NOT_RECORDED");
        /** NotEvaluated[ConvergenceAssessmentSubject] */
        readonly NotEvaluated_ConvergenceAssessmentSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            readonly subject: components["schemas"]["ConvergenceAssessmentSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /**
             * Detail
             * @default
             */
            readonly detail: string;
        };
        /** NotEvaluated[IndicatorCheckSubject] */
        readonly NotEvaluated_IndicatorCheckSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            readonly subject: components["schemas"]["IndicatorCheckSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /**
             * Detail
             * @default
             */
            readonly detail: string;
        };
        /** NotEvaluated[PredictiveSubject] */
        readonly NotEvaluated_PredictiveSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            readonly subject: components["schemas"]["PredictiveSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /**
             * Detail
             * @default
             */
            readonly detail: string;
        };
        /** NotEvaluated[QuestionSubject] */
        readonly NotEvaluated_QuestionSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            readonly subject: components["schemas"]["QuestionSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /**
             * Detail
             * @default
             */
            readonly detail: string;
        };
        /** NotEvaluated[str] */
        readonly NotEvaluated_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Subject */
            readonly subject: string;
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /**
             * Detail
             * @default
             */
            readonly detail: string;
        };
        /** @description A native NumPyro probability distribution serialized by its constructor tree. */
        readonly "NumPyroDistribution-Input": {
            /** Distribution */
            readonly distribution: string;
            /** Params */
            readonly params: {
                readonly [key: string]: Domain.JsonValue;
            };
        };
        /** @description A native NumPyro probability distribution serialized by its constructor tree. */
        readonly "NumPyroDistribution-Output": {
            /** Distribution */
            readonly distribution: string;
            /** Params */
            readonly params: {
                readonly [key: string]: Domain.JsonValue;
            };
        };
        /**
         * NumericCriterionEvidence
         * @description A measured scalar and the producer's numerical acceptance region.
         */
        readonly NumericCriterionEvidence: {
            /** Criterion */
            readonly criterion: string;
            /** Value */
            readonly value: number;
            /**
             * Lower
             * @default null
             */
            readonly lower: number | null;
            /**
             * Upper
             * @default null
             */
            readonly upper: number | null;
            /**
             * Lower Inclusive
             * @default true
             */
            readonly lower_inclusive: boolean;
            /**
             * Upper Inclusive
             * @default true
             */
            readonly upper_inclusive: boolean;
            /**
             * Note
             * @default
             */
            readonly note: string;
            /**
             * Display Value
             * @default
             */
            readonly display_value: string;
            /**
             * Band Label
             * @default
             */
            readonly band_label: string;
        };
        /**
         * ObservationHistory
         * @description All prepared observations, their true anchors and their measurement support.
         */
        readonly ObservationHistory: {
            /** Label */
            readonly label: string;
            /** Times */
            readonly times: readonly number[];
            /** Values */
            readonly values: readonly (number | null)[];
            /** Support Start */
            readonly support_start: readonly (number | null)[];
            /** Support End */
            readonly support_end: readonly (number | null)[];
            /** Time Origin */
            readonly time_origin: string | null;
            /** Levels */
            readonly levels: readonly string[] | null;
            /** Empirical */
            readonly empirical: readonly components["schemas"]["EmpiricalPoint"][];
        };
        readonly "ObservationLawSpec-Input": components["schemas"]["DeltaLawSpec_Expression_-Input"] | components["schemas"]["NormalLawSpec_Expression_-Input"] | components["schemas"]["StudentTLawSpec_Expression_-Input"] | components["schemas"]["PoissonLawSpec_Expression_-Input"] | components["schemas"]["GammaLawSpec_Expression_-Input"] | components["schemas"]["BernoulliLogitsLawSpec_Expression_-Input"] | components["schemas"]["BernoulliProbsLawSpec_Expression_-Input"] | components["schemas"]["NegativeBinomial2LawSpec_Expression_-Input"] | components["schemas"]["BetaLawSpec_Expression_-Input"] | components["schemas"]["OrderedLogisticLawSpec_Expression_-Input"] | components["schemas"]["CategoricalLawSpec_Expression_-Input"];
        readonly "ObservationLawSpec-Output": Domain.DeltaLawSpec<Domain.Expression> | Domain.NormalLawSpec<Domain.Expression> | Domain.StudentTLawSpec<Domain.Expression> | Domain.PoissonLawSpec<Domain.Expression> | Domain.GammaLawSpec<Domain.Expression> | Domain.BernoulliLogitsLawSpec<Domain.Expression> | Domain.BernoulliProbsLawSpec<Domain.Expression> | Domain.NegativeBinomial2LawSpec<Domain.Expression> | Domain.BetaLawSpec<Domain.Expression> | Domain.OrderedLogisticLawSpec<Domain.Expression> | Domain.CategoricalLawSpec<Domain.Expression>;
        /**
         * ObservationRecord
         * @description Canonical serialized extraction observation row.
         */
        readonly ObservationRecord: {
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /** Value */
            readonly value: string | number | boolean | null;
            /** Anchor Time */
            readonly anchor_time: string | null;
            /** Support Kind */
            readonly support_kind: string | null;
            /** Summary Operator */
            readonly summary_operator: string | null;
            /** Anchor Policy */
            readonly anchor_policy: string | null;
            /** Observation Window */
            readonly observation_window: string | null;
            /** Support Start */
            readonly support_start: string | null;
            /** Support End */
            readonly support_end: string | null;
        };
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
             * @default null
             */
            readonly observation_window?: string | null;
            /**
             * Ordinal Levels
             * @description Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
             * @default null
             */
            readonly ordinal_levels?: readonly string[] | null;
            /**
             * Categorical Levels
             * @description Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
             * @default null
             */
            readonly categorical_levels?: readonly string[] | null;
        };
        /** ObservationSpec[Annotated[Union[Duration, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]] */
        readonly "ObservationSpec_Annotated_Union_Duration__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Output": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["IndicatorId-Output"];
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
             * @default null
             */
            readonly observation_window: string | null;
            /**
             * Ordinal Levels
             * @description Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
             * @default null
             */
            readonly ordinal_levels: readonly string[] | null;
            /**
             * Categorical Levels
             * @description Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
             * @default null
             */
            readonly categorical_levels: readonly string[] | null;
        };
        /** ObservationSpec[Duration] */
        readonly ObservationSpec_Duration_: {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["IndicatorId-Output"];
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
            readonly observation_window: string;
            /**
             * Ordinal Levels
             * @description Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
             * @default null
             */
            readonly ordinal_levels: readonly string[] | null;
            /**
             * Categorical Levels
             * @description Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
             * @default null
             */
            readonly categorical_levels: readonly string[] | null;
        };
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
        readonly "OrderedLogisticLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "OrderedLogistic";
            readonly predictor: components["schemas"]["Expression-Output"];
            readonly cutpoints: components["schemas"]["Expression-Output"];
        };
        /**
         * OutcomeSubject
         * @description Whether the model defines the question's outcome as a measured, modeled course.
         */
        readonly OutcomeSubject: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly check: "outcome";
            readonly outcome: components["schemas"]["ConstructRef-Output"];
        };
        /**
         * PPCOverlay
         * @description A predictive overlay sets one indicator's observed values against simulated ones.
         *
         *     It carries the predictive median and a few individual replicated series, the
         *     spaghetti plot of a visual predictive check.
         */
        readonly PPCOverlay: {
            /** Times */
            readonly times: readonly number[];
            /** Time Origin */
            readonly time_origin: string | null;
            /** Standardized */
            readonly standardized: boolean;
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /** Observed */
            readonly observed: readonly (number | null)[];
            /** Median */
            readonly median: readonly (number | null)[];
            /** Spaghetti Draws */
            readonly spaghetti_draws: readonly (readonly (number | null)[])[];
        };
        /**
         * PPCTestStat
         * @description A predictive test statistic compares an observed summary with its distribution under
         *     replicated data.
         *
         *     Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
         *     with a vertical line at T(y_observed).
         */
        readonly PPCTestStat: {
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /**
             * Stat Name
             * @enum {string}
             */
            readonly stat_name: "mean" | "sd" | "min" | "max";
            /** Observed Value */
            readonly observed_value: number;
            /** Rep Values */
            readonly rep_values: readonly number[];
            /** P Value */
            readonly p_value: number | null;
            /** Histogram */
            readonly histogram: readonly components["schemas"]["HistogramBin"][];
        };
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
        readonly "PanelRef-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "panel";
            readonly revision: components["schemas"]["GitOid-Output"];
        };
        /**
         * ParameterConvergenceReport
         * @description Recorded-chain criteria cover parameters, not latent-path mixing.
         */
        readonly ParameterConvergenceReport: {
            /** Assessments */
            readonly assessments: readonly Domain.Assessment<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence>[];
            /** Checked */
            readonly checked: number;
            /**
             * Status
             * @enum {string}
             */
            readonly status: "passed" | "failed" | "not_evaluated";
            /** Messages */
            readonly messages: readonly string[];
        };
        /**
         * ParameterDiagnostics
         * @description Measurements on one scientifically identified retained scalar chain.
         */
        readonly ParameterDiagnostics: {
            /** Parameter */
            readonly parameter: string;
            readonly subject: components["schemas"]["ParameterRef"];
            /** R Hat */
            readonly r_hat: number | null;
            /** Ess Bulk */
            readonly ess_bulk: number | null;
            /** Ess Tail */
            readonly ess_tail: number | null;
            /** Mcse Mean */
            readonly mcse_mean: number | null;
        };
        /** ParameterDrawColumn */
        readonly ParameterDrawColumn: {
            /** Label */
            readonly label: string;
            readonly subject: components["schemas"]["ParameterRef"];
            /** Values */
            readonly values: readonly number[];
            /** Empirical */
            readonly empirical: readonly components["schemas"]["EmpiricalPoint"][];
        };
        /** @description Every retained parameter coordinate is available without thinning or pair selection, or has an explicit unavailable reason. */
        readonly ParameterDraws: Domain.Available<readonly (Domain.ParameterDrawColumn)[]> | components["schemas"]["Unavailable"];
        /** @description A parameter element identity identifies a logical scalar component across model revisions. */
        readonly "ParameterElementId-Input": `element:${string}`;
        /** @description A parameter element identity identifies a logical scalar component across model revisions. */
        readonly "ParameterElementId-Output": `element:${string}`;
        /** @description A scientific parameter identity connects component coefficients to one parameter definition. */
        readonly "ParameterId-Input": `parameter:${string}`;
        /** @description A scientific parameter identity connects component coefficients to one parameter definition. */
        readonly "ParameterId-Output": `parameter:${string}`;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        readonly ParameterRef: {
            readonly parameter_id: components["schemas"]["ParameterId-Output"];
            readonly element_id: components["schemas"]["ParameterElementId-Output"];
        };
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
            /**
             * @description Membership in a native law in ModelSpec.distributions; may be joint. None means the law has not been assigned yet.
             * @default null
             */
            readonly distribution?: components["schemas"]["DistributionId-Input"] | null;
            /**
             * Reasoning
             * @description Why the authored prior law fits this quantity, and where its values come from.
             * @default null
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
        readonly "ParameterSpec-Output": {
            readonly id: components["schemas"]["ParameterId-Output"];
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
            readonly transform: components["schemas"]["ParameterTransformSpec-Output"];
            /**
             * @description Membership in a native law in ModelSpec.distributions; may be joint. None means the law has not been assigned yet.
             * @default null
             */
            readonly distribution: components["schemas"]["DistributionId-Output"] | null;
            /**
             * Reasoning
             * @description Why the authored prior law fits this quantity, and where its values come from.
             * @default null
             */
            readonly reasoning: string | null;
            /**
             * Sources
             * @description Evidence behind the authored prior law.
             * @default []
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
        readonly "ParameterTransformSpec-Input": components["schemas"]["IdentityTransformSpec-Input"] | components["schemas"]["PersistenceTransformSpec-Input"] | components["schemas"]["IntervalEffectTransformSpec-Input"] | components["schemas"]["InitialCorrelationTransformSpec-Input"];
        readonly "ParameterTransformSpec-Output": components["schemas"]["IdentityTransformSpec-Output"] | components["schemas"]["PersistenceTransformSpec-Output"] | components["schemas"]["IntervalEffectTransformSpec-Output"] | components["schemas"]["InitialCorrelationTransformSpec-Output"];
        /**
         * ParameterWarmupDiagnostics
         * @description Realized initialization and preconditioning, with the complete Pathfinder evidence once.
         */
        readonly ParameterWarmupDiagnostics: {
            /** @default null */
            readonly pathfinder: components["schemas"]["PathfinderDiagnostics"] | null;
            /** Pathfinder Run Count */
            readonly pathfinder_run_count: number;
            /** Pathfinder Consumers */
            readonly pathfinder_consumers: readonly string[];
            /** Init Source */
            readonly init_source: string;
            /** Preconditioner Source */
            readonly preconditioner_source: string;
            /**
             * Preconditioner Device
             * @default null
             */
            readonly preconditioner_device: string | null;
            /** Dim */
            readonly dim: number;
            /** Duration Seconds */
            readonly duration_seconds: number;
            /**
             * Pathfinder Sampling Mode
             * @default null
             */
            readonly pathfinder_sampling_mode: string | null;
            /**
             * Pathfinder Init Scale
             * @default null
             */
            readonly pathfinder_init_scale: number | null;
            /**
             * Prior Released Site Names
             * @default []
             */
            readonly prior_released_site_names: readonly string[];
            /**
             * Prior Released Site Indices
             * @default []
             */
            readonly prior_released_site_indices: readonly number[];
            /**
             * Prior Release Scale
             * @default 0
             */
            readonly prior_release_scale: number;
        };
        /**
         * ParetoKPoint
         * @description One PSIS influence measurement with its original row and scientific class.
         */
        readonly ParetoKPoint: {
            /** Rank */
            readonly rank: number;
            /** Timestep */
            readonly timestep: number;
            /** K */
            readonly k: number | ("infinity" | "-infinity" | "undefined");
            /**
             * Status
             * @enum {string}
             */
            readonly status: "passed" | "warning" | "failed" | "not_evaluated";
        };
        /**
         * ParticleMCMCEvidence
         * @description The production particle-MCMC target and its exact latent transition.
         */
        readonly ParticleMCMCEvidence: Record<string, never>;
        /**
         * ParticleSamplerDiagnostics
         * @description Typed exact-sampler settings and transition telemetry from the native producer.
         */
        readonly ParticleSamplerDiagnostics: {
            readonly settings: components["schemas"]["SamplerSpec"];
            /** Parameter Kernel */
            readonly parameter_kernel: string;
            /** Mcmc Phase Seconds */
            readonly mcmc_phase_seconds: number;
            /** Dsmc Leaf Proposal */
            readonly dsmc_leaf_proposal: string;
            /** Diagnostic Metrics */
            readonly diagnostic_metrics: readonly string[];
            /** Param Target Accept */
            readonly param_target_accept: number;
            /** Parameter Preconditioned */
            readonly parameter_preconditioned: boolean;
            /** Diagnostic Summary Phase */
            readonly diagnostic_summary_phase: string;
            /** Parameter Accept Rate */
            readonly parameter_accept_rate: number;
            /** Latent Update Fraction */
            readonly latent_update_fraction: number;
            /** Latent Frozen Fraction */
            readonly latent_frozen_fraction: number;
            /** Latent Block Coords */
            readonly latent_block_coords: number | null;
            /** Initial Param Step Size */
            readonly initial_param_step_size: readonly number[];
            /** Final Param Step Size */
            readonly final_param_step_size: readonly number[];
            /**
             * Latent Sign Flip Moves
             * @default null
             */
            readonly latent_sign_flip_moves: boolean | null;
            /** Chain Post Warmup Complete Log Posterior Mean */
            readonly chain_post_warmup_complete_log_posterior_mean: readonly number[];
            /**
             * Latent Move Rms Mean
             * @default null
             */
            readonly latent_move_rms_mean: number | null;
            /**
             * Parameter Jump Rms Mean
             * @default null
             */
            readonly parameter_jump_rms_mean: number | null;
            /**
             * Reference Path Hit Rate Mean
             * @default null
             */
            readonly reference_path_hit_rate_mean: number | null;
            /**
             * Selected Particle Unique Count Mean
             * @default null
             */
            readonly selected_particle_unique_count_mean: number | null;
            /**
             * Amala Grad Norm Mean
             * @default null
             */
            readonly amala_grad_norm_mean: number | null;
            /**
             * Amala Grad Norm Max
             * @default null
             */
            readonly amala_grad_norm_max: number | null;
            readonly parameter_warmup: components["schemas"]["ParameterWarmupDiagnostics"];
        };
        /** PathSeries */
        readonly PathSeries: {
            /** Label */
            readonly label: string;
            /** Action */
            readonly action: readonly components["schemas"]["RecordedPath"][];
            /**
             * Reference
             * @default []
             */
            readonly reference: readonly components["schemas"]["RecordedPath"][];
            /**
             * Levels
             * @default null
             */
            readonly levels: readonly string[] | null;
        };
        /**
         * PathfinderDiagnostics
         * @description Retained native initialization measurements; never posterior evidence.
         */
        readonly PathfinderDiagnostics: {
            /** N Pathfinder Starts */
            readonly n_pathfinder_starts: number;
            /** N Pathfinder Starts Finite */
            readonly n_pathfinder_starts_finite: number;
            /** Pathfinder Parallel Workers */
            readonly pathfinder_parallel_workers: number;
            /** Pathfinder Setup Seconds */
            readonly pathfinder_setup_seconds: number;
            /** Pathfinder Jax Compile Seconds */
            readonly pathfinder_jax_compile_seconds: number;
            /** Pathfinder Jax Compile Batch Sizes */
            readonly pathfinder_jax_compile_batch_sizes: readonly number[];
            /** Pathfinder Runtime Seconds */
            readonly pathfinder_runtime_seconds: number;
            /** Pathfinder Total Seconds */
            readonly pathfinder_total_seconds: number;
            /** Best Pathfinder Elbo */
            readonly best_pathfinder_elbo: number;
            /** Pathfinder Elbo Min */
            readonly pathfinder_elbo_min: number;
            /** Pathfinder Elbo Max */
            readonly pathfinder_elbo_max: number;
            /** Pathfinder Elbo Spread */
            readonly pathfinder_elbo_spread: number;
            /** Pathfinder Elbos */
            readonly pathfinder_elbos: readonly number[];
            /** Pathfinder Maxiter */
            readonly pathfinder_maxiter: number;
            /** Pathfinder Lbfgs Memory */
            readonly pathfinder_lbfgs_memory: number;
            /** Pathfinder Elbo Samples */
            readonly pathfinder_elbo_samples: number;
            /** Pathfinder Elbo Screen Samples */
            readonly pathfinder_elbo_screen_samples: number;
            /** Pathfinder Elbo Refine Candidates */
            readonly pathfinder_elbo_refine_candidates: number;
            /** Pathfinder Elbo Candidate Batch Size */
            readonly pathfinder_elbo_candidate_batch_size: number;
            /** Pathfinder Per Start */
            readonly pathfinder_per_start: readonly components["schemas"]["PathfinderStartDiagnostics"][];
        };
        /**
         * PathfinderStartDiagnostics
         * @description Retained native initialization measurements; never posterior evidence.
         */
        readonly PathfinderStartDiagnostics: {
            /** N Elbo Batch Evaluations */
            readonly n_elbo_batch_evaluations: number;
            /** N Elbo Screen Candidates */
            readonly n_elbo_screen_candidates: number;
            /** N Elbo Refine Candidates */
            readonly n_elbo_refine_candidates: number;
            /** Best Elbo Candidate Index */
            readonly best_elbo_candidate_index: number;
            /** Start Idx */
            readonly start_idx: number;
            /** N Trajectory Points */
            readonly n_trajectory_points: number;
            /** N Valid Iterates */
            readonly n_valid_iterates: number;
            /** N Elbo Candidates */
            readonly n_elbo_candidates: number;
            /** N Lbfgs Iterations */
            readonly n_lbfgs_iterations: number;
            /** Final Log Posterior */
            readonly final_log_posterior: number;
            /** Best Elbo This Start */
            readonly best_elbo_this_start: number | null;
            /** Scipy Success */
            readonly scipy_success: boolean;
            /** Scipy Status */
            readonly scipy_status: number;
        };
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
        readonly "PersistenceTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_persistence_to_ct_decay";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
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
        readonly "PoissonLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Poisson";
            readonly rate: components["schemas"]["Expression-Output"];
        };
        /**
         * PosteriorMarginal
         * @description One parameter's posterior interval, scale and density plot.
         */
        readonly PosteriorMarginal: {
            /** Parameter */
            readonly parameter: string;
            readonly subject: components["schemas"]["ParameterRef"];
            readonly density_curve: components["schemas"]["DensityCurve"];
            /** Mean */
            readonly mean: number;
            /** Lower */
            readonly lower: number;
            /** Upper */
            readonly upper: number;
            /**
             * Interval Kind
             * @enum {string}
             */
            readonly interval_kind: "hdi" | "equal_tail";
            /** Interval Mass */
            readonly interval_mass: number;
            /** Sd */
            readonly sd: number;
        };
        /**
         * PosteriorPredictiveChecks
         * @description Posterior predictive checks report exact-model checks and their supporting plot data.
         */
        readonly PosteriorPredictiveChecks: {
            /** Per Variable Warnings */
            readonly per_variable_warnings: readonly Domain.Assessment<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence>[];
            /**
             * Checked
             * @default false
             */
            readonly checked: boolean;
            /**
             * N Subsample
             * @default 0
             */
            readonly n_subsample: number;
            /** Overlays */
            readonly overlays: readonly components["schemas"]["PPCOverlay"][];
            /** Test Stats */
            readonly test_stats: readonly components["schemas"]["PPCTestStat"][];
        };
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
        readonly "PotentialMechanismSpec-Output": {
            readonly id: components["schemas"]["MechanismId-Output"];
            readonly expression: components["schemas"]["Expression-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "potential";
        };
        readonly PredictiveAssessment: Domain.Evaluated<Domain.PredictiveSubject, readonly (Domain.NumericCriterionEvidence)[]> | Domain.NotEvaluated<Domain.PredictiveSubject>;
        /**
         * @description A predictive check reason explains why a battery could not be evaluated for the selected model and observations.
         * @enum {string}
         */
        readonly PredictiveCheckReason: "MODEL_INCOMPLETE" | "MODEL_NOT_EXECUTABLE" | "NO_COMPATIBLE_PANEL" | "INSUFFICIENT_OBSERVATION_TIMES" | "SIMULATION_UNSUPPORTED" | "ARCHIVED_MEASUREMENT_NOT_RETAINED";
        /**
         * PredictiveComparison
         * @description A selected reference history retains its role even when checks are unavailable.
         */
        readonly PredictiveComparison: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "comparison";
            /**
             * Reference Side
             * @enum {string}
             */
            readonly reference_side: "left" | "right";
            readonly evaluation: Domain.Evaluation<Domain.PosteriorPredictiveChecks>;
        };
        /** @description A predictive comparison selects one reference history or records why no reference comparison applies. */
        readonly PredictiveComparisonResult: components["schemas"]["PredictiveComparison"] | components["schemas"]["Unavailable"] | components["schemas"]["NotApplicable"];
        readonly PredictiveLawProvenance: components["schemas"]["AuthoredLawProvenance"] | components["schemas"]["FittedLawProvenance"] | components["schemas"]["MixedLawProvenance"] | components["schemas"]["UnknownLawProvenance"];
        /**
         * PredictiveSubject
         * @description One named check and its stable target in a construct's scientific context.
         */
        readonly PredictiveSubject: {
            /** Check */
            readonly check: string;
            /** @default null */
            readonly construct_id: components["schemas"]["ConstructId-Output"] | null;
            /** Target */
            readonly target: components["schemas"]["EntityRef"] | ("whole_model" | "observations");
        };
        /**
         * PrepareDataRequest
         * @description Prepare uploaded sources or a simulation replicate without a model.
         */
        readonly "PrepareDataRequest-Input": {
            /**
             * Action
             * @default prepare_data
             * @constant
             */
            readonly action?: "prepare_data";
            /** Input */
            readonly input: components["schemas"]["FilePreparationSpec-Input"] | components["schemas"]["SimulationReplicateRef-Input"];
        };
        /**
         * PrepareDataRequest
         * @description Prepare uploaded sources or a simulation replicate without a model.
         */
        readonly "PrepareDataRequest-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** Input */
            readonly input: components["schemas"]["FilePreparationSpec-Output"] | components["schemas"]["SimulationReplicateRef-Output"];
        };
        readonly PreparedDataMetadata: components["schemas"]["FilePreparedDataMetadata"] | components["schemas"]["SimulationPreparedDataMetadata"];
        /** @description A progress event records one running attempt's step status or extraction telemetry. */
        readonly ProgressEvent: components["schemas"]["StepEvent"] | components["schemas"]["ExtractionPlanEvent"] | components["schemas"]["ExtractionWorkerEvent"] | components["schemas"]["ExtractionSnapshotEvent"];
        /** @enum {string} */
        readonly ProgressStep: "ingestion" | "extraction";
        /** @description A query name labels one contrast of the study question. */
        readonly QueryName: string;
        /**
         * QueryTargetSubject
         * @description One query's intervention target: defined, identified, or set inside the record.
         */
        readonly QueryTargetSubject: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly check: "identification" | "range" | "target";
            /** Query */
            readonly query: string;
            readonly target: components["schemas"]["ConstructRef-Output"];
        };
        /**
         * QueryWindowSubject
         * @description Whether the record supports one query's window.
         */
        readonly QueryWindowSubject: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly check: "window";
            /** Query */
            readonly query: string;
        };
        /** @description A question assessment records one check of the question against the model or the record. */
        readonly QuestionAssessment: Domain.Evaluated<Domain.QuestionSubject, string> | Domain.NotEvaluated<Domain.QuestionSubject>;
        /**
         * QuestionCheckReport
         * @description The study question checked against the model and, once prepared, the record.
         */
        readonly QuestionCheckReport: {
            readonly question_revision: components["schemas"]["GitOid-Output"];
            readonly panel_revision: components["schemas"]["GitOid-Output"] | null;
            /** Findings */
            readonly findings: readonly components["schemas"]["QuestionAssessment"][];
        };
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
            /**
             * @description The construct whose course answers the question.
             * @default null
             */
            readonly outcome?: components["schemas"]["ConstructId-Input"] | null;
            /**
             * Queries
             * @description Named contrasts against the recorded course, each with at least one intervention.
             */
            readonly queries?: Readonly<Partial<Record<components["schemas"]["QueryName"], components["schemas"]["SimulationSpec-Input"]>>>;
        };
        /**
         * QuestionSpec
         * @description What the study asks: the user's words, the outcome, and named contrasts.
         *
         *     Each query is a contrast of its interventions against the recorded course.
         *     Constructs are named by identity before a model defines them.
         */
        readonly "QuestionSpec-Output": {
            /**
             * Text
             * @description The user's question in their own words.
             */
            readonly text: string;
            /**
             * @description The construct whose course answers the question.
             * @default null
             */
            readonly outcome: components["schemas"]["ConstructId-Output"] | null;
            /**
             * Queries
             * @description Named contrasts against the recorded course, each with at least one intervention.
             */
            readonly queries: Readonly<Partial<Record<components["schemas"]["QueryName"], components["schemas"]["SimulationSpec-Output"]>>>;
        };
        /** @description A question check subject names the outcome, one query's window, or one query's intervention target. */
        readonly QuestionSubject: components["schemas"]["OutcomeSubject"] | components["schemas"]["QueryTargetSubject"] | components["schemas"]["QueryWindowSubject"];
        /** Raised */
        readonly Raised: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "raised";
            /** Error Type */
            readonly error_type: string;
            /** Error Message */
            readonly error_message: string;
            /**
             * Details
             * @default []
             */
            readonly details: readonly string[];
        };
        /**
         * RankHistogram
         * @description Pooled-rank bin counts, grouped in original chain order.
         */
        readonly RankHistogram: {
            readonly subject: components["schemas"]["ParameterRef"];
            /** Expected Per Bin */
            readonly expected_per_bin: number;
            /** Chains */
            readonly chains: readonly (readonly number[])[];
        };
        /**
         * RawDataColumnDescription
         * @description A stored column's physical type and authored interpretation.
         */
        readonly RawDataColumnDescription: {
            /** Name */
            readonly name: string;
            /** Dtype */
            readonly dtype: string;
            /** Description */
            readonly description: string;
        };
        /**
         * RawDataData
         * @description Profile and representative rows from one uploaded table revision.
         */
        readonly RawDataData: {
            /** N Records */
            readonly n_records: number;
            /** N Columns */
            readonly n_columns: number;
            readonly date_range: components["schemas"]["RawDataDateRange"] | null;
            /** Sample */
            readonly sample: readonly {
                readonly [key: string]: string | null;
            }[];
            /** Column Descriptions */
            readonly column_descriptions: readonly components["schemas"]["RawDataColumnDescription"][];
        };
        /**
         * RawDataDateRange
         * @description Observed date bounds of the uploaded table, when it contains a date column.
         */
        readonly RawDataDateRange: {
            /** Start */
            readonly start: string;
            /** End */
            readonly end: string;
        };
        /** RecordDependency */
        readonly RecordDependency: {
            /** Seq */
            readonly seq: number;
            /** Source Seq */
            readonly source_seq: number;
            /** Argument */
            readonly argument: string;
            /**
             * Check
             * @description Only the action's checks read the output; its request did not name it.
             */
            readonly check: boolean;
        };
        /** RecordedPath */
        readonly RecordedPath: {
            /** Draw */
            readonly draw: number;
            /** Values */
            readonly values: readonly (number | null)[];
        };
        /** Rejected */
        readonly Rejected: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "rejected";
            readonly reason: components["schemas"]["RejectionReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * @description A rejection reason identifies the expected input or publication condition that prevented the action.
         * @enum {string}
         */
        readonly RejectionReason: "revision_conflict" | "input_unavailable" | "scientific_inputs" | "recorded_rejection";
        /** Removed[ConstructRef] */
        readonly Removed_ConstructRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "removed";
            readonly before: components["schemas"]["ConstructRef-Output"];
        };
        /** Removed[DataPoint] */
        readonly Removed_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "removed";
            readonly before: components["schemas"]["DataPoint"];
        };
        /** Removed[EdgeRef] */
        readonly Removed_EdgeRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "removed";
            readonly before: components["schemas"]["EdgeRef"];
        };
        /** Removed[ParameterSpec] */
        readonly Removed_ParameterSpec_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "removed";
            readonly before: components["schemas"]["ParameterSpec-Output"];
        };
        /**
         * RetractedArtifact
         * @description A current artifact removed by an action, with the finding that caused it.
         */
        readonly RetractedArtifact: {
            readonly artifact_id: components["schemas"]["ArtifactId"];
            /** Reason Ref */
            readonly reason_ref: string;
        };
        /** Revised[ConstructRef] */
        readonly Revised_ConstructRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "revised";
            readonly before: components["schemas"]["ConstructRef-Output"];
            readonly after: components["schemas"]["ConstructRef-Output"];
        };
        /** Revised[DataPoint] */
        readonly Revised_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "revised";
            readonly before: components["schemas"]["DataPoint"];
            readonly after: components["schemas"]["DataPoint"];
        };
        /** Revised[EdgeRef] */
        readonly Revised_EdgeRef_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "revised";
            readonly before: components["schemas"]["EdgeRef"];
            readonly after: components["schemas"]["EdgeRef"];
        };
        /** Revised[ParameterSpec] */
        readonly Revised_ParameterSpec_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "revised";
            readonly before: components["schemas"]["ParameterSpec-Output"];
            readonly after: components["schemas"]["ParameterSpec-Output"];
        };
        /**
         * Role
         * @description A construct role states whether the variable is modeled as endogenous or treated as
         *     exogenous.
         * @enum {string}
         */
        readonly Role: "endogenous" | "exogenous";
        /** RunningAction */
        readonly RunningAction: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            readonly action: components["schemas"]["ActionId"];
            /** Request */
            readonly request: components["schemas"]["ScientificActionRequest"] | components["schemas"]["DataDiffRequest-Output"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /**
             * Events
             * @default []
             */
            readonly events: readonly components["schemas"]["ProgressEvent"][];
        };
        /** RunningPoll */
        readonly RunningPoll: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "running";
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /** Request */
            readonly request: components["schemas"]["ScientificActionRequest"] | components["schemas"]["DataDiffRequest-Output"];
            /**
             * Messages
             * @default []
             */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /**
             * Events
             * @default []
             */
            readonly events: readonly components["schemas"]["ProgressEvent"][];
        };
        /**
         * SamplerSpec
         * @description Fully resolved controls for the exact particle sampler.
         */
        readonly SamplerSpec: {
            /**
             * Num Warmup
             * @default 4000
             */
            readonly num_warmup: number;
            /**
             * Num Samples
             * @default 1000
             */
            readonly num_samples: number;
            /**
             * Num Chains
             * @default 4
             */
            readonly num_chains: number;
            /**
             * Seed
             * @default 0
             */
            readonly seed: number;
            /**
             * N Particles
             * @default 64
             */
            readonly n_particles: number;
            /**
             * Retain Latent Paths
             * @default true
             */
            readonly retain_latent_paths: boolean;
            readonly marginal_particle_gibbs: components["schemas"]["MarginalParticleGibbsSpec"];
        };
        /**
         * @description A scientific action identity selects setting the question, model editing, data preparation, fitting, or simulation.
         * @enum {string}
         */
        readonly ScientificActionId: "set_question" | "edit_model" | "prepare_data" | "fit" | "simulate";
        readonly ScientificActionRequest: components["schemas"]["SetQuestionRequest-Output"] | components["schemas"]["EditModelRequest-Output"] | components["schemas"]["PrepareDataRequest-Output"] | components["schemas"]["FitRequest-Output"] | components["schemas"]["SimulateRequest-Output"];
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
        readonly "SemanticExtractionSpec-Output": {
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
            readonly source_columns: readonly string[];
        };
        /**
         * SetQuestionRequest
         * @description Set the study question; it is the first action of every study.
         */
        readonly "SetQuestionRequest-Input": {
            /**
             * Action
             * @default set_question
             * @constant
             */
            readonly action?: "set_question";
            readonly question: components["schemas"]["QuestionSpec-Input"];
        };
        /**
         * SetQuestionRequest
         * @description Set the study question; it is the first action of every study.
         */
        readonly "SetQuestionRequest-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "set_question";
            readonly question: components["schemas"]["QuestionSpec-Output"];
        };
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
             * Action
             * @default simulate
             * @constant
             */
            readonly action?: "simulate";
            readonly model_revision: components["schemas"]["GitOid-Input"];
            /**
             * @description Exact panel dating authored-law simulations; fitted laws retain their own origin.
             * @default null
             */
            readonly panel_revision?: components["schemas"]["GitOid-Input"] | null;
        };
        /**
         * SimulateRequest
         * @description Generate a dated window with optional interventions; compare saved data with data_diff.
         */
        readonly "SimulateRequest-Output": {
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
            readonly interventions: readonly components["schemas"]["InterventionSpec-Output"][];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "simulate";
            readonly model_revision: components["schemas"]["GitOid-Output"];
            /**
             * @description Exact panel dating authored-law simulations; fitted laws retain their own origin.
             * @default null
             */
            readonly panel_revision: components["schemas"]["GitOid-Output"] | null;
        };
        /**
         * SimulationEvidence
         * @description Exact generated histories with their production coordinates and input provenance.
         */
        readonly SimulationEvidence: {
            readonly model: components["schemas"]["GitRef"];
            readonly design: components["schemas"]["SimulationSpec-Output"];
            /**
             * Time Origin
             * Format: date-time
             * @description Calendar instant of model day zero.
             */
            readonly time_origin: string;
            /** Times */
            readonly times: readonly [
                number,
                number,
                ...number[]
            ];
            /** Draws */
            readonly draws: number;
            /** Seed */
            readonly seed: number;
            /**
             * @description Panel that supplied the time origin: the fit's panel for fitted laws, otherwise the explicitly named panel when present.
             * @default null
             */
            readonly origin_panel_revision: components["schemas"]["GitOid-Output"] | null;
            /** State Ids */
            readonly state_ids: readonly components["schemas"]["ConstructId-Output"][];
            /** Parameter Draws */
            readonly parameter_draws: {
                readonly [key: string]: string;
            };
            /** Latent Paths */
            readonly latent_paths: string;
            /** Observations */
            readonly observations: string;
            readonly observation_layout: components["schemas"]["SimulationObservationLayout"];
            /**
             * Reference Latent Paths
             * @default null
             */
            readonly reference_latent_paths: string | null;
            /**
             * Reference Observations
             * @default null
             */
            readonly reference_observations: string | null;
            /** Assignments */
            readonly assignments: readonly components["schemas"]["StateAssignment"][];
        };
        /**
         * SimulationObservationLayout
         * @description Saved observation semantics and coordinates; generation truths remain separate.
         */
        readonly SimulationObservationLayout: {
            /** Variables */
            readonly variables: readonly Domain.ObservationSpec<string>[];
            /** Support Start Times */
            readonly support_start_times: string;
            /** Support End Times */
            readonly support_end_times: string;
            /** Mask */
            readonly mask: string;
        };
        /**
         * SimulationPaths
         * @description Contiguous pages of original draws, with every recorded time point intact.
         */
        readonly SimulationPaths: {
            /** Times */
            readonly times: readonly number[];
            /** Time Origin */
            readonly time_origin: string | null;
            /** Total Draws */
            readonly total_draws: number;
            /** Start */
            readonly start: number;
            /** Count */
            readonly count: number;
            /** States */
            readonly states: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], components["schemas"]["PathSeries"]>>>;
            /** Indicators */
            readonly indicators: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["PathSeries"]>>>;
            /** @default null */
            readonly effect: components["schemas"]["PathSeries"] | null;
            /** @default null */
            readonly effect_summary: components["schemas"]["EffectSummary"] | null;
            /**
             * Reference Mean
             * @default null
             */
            readonly reference_mean: number | null;
            /** Manifest Effects */
            readonly manifest_effects: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], number>>>;
            /** Action Category Probabilities */
            readonly action_category_probabilities: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["CategoryProbabilitySummary"]>>>;
            /** Reference Category Probabilities */
            readonly reference_category_probabilities: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["CategoryProbabilitySummary"]>>>;
        };
        /**
         * SimulationPreparedDataMetadata
         * @description A simulation panel retains the schema of its recorded observation history.
         */
        readonly SimulationPreparedDataMetadata: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "simulation";
            readonly source: components["schemas"]["SimulationReplicateRef-Output"];
            /** Variables */
            readonly variables: readonly [
                Domain.ObservationSpec<string>,
                ...Domain.ObservationSpec<string>[]
            ];
            /**
             * Time Origin
             * @description Calendar instant of model day zero; null denotes a calendar-free history.
             */
            readonly time_origin: string | null;
        };
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
            /**
             * Replicate
             * @default null
             */
            readonly replicate?: number | null;
        };
        /**
         * SimulationRef
         * @description A saved simulation; a null replicate selects all its recorded draws.
         */
        readonly "SimulationRef-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "simulation";
            readonly revision: components["schemas"]["GitOid-Output"];
            /**
             * Replicate
             * @default null
             */
            readonly replicate: number | null;
        };
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
        readonly "SimulationReplicateRef-Output": {
            readonly revision: components["schemas"]["GitOid-Output"];
            /** Replicate */
            readonly replicate: number;
        };
        /**
         * SimulationReport
         * @description A simulation report records forward histories, resolved execution settings, and certified effects when supported.
         */
        readonly SimulationReport: {
            readonly evidence: components["schemas"]["SimulationEvidence"];
            readonly law: components["schemas"]["PredictiveLawProvenance"];
            /**
             * Findings
             * @default []
             */
            readonly findings: readonly components["schemas"]["PredictiveAssessment"][];
            readonly fit_reliability: components["schemas"]["FitReliability"];
            readonly causal: Domain.Evaluation<Domain.CausalEffectResult>;
        };
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
        readonly "SimulationSpec-Output": {
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
            readonly interventions: readonly components["schemas"]["InterventionSpec-Output"][];
        };
        /**
         * SourceValidity
         * @description Whether a selected artifact still matches its pinned inputs.
         * @enum {string}
         */
        readonly SourceValidity: "fresh" | "stale";
        /**
         * Sourced[DataProfileArtifact]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_DataProfileArtifact_: {
            readonly value: components["schemas"]["DataProfileArtifact"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[FitSummary]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_FitSummary_: {
            readonly value: components["schemas"]["FitSummary"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[IdentificationReport]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_IdentificationReport_: {
            readonly value: components["schemas"]["IdentificationReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[InferenceReport]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_InferenceReport_: {
            readonly value: components["schemas"]["InferenceReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[MeasurementsData]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_MeasurementsData_: {
            readonly value: components["schemas"]["MeasurementsData"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[ModelPredictiveReport]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_ModelPredictiveReport_: {
            readonly value: components["schemas"]["ModelPredictiveReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[ModelSpec]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_ModelSpec_: {
            readonly value: components["schemas"]["ModelSpec-Output"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[PreparedDataMetadata]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_PreparedDataMetadata_: {
            readonly value: components["schemas"]["PreparedDataMetadata"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[QuestionCheckReport]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_QuestionCheckReport_: {
            readonly value: components["schemas"]["QuestionCheckReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[QuestionSpec]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_QuestionSpec_: {
            readonly value: components["schemas"]["QuestionSpec-Output"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[RawDataData]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_RawDataData_: {
            readonly value: components["schemas"]["RawDataData"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[SimulationReport]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_SimulationReport_: {
            readonly value: components["schemas"]["SimulationReport"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[ValidationReportArtifact]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_ValidationReportArtifact_: {
            readonly value: components["schemas"]["ValidationReportArtifact"];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[tuple[SpecificationAssessment, ...]]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_tuple_SpecificationAssessment__________: {
            /** Value */
            readonly value: readonly components["schemas"]["SpecificationAssessment"][];
            readonly source: components["schemas"]["FactSource"];
        };
        /**
         * Sourced[tuple[StructuralItemDisposition, ...]]
         * @description A sourced value pairs one model finding with its supporting artifact revision.
         */
        readonly Sourced_tuple_StructuralItemDisposition__________: {
            /** Value */
            readonly value: readonly components["schemas"]["StructuralItemDisposition"][];
            readonly source: components["schemas"]["FactSource"];
        };
        readonly SpecificationAssessment: Domain.Evaluated<string, string> | Domain.NotEvaluated<string>;
        /**
         * StateAssignment
         * @description A state set at one model time: a resolved intervention or a replayed input reading.
         */
        readonly StateAssignment: {
            readonly target: components["schemas"]["ConstructId-Output"];
            /**
             * Time
             * @description Absolute time in model days.
             */
            readonly time: number;
            /** Value */
            readonly value: number;
        };
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
        readonly "StateExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "state";
            readonly construct_id: components["schemas"]["ConstructId-Output"];
        };
        /**
         * StepError
         * @description The error type and message of a failed step.
         */
        readonly StepError: {
            /** Type */
            readonly type: string;
            /** Message */
            readonly message: string;
        };
        /**
         * StepEvent
         * @description A data-preparation step changed status.
         */
        readonly StepEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "nof1-causal-lab.step";
            readonly step: components["schemas"]["ProgressStep"];
            readonly status: components["schemas"]["StepStatus"];
            /** @default null */
            readonly error: components["schemas"]["StepError"] | null;
        };
        /** @enum {string} */
        readonly StepStatus: "running" | "completed" | "failed";
        /**
         * StructuralDisposition
         * @description A structural disposition classifies how compilation uses or excludes an authored model
         *     entity.
         * @enum {string}
         */
        readonly StructuralDisposition: "retained_state" | "marginalized" | "identification_only" | "retained_edge" | "projected_edge" | "manifest" | "excluded_indicator" | "unsupported";
        /**
         * StructuralItemDisposition
         * @description An item disposition explains the compilation decision for one identified authored
         *     entity.
         */
        readonly StructuralItemDisposition: {
            /** Target */
            readonly target: components["schemas"]["ConstructRef-Output"] | components["schemas"]["EdgeRef"] | components["schemas"]["IndicatorRef"];
            readonly disposition: components["schemas"]["StructuralDisposition"];
            /** Reason */
            readonly reason: string;
        };
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
        readonly "StudentTLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "StudentT";
            readonly df: components["schemas"]["Expression-Output"];
            readonly loc: components["schemas"]["Expression-Output"];
            readonly scale: components["schemas"]["Expression-Output"];
        };
        /**
         * StudyRevision
         * @description Git publication wraps its already-owned record, without copying its fields.
         */
        readonly StudyRevision: {
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Parent Ids */
            readonly parent_ids: readonly components["schemas"]["GitOid-Output"][];
            readonly record: components["schemas"]["AttemptRecord"];
        };
        /**
         * StudyState
         * @description Study state projects the artifact trees selected by one Git commit.
         *
         *     ``current`` maps artifact id → the revision info that is *current* for the
         *     study. Absent key = the artifact does not exist (either never produced,
         *     or produced-when-nonempty semantics withheld it).
         */
        readonly StudyState: {
            /** Current */
            readonly current: Readonly<Partial<Record<components["schemas"]["ArtifactId"], components["schemas"]["ArtifactRecord"]>>>;
        };
        /**
         * SummaryOperator
         * @description A summary operator specifies how values within a measurement window produce one
         *     observation.
         * @enum {string}
         */
        readonly SummaryOperator: "first" | "last" | "sum" | "count" | "mean" | "std";
        /**
         * TemporalStatus
         * @description Temporal status states whether a construct varies within the individual over time.
         * @enum {string}
         */
        readonly TemporalStatus: "time_varying" | "time_invariant";
        /**
         * TimelineRecord
         * @description A replayable call log, excluding every scientific result and check payload.
         */
        readonly TimelineRecord: {
            /** Seq */
            readonly seq: number;
            /** Ts */
            readonly ts: string;
            /** Messages */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /** Trace Ids */
            readonly trace_ids: readonly string[];
            readonly attempt: Domain.Attempt<Domain.ActionId, Domain.ScientificActionRequest | Domain.DataDiffRequest, null>;
        };
        /** TimelineResponse */
        readonly TimelineResponse: {
            /** Attempts */
            readonly attempts: readonly components["schemas"]["TimelineRevision"][];
            /** Dependencies */
            readonly dependencies: readonly components["schemas"]["RecordDependency"][];
            readonly running: components["schemas"]["RunningAction"] | null;
        };
        /** TimelineRevision */
        readonly TimelineRevision: {
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Parent Ids */
            readonly parent_ids: readonly components["schemas"]["GitOid-Output"][];
            readonly record: components["schemas"]["TimelineRecord"];
        };
        /**
         * TraceMessage
         * @description A trace message records one conversational step, including any reasoning or tool
         *     interaction.
         */
        readonly TraceMessage: {
            /** Role */
            readonly role: string;
            /** Content */
            readonly content: string;
            /**
             * Reasoning
             * @default null
             */
            readonly reasoning: string | null;
            /**
             * Tool Calls
             * @default null
             */
            readonly tool_calls: readonly components["schemas"]["TraceToolCall"][] | null;
            /**
             * Tool Call Id
             * @default null
             */
            readonly tool_call_id: string | null;
            /**
             * Tool Name
             * @default null
             */
            readonly tool_name: string | null;
            /**
             * Tool Result
             * @default null
             */
            readonly tool_result: string | null;
            /**
             * Tool Is Error
             * @default false
             */
            readonly tool_is_error: boolean;
        };
        /**
         * TraceSeries
         * @description Every retained draw, grouped in original chain order.
         */
        readonly TraceSeries: {
            readonly subject: components["schemas"]["ParameterRef"];
            /** Chains */
            readonly chains: readonly (readonly number[])[];
        };
        /**
         * TraceToolCall
         * @description A function invocation with the call identity used to match its result.
         */
        readonly TraceToolCall: {
            /** Id */
            readonly id: string;
            /** Name */
            readonly name: string;
            /** Arguments */
            readonly arguments: string;
        };
        /**
         * TraceUsage
         * @description Trace usage records the input, output, and reasoning tokens consumed by a conversation.
         */
        readonly TraceUsage: {
            /**
             * Input Tokens
             * @default 0
             */
            readonly input_tokens: number;
            /**
             * Output Tokens
             * @default 0
             */
            readonly output_tokens: number;
            /**
             * Reasoning Tokens
             * @default null
             */
            readonly reasoning_tokens: number | null;
        };
        /**
         * Unavailable
         * @description An applicable result could not be produced, for an explicit reason.
         */
        readonly Unavailable: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "unavailable";
            /** Reason */
            readonly reason: string;
        };
        /**
         * UnavailablePredictiveChecks
         * @description The run could not evaluate its scientific battery.
         */
        readonly UnavailablePredictiveChecks: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "unavailable";
            readonly reason: components["schemas"]["PredictiveCheckReason"];
            /**
             * Detail
             * @default null
             */
            readonly detail: string | null;
        };
        /** Unchanged[ConstructRef] */
        readonly Unchanged_ConstructRef_: {
            /**
             * Kind
             * @default unchanged
             * @constant
             */
            readonly kind: "unchanged";
            readonly before: components["schemas"]["ConstructRef-Output"];
            readonly after: components["schemas"]["ConstructRef-Output"];
        };
        /** Unchanged[EdgeRef] */
        readonly Unchanged_EdgeRef_: {
            /**
             * Kind
             * @default unchanged
             * @constant
             */
            readonly kind: "unchanged";
            readonly before: components["schemas"]["EdgeRef"];
            readonly after: components["schemas"]["EdgeRef"];
        };
        /**
         * UnknownLawProvenance
         * @description Imported laws do not establish a conditioning history.
         */
        readonly UnknownLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "unknown";
            /**
             * Interpretation
             * @constant
             */
            readonly interpretation: "unknown";
        };
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
        readonly ValidationIssue: {
            /**
             * @description Affected indicator; null for a dataset-wide issue.
             * @default null
             */
            readonly indicator_id: components["schemas"]["IndicatorId-Output"] | null;
            /** Issue Type */
            readonly issue_type: string;
            /**
             * Severity
             * @enum {string}
             */
            readonly severity: "error" | "warning" | "info";
            /** Message */
            readonly message: string;
        };
        /**
         * ValidationReportArtifact
         * @description Data findings composed with model-dependent execution checks.
         */
        readonly ValidationReportArtifact: {
            readonly data: components["schemas"]["DataProfileArtifact"];
            /** Preflight */
            readonly preflight: readonly components["schemas"]["SpecificationAssessment"][];
            /**
             * Is Valid
             * @description Whether both the data findings and model preflight contain no failures.
             */
            readonly is_valid: boolean;
        };
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
    readonly set_question: {
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
                readonly "application/json": components["schemas"]["SetQuestionRequest-Input"];
            };
        };
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
    readonly edit_model: {
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
                readonly "application/json": components["schemas"]["EditModelRequest-Input"];
            };
        };
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
    readonly prepare_data: {
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
                readonly "application/json": components["schemas"]["PrepareDataRequest-Input"];
            };
        };
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
    readonly fit: {
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
                readonly "application/json": components["schemas"]["FitRequest-Input"];
            };
        };
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
    readonly simulate: {
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
                readonly "application/json": components["schemas"]["SimulateRequest-Input"];
            };
        };
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
    readonly data_diff: {
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
                readonly "application/json": components["schemas"]["DataDiffRequest-Input"];
            };
        };
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
    readonly model_diff: {
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
                readonly "application/json": components["schemas"]["ModelDiffRequest-Input"];
            };
        };
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
}
