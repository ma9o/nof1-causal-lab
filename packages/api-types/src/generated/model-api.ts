/** AUTO-GENERATED from FastAPI OpenAPI. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
    "/api/episodes/{workspace_id}/actions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Execute Scientific Action
         * @description Execute edit_model, prepare_data, fit, or simulate with explicit input revisions.
         */
        post: operations["execute_scientific_action_api_episodes__workspace_id__actions_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Snapshot
         * @description Batch canonical aggregates in one committed read transaction.
         *
         *     Omit `at_seq` for the latest applied move, or select a committed journal sequence.
         *     Zero selects the empty model. Rejected/raised attempts are not revisions (404).
         *     Use the returned `context.seq` for subsequent aggregate or collection reads at the same revision.
         */
        get: operations["get_model_snapshot_api_episodes__workspace_id__model_get"];
        /**
         * Update Model
         * @description Validate and atomically replace the named base model revision.
         */
        put: operations["update_model_api_episodes__workspace_id__model_put"];
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/revisions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Revisions
         * @description List stored model, observation and source revisions for deliberate selection.
         */
        get: operations["get_revisions_api_episodes__workspace_id__revisions_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/revisions/model/{version}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Read Model Revision
         * @description Read a historical definition, including the input to an earlier fit.
         */
        get: operations["read_model_revision_api_episodes__workspace_id__revisions_model__version__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/revisions/compare": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Compare Model Revisions
         * @description Compare fixed/free decisions, laws and scientific dependencies in the backend.
         */
        get: operations["compare_model_revisions_api_episodes__workspace_id__revisions_compare_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/revisions/data-profile/{panel_version}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Read Data Profile
         * @description Read the empirical profile for an observation revision independently of the model.
         */
        get: operations["read_data_profile_api_episodes__workspace_id__revisions_data_profile__panel_version__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/definition": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Definition
         * @description The canonical scientific value selected by this journal revision.
         */
        get: operations["get_model_definition_api_episodes__workspace_id__model_definition_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/inference-report": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Inference Report
         * @description Read the inference transition report associated with the selected model revision.
         */
        get: operations["get_model_inference_report_api_episodes__workspace_id__model_inference_report_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/constructs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Constructs
         * @description Authored constructs, using their canonical domain type.
         */
        get: operations["get_model_constructs_api_episodes__workspace_id__model_constructs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/edges": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Edges
         * @description Authored edges, using their canonical domain type.
         */
        get: operations["get_model_edges_api_episodes__workspace_id__model_edges_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/indicators": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Indicators
         * @description Authored indicators whose owners survive at the selected revision.
         */
        get: operations["get_model_indicators_api_episodes__workspace_id__model_indicators_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/parameters": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Parameters
         * @description Scientific parameter definitions from the selected model, without inference execution.
         */
        get: operations["get_model_parameters_api_episodes__workspace_id__model_parameters_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/views/{artifact_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model View
         * @description One display projection from the selected committed model revision.
         */
        get: operations["get_model_view_api_episodes__workspace_id__model_views__artifact_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** @enum {string} */
        AggregationFunction: "mean" | "sum" | "min" | "max" | "std" | "var" | "last" | "first" | "count" | "median" | "p10" | "p25" | "p75" | "p90" | "p99" | "skew" | "kurtosis" | "iqr" | "range" | "cv" | "entropy" | "instability" | "trend" | "n_unique";
        /** ArtifactFreshness */
        ArtifactFreshness: Domain.ArtifactFreshness;
        /** @enum {string} */
        ArtifactId: Domain.ArtifactId;
        /**
         * ArtifactRef
         * @description An artifact reference identifies the exact stored version that supports a model fact.
         */
        ArtifactRef: Domain.ArtifactRef;
        /**
         * ArtifactVersionInfo
         * @description Artifact version metadata records how a stored artifact was produced and which inputs it
         *     used.
         *
         *     ``derived_from`` pins the exact input versions the payload was computed
         *     from. For root artifacts (user writes) it is empty. ``created_at`` is
         *     stamped by the activity that produced the version — never inside workflow
         *     code, where wall-clock time is non-deterministic.
         */
        ArtifactVersionInfo: Domain.ArtifactVersionInfo;
        /**
         * ArtifactViewResponse
         * @description One available artifact projection returned by the model view endpoint.
         */
        ArtifactViewResponse: Domain.ArtifactViewResponse;
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        "BinaryExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "binary";
            operator: components["schemas"]["BinaryOperator"];
            left: components["schemas"]["Expression-Input"];
            right: components["schemas"]["Expression-Input"];
        };
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        "BinaryExpression-Output": Domain.BinaryExpression;
        /** @enum {string} */
        BinaryOperator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        "CallExpression-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "call";
            function: components["schemas"]["ExpressionFunction"];
            /** Arguments */
            arguments: components["schemas"]["Expression-Input"][];
        };
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        "CallExpression-Output": Domain.CallExpression;
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        "CausalEdgeSpec-Input": {
            /** @description Persistent identity. Preserve when revising the same edge. */
            id: components["schemas"]["EdgeId"];
            /**
             * Mechanisms
             * @default []
             */
            mechanisms?: components["schemas"]["DynamicsMechanismSpec-Input"][];
            /** @description Cause construct; shared endpoints have one identity. */
            cause: components["schemas"]["ConstructSpec-Input"] | components["schemas"]["ConstructRef"];
            /** @description Effect construct; shared endpoints have one identity. */
            effect: components["schemas"]["ConstructSpec-Input"] | components["schemas"]["ConstructRef"];
            /**
             * Description
             * @description Theoretical justification for this causal link
             */
            description: string;
            /**
             * Lagged
             * @description If True, effect at t is caused by cause at t-1 (one model_clock tick delay). If False (contemporaneous), effect at t is caused by cause at t.
             * @default true
             */
            lagged?: boolean;
            /**
             * Sources
             * @description Literature sources supporting this causal link
             */
            sources?: components["schemas"]["LiteratureSource"][];
        };
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        "CausalEdgeSpec-Output": Domain.CausalEdgeSpec;
        /**
         * CausalSimulationSpec
         * @description An identified paired scenario, certified against the selected production fit.
         */
        "CausalSimulationSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "causal";
            query: components["schemas"]["ScenarioRequest-Input"];
            /**
             * Draws
             * @default 100
             */
            draws?: number;
            /**
             * Seed
             * @default 0
             */
            seed?: number;
            /**
             * Process Noise
             * @default false
             */
            process_noise?: boolean;
            /**
             * Observation Noise
             * @default false
             */
            observation_noise?: boolean;
        };
        /**
         * CausalSimulationSpec
         * @description An identified paired scenario, certified against the selected production fit.
         */
        "CausalSimulationSpec-Output": Domain.CausalSimulationSpec;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        CoefficientExpression: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "coefficient";
            role: components["schemas"]["CoefficientRole"];
            /**
             * Value
             * @description Finite literal or persistent parameter ID; null leaves the operand unassigned.
             */
            value?: number | components["schemas"]["ParameterId"] | null;
            /**
             * Construct Ids
             * @description Additional constructs participating in this coefficient use.
             * @default []
             */
            construct_ids?: components["schemas"]["ConstructId"][];
        };
        /** @enum {string} */
        CoefficientRole: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
        ConstructId: string;
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        ConstructRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "construct";
            id: components["schemas"]["ConstructId"];
        };
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        "ConstructSpec-Input": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            id: components["schemas"]["ConstructId"];
            /**
             * Name
             * @description Construct name (e.g., 'stress', 'sleep_quality')
             */
            name: string;
            /**
             * Description
             * @description What this theoretical construct represents
             */
            description: string;
            /**
             * Indicators
             * @default []
             */
            indicators?: components["schemas"]["IndicatorSpec-Input"][];
            /**
             * Dynamics
             * @default []
             */
            dynamics?: components["schemas"]["DynamicsMechanismSpec-Input"][];
            /**
             * Coefficients
             * @default []
             */
            coefficients?: components["schemas"]["CoefficientExpression"][];
            /**
             * Innovation Family
             * @default gaussian
             * @enum {string}
             */
            innovation_family?: "gaussian" | "student_t";
            /** @description Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points. */
            distribution?: components["schemas"]["DistributionId"] | null;
            /** @description 'endogenous' or 'exogenous' (no modeled causal parents; may still be uncertain) */
            role: components["schemas"]["Role"];
            /** @description 'time_varying' (changes over time) or 'time_invariant' (fixed) */
            temporal_status: components["schemas"]["TemporalStatus"];
        };
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        "ConstructSpec-Output": Domain.ConstructSpec;
        /**
         * DataProfileArtifact
         * @description Model-independent empirical measurements and data-quality findings.
         */
        DataProfileArtifact: Domain.DataProfileArtifact;
        /**
         * DensityPoint
         * @description A plotting coordinate evaluated from the native prior's log density.
         */
        DensityPoint: Domain.DensityPoint;
        /** @description A native law whose membership is defined by the model's scientific quantities. */
        DistributionId: string;
        /**
         * DynamicsMechanismSpec
         * @description A symbolic specification of an additive drift term or a node potential.
         *
         *     A node potential contributes its negative gradient to the drift.
         */
        "DynamicsMechanismSpec-Input": {
            id: components["schemas"]["MechanismId"];
            /**
             * Kind
             * @default drift
             * @enum {string}
             */
            kind?: "drift" | "potential";
            expression: components["schemas"]["Expression-Input"];
        };
        /**
         * DynamicsMechanismSpec
         * @description A symbolic specification of an additive drift term or a node potential.
         *
         *     A node potential contributes its negative gradient to the drift.
         */
        "DynamicsMechanismSpec-Output": Domain.DynamicsMechanismSpec;
        EdgeId: string;
        /**
         * EdgeRef
         * @description An edge reference identifies a causal relationship independently of edits to its
         *     definition.
         */
        EdgeRef: Domain.EdgeRef;
        /**
         * EditModelRequest
         * @description Replace one named base revision with a validated scientific definition.
         */
        EditModelRequest: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            action: "edit_model";
            /** Expected Version */
            expected_version: number;
            model: components["schemas"]["ModelSpec-Input"];
        };
        /**
         * EffectSummary
         * @description An effect summary reports posterior location, uncertainty, and sign probability.
         */
        EffectSummary: Domain.EffectSummary;
        /**
         * EffectTrajectoryPoint
         * @description An effect trajectory point records a causal delta at one elapsed rollout time.
         */
        EffectTrajectoryPoint: Domain.EffectTrajectoryPoint;
        /**
         * EpisodeState
         * @description Episode state identifies the artifact versions currently selected by the transition
         *     journal.
         *
         *     ``current`` maps artifact id → the version info that is *current* for the
         *     episode. Absent key = the artifact does not exist (either never produced,
         *     or produced-when-nonempty semantics withheld it).
         */
        EpisodeState: Domain.EpisodeState;
        "Expression-Input": components["schemas"]["LiteralExpression"] | components["schemas"]["StateExpression"] | components["schemas"]["CoefficientExpression"] | components["schemas"]["BinaryExpression-Input"] | components["schemas"]["CallExpression-Input"];
        "Expression-Output": Domain.Expression;
        /** @enum {string} */
        ExpressionFunction: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact version and records its freshness.
         */
        FactSource: Domain.FactSource;
        /**
         * FitRequest
         * @description Condition explicitly selected model and observation revisions.
         */
        FitRequest: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            action: "fit";
            /** Model Version */
            model_version: number;
            /** Panel Version */
            panel_version: number;
            settings?: components["schemas"]["FitSettingsSpec"];
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured particle sampler.
         */
        FitSettingsSpec: {
            /** Num Samples */
            num_samples?: number | null;
            /** Num Warmup */
            num_warmup?: number | null;
            /** Num Chains */
            num_chains?: number | null;
            /** N Particles */
            n_particles?: number | null;
            /** Seed */
            seed?: number | null;
        };
        /**
         * FitSummary
         * @description A fit read contains the inference log report and server-composed display findings.
         */
        FitSummary: Domain.FitSummary;
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * HistogramBin
         * @description A histogram bin gives its interval, center, and number of posterior draws.
         */
        HistogramBin: Domain.HistogramBin;
        /**
         * IdentificationReport
         * @description Positive and negative causal identification findings for the model's default query.
         */
        IdentificationReport: Domain.IdentificationReport;
        /**
         * IdentifiedTreatmentStatus
         * @description Details on how a treatment effect is identified.
         */
        IdentifiedTreatmentStatus: Domain.IdentifiedTreatmentStatus;
        /**
         * IndicatorAudit
         * @description An indicator audit combines its empirical data profile with the results of validation
         *     checks.
         */
        IndicatorAudit: Domain.IndicatorAudit;
        /**
         * IndicatorEmpiricalProfile
         * @description An empirical profile summarizes an indicator's observed values, coverage, and data-
         *     quality signals.
         */
        IndicatorEmpiricalProfile: Domain.IndicatorEmpiricalProfile;
        IndicatorId: string;
        /**
         * IndicatorPolarity
         * @description Indicator polarity states whether a measurement increases or decreases with its
         *     construct.
         * @enum {string}
         */
        IndicatorPolarity: "positive" | "negative";
        /**
         * IndicatorRef
         * @description An indicator reference identifies a measurement definition independently of its name or
         *     revision.
         */
        IndicatorRef: Domain.IndicatorRef;
        /**
         * IndicatorSpec
         * @description A specification of a construct's observed measurement and how to extract it.
         */
        "IndicatorSpec-Input": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            id: components["schemas"]["IndicatorId"];
            likelihood?: components["schemas"]["LikelihoodSpec-Input"] | null;
            /**
             * Name
             * @description Indicator name (e.g., 'hrv', 'self_reported_stress')
             */
            name: string;
            /**
             * How To Measure
             * @description Instructions for workers on how to extract this from data
             */
            how_to_measure: string;
            /** @description Whether higher indicator values move in the same direction as the construct (`positive`) or the opposite direction (`negative`). */
            construct_polarity: components["schemas"]["IndicatorPolarity"];
            /** @description 'continuous', 'binary', 'count', 'ordinal', 'categorical' */
            measurement_dtype: components["schemas"]["MeasurementDtype"];
            /** @description Aggregation function applied when bucketing raw extractions within the indicator support window. Measurement-structure support is currently limited to: first, last, sum, count, mean, std. Available parser operators: count, cv, entropy, first, instability, iqr, kurtosis, last, max, mean, median, min, n_unique, p10, p25, p75, p90, p99, range, skew, std, sum, trend, var */
            aggregation: components["schemas"]["AggregationFunction"];
            /**
             * Recording
             * @description Source recording semantics within the raw dataset's covered time span. samples: absent readings are unknown. events: a complete event record; empty sum/count windows are zero. changes: a complete change record; the last recorded value persists, with leading gaps unknown. events and changes require computed extraction.
             * @default samples
             * @enum {string}
             */
            recording?: "samples" | "events" | "changes";
            /**
             * Observation Window
             * @description Optional duration string describing the support window summarized by this indicator (for example '1mo' for a monthly average on a daily model clock). If omitted, the support window defaults to the global model_clock.
             */
            observation_window?: string | null;
            /**
             * Ordinal Levels
             * @description Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
             */
            ordinal_levels?: string[] | null;
            /**
             * Categorical Levels
             * @description Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
             */
            categorical_levels?: string[] | null;
            /**
             * Source Columns
             * @description Raw data column names referenced by how_to_measure. Used to project chunks to only relevant columns before extraction.
             */
            source_columns?: string[];
            /** @description Optional deterministic support-window expression for extraction_mode='computed'. Use this when a computed indicator needs formulas, thresholds, or multiple source columns instead of a direct single-column aggregation. The expression must return one scalar per support window. */
            computed_rule?: components["schemas"]["WindowExpression"] | null;
            /**
             * Extraction Mode
             * @description 'computed' (deterministic pipeline extraction) or 'semantic' (LLM extraction). Use 'computed' when the indicator can be derived deterministically either from a direct source-column aggregation or from a computed_rule support-window expression over the declared source_columns.
             * @default semantic
             * @enum {string}
             */
            extraction_mode?: "computed" | "semantic";
        };
        /**
         * IndicatorSpec
         * @description A specification of a construct's observed measurement and how to extract it.
         */
        "IndicatorSpec-Output": Domain.IndicatorSpec;
        /**
         * InferenceMetadata
         * @description Inference metadata records the sampling method, sample count, and run duration.
         */
        InferenceMetadata: Domain.InferenceMetadata;
        /**
         * InferenceReport
         * @description Display findings recorded by an inference transition, separate from ModelSpec.
         */
        InferenceReport: Domain.InferenceReport;
        /** @enum {string} */
        JournalStatus: Domain.JournalStatus;
        "JsonArray-Input": Domain.JsonArray;
        "JsonArray-Output": Domain.JsonArray;
        "JsonObject-Input": Domain.JsonObject;
        "JsonObject-Output": Domain.JsonObject;
        JsonScalar: boolean | number | string | null;
        "JsonValue-Input": Domain.JsonValue;
        "JsonValue-Output": Domain.JsonValue;
        /**
         * LOODiagnostics
         * @description Leave-one-out diagnostics assess predictive fit and the reliability of its cross-
         *     validation estimate.
         *
         *     Exact emission factors on joint parameter/state draws support holding out
         *     one measurement row. All other rows, including future rows, are available
         *     for interpolation. PSIS reliability is assessed with Pareto-k diagnostics.
         */
        LOODiagnostics: Domain.LOODiagnostics;
        /**
         * LikelihoodDiagnostics
         * @description Observed values and validation profile for one likelihood's pinned panel.
         */
        LikelihoodDiagnostics: Domain.LikelihoodDiagnostics;
        /**
         * LikelihoodSpec
         * @description An indicator's conditional probability law and its scientific justification.
         */
        "LikelihoodSpec-Input": {
            law: components["schemas"]["ObservationLawSpec-Input"];
            /**
             * Standardized
             * @description Whether observations are mean-centered and scaled before fitting.
             * @default false
             */
            standardized?: boolean;
            /**
             * Reasoning
             * @description Why this conditional law was chosen for the indicator
             */
            reasoning: string;
            /**
             * Sources
             * @default []
             */
            sources?: components["schemas"]["LiteratureSource"][];
        };
        /**
         * LikelihoodSpec
         * @description An indicator's conditional probability law and its scientific justification.
         */
        "LikelihoodSpec-Output": Domain.LikelihoodSpec;
        /**
         * LiteralExpression
         * @description A finite scalar constant in a model equation.
         */
        LiteralExpression: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "literal";
            /** Value */
            value: number;
        };
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        LiteratureSource: {
            /**
             * Title
             * @description Title of the source (paper, meta-analysis, textbook, etc.)
             */
            title: string;
            /**
             * Url
             * @description URL of the source if available
             */
            url?: string | null;
            /**
             * Snippet
             * @description Relevant excerpt or paraphrase from the source
             */
            snippet: string;
        };
        /** @enum {string} */
        MeasurementDtype: "continuous" | "binary" | "count" | "ordinal" | "categorical";
        /**
         * MeasurementsData
         * @description Counts and representative observations read directly from one panel version.
         */
        MeasurementsData: Domain.MeasurementsData;
        MechanismId: string;
        /** ModelComparison */
        ModelComparison: Domain.ModelComparison;
        /**
         * ModelData
         * @description Observed evidence paired with its source versions.
         */
        ModelData: Domain.ModelData;
        /**
         * ModelDiagnostics
         * @description Server-derived equations and comparisons with pinned observations.
         */
        ModelDiagnostics: Domain.ModelDiagnostics;
        /**
         * ModelFindings
         * @description ModelSpec findings collect identification, validation, and fitted results with their provenance.
         */
        ModelFindings: Domain.ModelFindings;
        /**
         * ModelRevision
         * @description The workspace and version of the scientific design supporting an inference.
         */
        ModelRevision: Domain.ModelRevision;
        /**
         * ModelSnapshot
         * @description The canonical scientific definition with independently sourced inputs and findings.
         */
        ModelSnapshot: Domain.ModelSnapshot;
        /**
         * ModelSpec
         * @description An evolving research question and connected causal graph with owned scientific detail.
         */
        "ModelSpec-Input": {
            /** Question */
            question?: string | null;
            /**
             * Edges
             * @default []
             */
            edges?: components["schemas"]["CausalEdgeSpec-Input"][];
            /**
             * Parameters
             * @default []
             */
            parameters?: components["schemas"]["ParameterSpec"][];
            /**
             * Distributions
             * @description All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements.
             */
            distributions?: {
                [key: string]: components["schemas"]["NumPyroDistribution-Input"];
            };
            /**
             * Time Points
             * @default []
             */
            time_points?: number[];
            /** Measurement Clock */
            measurement_clock?: string | null;
            default_outcome?: components["schemas"]["ConstructId"] | null;
        };
        /**
         * ModelSpec
         * @description An evolving research question and connected causal graph with owned scientific detail.
         */
        "ModelSpec-Output": Domain.ModelSpec;
        /** ModelUpdateBody */
        ModelUpdateBody: {
            /** Expected Version */
            expected_version: number;
            model: components["schemas"]["ModelSpec-Input"];
        };
        /**
         * MoveOutcome
         * @description A move outcome reports the attempted transition and resulting committed state.
         */
        MoveOutcome: Domain.MoveOutcome;
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        NonIdentifiableTreatmentStatus: Domain.NonIdentifiableTreatmentStatus;
        "NumPyroDistribution-Input": {
            /** Distribution */
            distribution: string;
            /** Params */
            params: {
                [key: string]: components["schemas"]["JsonValue-Input"];
            };
        };
        "NumPyroDistribution-Output": Domain.NumPyroDistribution;
        /**
         * ObservationLawSpec
         * @description A symbolic specification of an indicator's conditional observation distribution.
         */
        "ObservationLawSpec-Input": {
            /**
             * Distribution
             * @enum {string}
             */
            distribution: "Delta" | "Normal" | "StudentT" | "Poisson" | "Gamma" | "Bernoulli" | "NegativeBinomial2" | "Beta" | "OrderedLogistic" | "Categorical";
            /** Arguments */
            arguments: {
                [key: string]: components["schemas"]["Expression-Input"];
            };
        };
        /**
         * ObservationLawSpec
         * @description A symbolic specification of an indicator's conditional observation distribution.
         */
        "ObservationLawSpec-Output": Domain.ObservationLawSpec;
        /**
         * ObservationRecord
         * @description Canonical serialized extraction observation row.
         */
        ObservationRecord: Domain.ObservationRecord;
        /**
         * PPCOverlay
         * @description A predictive overlay compares observed values with posterior predictive bands for one
         *     indicator.
         *
         *     Provides the data for Gabry's ppc_dens_overlay / ppc_ribbon plots:
         *     observed time series vs posterior predictive quantile bands.
         *     Optionally includes individual y_rep draw lines for spaghetti plots.
         */
        PPCOverlay: Domain.PPCOverlay;
        /**
         * PPCTestStat
         * @description A predictive test statistic compares an observed summary with its distribution under
         *     replicated data.
         *
         *     Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
         *     with a vertical line at T(y_observed).
         */
        PPCTestStat: Domain.PPCTestStat;
        /**
         * PPCWarning
         * @description A predictive-check finding records whether one indicator passes a calibration,
         *     dependence, or variance check.
         */
        PPCWarning: Domain.PPCWarning;
        /** ParameterChange */
        ParameterChange: Domain.ParameterChange;
        ParameterElementId: Domain.ParameterElementId;
        ParameterId: string;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        ParameterRef: Domain.ParameterRef;
        /**
         * ParameterSpec
         * @description A named quantity's current uncertainty; component slots define its meaning.
         */
        ParameterSpec: {
            id: components["schemas"]["ParameterId"];
            /**
             * Name
             * @description Authored parameter label; relationships use its persistent ID
             */
            name: string;
            /**
             * Description
             * @description Human-readable description of what this parameter represents
             */
            description: string;
            /**
             * Distribution Transform
             * @default identity
             * @enum {string}
             */
            distribution_transform?: "identity" | "dt_persistence_to_ct_decay" | "dt_effect_to_ct_rate" | "initial_state_correlation";
            /**
             * Value
             * @description Known constant on the model quantity scale, exclusive with a distribution.
             */
            value?: number | null;
            /** @description Membership in a native law in ModelSpec.distributions; may be joint. */
            distribution?: components["schemas"]["DistributionId"] | null;
            /** Reference Interval Days */
            reference_interval_days?: number | null;
        };
        /**
         * PosteriorEstimate
         * @description A posterior estimate reports a mean and a credible interval with explicit semantics.
         */
        PosteriorEstimate: Domain.PosteriorEstimate;
        /**
         * PosteriorMarginal
         * @description A posterior marginal summarizes uncertainty in one scalar parameter and supplies its
         *     density plot.
         */
        PosteriorMarginal: Domain.PosteriorMarginal;
        /**
         * PosteriorPair
         * @description A posterior pair supplies joint samples of two parameters to visualize their dependence.
         */
        PosteriorPair: Domain.PosteriorPair;
        /**
         * PosteriorPredictiveChecks
         * @description Posterior predictive checks report exact-model checks and their supporting plot data.
         */
        PosteriorPredictiveChecks: Domain.PosteriorPredictiveChecks;
        /**
         * PrepareDataRequest
         * @description Import uploaded files or extract measurements from a pinned source table.
         */
        PrepareDataRequest: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            action: "prepare_data";
            /**
             * Source
             * @enum {string}
             */
            source: "files" | "raw_data";
            /** Raw Data Version */
            raw_data_version?: number | null;
            /** Model Version */
            model_version?: number | null;
            /** Max Windows */
            max_windows?: number | null;
        };
        /**
         * PriorPredictiveDiagnostic
         * @description A measured prior-predictive check and its evaluation criteria.
         */
        PriorPredictiveDiagnostic: Domain.PriorPredictiveDiagnostic;
        /**
         * PriorPredictiveResult
         * @description Simulated observations and checks recorded by a model-authoring operation.
         */
        PriorPredictiveResult: Domain.PriorPredictiveResult;
        /** @enum {string} */
        Provenance: Domain.Provenance;
        /**
         * RawDataColumnDescription
         * @description A stored column's physical type and authored interpretation.
         */
        RawDataColumnDescription: Domain.RawDataColumnDescription;
        /**
         * RawDataData
         * @description Profile and representative rows from one uploaded table version.
         */
        RawDataData: Domain.RawDataData;
        /**
         * RawDataDateRange
         * @description Observed date bounds of the uploaded table, when it contains a date column.
         */
        RawDataDateRange: Domain.RawDataDateRange;
        /**
         * RetractedArtifact
         * @description A current artifact removed by a move, with the finding that caused it.
         */
        RetractedArtifact: Domain.RetractedArtifact;
        /** RevisionCatalog */
        RevisionCatalog: Domain.RevisionCatalog;
        /**
         * Role
         * @description A construct role states whether the variable is modeled as endogenous or treated as
         *     exogenous.
         * @enum {string}
         */
        Role: "endogenous" | "exogenous";
        /**
         * ScenarioClamp
         * @description A do-operator on one latent variable over a time window.
         *
         *     The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
         *     the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
         *     value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
         *     interpolates across the window, and ``trajectory`` tracks a list of values across it.
         */
        ScenarioClamp: {
            /** @description Persistent ID of the construct to clamp. */
            target: components["schemas"]["ConstructId"];
            /**
             * Mode
             * @description How the clamped value is specified over the window.
             * @enum {string}
             */
            mode: "set" | "shift" | "ramp" | "trajectory";
            /**
             * Value
             * @description Required when mode='set'. Absolute latent-space value.
             */
            value?: number | null;
            /**
             * Amount
             * @description Required when mode='shift'. Additive delta from the start-state value.
             */
            amount?: number | null;
            /**
             * Value Start
             * @description Required when mode='ramp'. Value at from_day.
             */
            value_start?: number | null;
            /**
             * Value End
             * @description Required when mode='ramp'. Value at to_day.
             */
            value_end?: number | null;
            /**
             * Values
             * @description Required when mode='trajectory'. Values sampled evenly across the window.
             */
            values?: number[] | null;
            /**
             * From Day
             * @description Window onset in days from the rollout start.
             * @default 0
             */
            from_day?: number;
            /**
             * To Day
             * @description Window end in days from the rollout start. Null runs through the horizon.
             */
            to_day?: number | null;
        };
        /** ScenarioQueryInput */
        ScenarioQueryInput: {
            /**
             * Estimand
             * @description Report the final-horizon outcome effect or the full effect trajectory.
             * @default trajectory
             * @enum {string}
             */
            estimand?: "end_state" | "trajectory";
            /**
             * Horizon Days
             * @description Forward horizon in days from the rollout start.
             * @default 30
             */
            horizon_days?: number;
            /**
             * Projection
             * @description Report latent outcome effects, manifest projections, or both.
             * @default latent
             * @enum {string}
             */
            projection?: "latent" | "manifest" | "both";
        };
        /**
         * ScenarioRequest
         * @description One reusable request for an on-demand simulation of a fitted model.
         */
        "ScenarioRequest-Input": {
            start?: components["schemas"]["ScenarioStartInput"];
            /**
             * Clamps
             * @description One or more timed latent clamps composing the scenario.
             */
            clamps: components["schemas"]["ScenarioClamp"][];
            /** @description Persistent ID of the requested outcome. */
            outcome: components["schemas"]["ConstructId"];
            readout?: components["schemas"]["ScenarioQueryInput"];
        };
        /**
         * ScenarioRequest
         * @description One reusable request for an on-demand simulation of a fitted model.
         */
        "ScenarioRequest-Output": Domain.ScenarioRequest;
        /**
         * ScenarioStartInput
         * @description Where the forward rollout begins (replaces the rung-2/rung-3 split).
         */
        ScenarioStartInput: {
            /**
             * Kind
             * @description 'baseline' starts from the deterministic drift equilibrium (an interventional, rung-2 query). 'abducted' conditions on the individual's observed evidence and starts from the recovered fitted latent state (a counterfactual, rung-3 query).
             * @default baseline
             * @enum {string}
             */
            kind?: "baseline" | "abducted";
            /**
             * Time Index
             * @description Abducted start only: observed fitted-state index to begin from. Defaults to the final retained fitted latent state.
             */
            time_index?: number | null;
            /**
             * Time
             * @description Abducted start only: ISO-8601 observed timestamp matching a retained fitted latent state. Use either time_index or time, not both.
             */
            time?: string | null;
        };
        /**
         * SimulateRequest
         * @description Simulate current model uncertainty, optionally comparing with observations.
         */
        SimulateRequest: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            action: "simulate";
            /** Model Version */
            model_version: number;
            design: components["schemas"]["SimulationDesign-Input"];
            /** Comparison Panel Version */
            comparison_panel_version?: number | null;
        };
        "SimulationDesign-Input": components["schemas"]["SimulationSpec-Input"] | components["schemas"]["CausalSimulationSpec-Input"];
        "SimulationDesign-Output": Domain.SimulationDesign;
        /**
         * SimulationFinding
         * @description A measured quantity with the criterion used to interpret it.
         */
        SimulationFinding: Domain.SimulationFinding;
        /**
         * SimulationReport
         * @description Evidence from one explicit simulation, separate from an inference report.
         */
        SimulationReport: Domain.SimulationReport;
        /**
         * SimulationResult
         * @description Ephemeral response to a runtime simulation request.
         *
         *     This engine integrates the true nonlinear drift for each posterior draw.
         *     It does not include future process noise or claim the mean of the SDE.
         */
        SimulationResult: Domain.SimulationResult;
        /**
         * SimulationSpec
         * @description A replicated study generated from the selected model's current laws.
         */
        "SimulationSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "trajectory";
            /** Times */
            times: number[];
            /**
             * Draws
             * @default 100
             */
            draws?: number;
            /**
             * Seed
             * @default 0
             */
            seed?: number;
            /**
             * Edge Contrasts
             * @default false
             */
            edge_contrasts?: boolean;
            /**
             * Initial State
             * @default new_study
             * @enum {string}
             */
            initial_state?: "new_study" | "retained" | "fixed" | "equilibrium";
            /** State Time */
            state_time?: number | null;
            /** State Values */
            state_values?: {
                [key: string]: number;
            };
            /**
             * Process Noise
             * @default true
             */
            process_noise?: boolean;
            /**
             * Observation Noise
             * @default true
             */
            observation_noise?: boolean;
            /**
             * Interventions
             * @default []
             */
            interventions?: components["schemas"]["ScenarioClamp"][];
            /**
             * Context
             * @default exploration
             * @enum {string}
             */
            context?: "exploration" | "calibration" | "prediction";
            /**
             * Checks
             * @default [
             *       "dynamics",
             *       "measurement",
             *       "data_comparison"
             *     ]
             */
            checks?: ("dynamics" | "measurement" | "data_comparison")[];
            /**
             * Confinement Growth Ratio
             * @default 5
             */
            confinement_growth_ratio?: number;
            /**
             * Confinement Failure Fraction
             * @default 0.01
             */
            confinement_failure_fraction?: number;
            /**
             * Comparison Time Offset
             * @default 0
             */
            comparison_time_offset?: number;
        };
        /**
         * SimulationSpec
         * @description A replicated study generated from the selected model's current laws.
         */
        "SimulationSpec-Output": Domain.SimulationSpec;
        /**
         * SimulationTrajectory
         * @description One construct's mean reference and intervention paths across simulated draws.
         */
        SimulationTrajectory: Domain.SimulationTrajectory;
        /**
         * SnapshotContext
         * @description A snapshot context identifies the selected journal revision and its artifact versions.
         */
        SnapshotContext: Domain.SnapshotContext;
        /**
         * SourceValidity
         * @description Source validity records whether a fact still matches its pinned inputs.
         * @enum {string}
         */
        SourceValidity: Domain.SourceValidity;
        /** Sourced[FitSummary] */
        Sourced_FitSummary_: {
            value: components["schemas"]["FitSummary"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[IdentificationReport] */
        Sourced_IdentificationReport_: {
            value: components["schemas"]["IdentificationReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[InferenceReport] */
        Sourced_InferenceReport_: {
            value: components["schemas"]["InferenceReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[MeasurementsData] */
        Sourced_MeasurementsData_: {
            value: components["schemas"]["MeasurementsData"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[ModelSpec] */
        Sourced_ModelSpec_: {
            value: components["schemas"]["ModelSpec-Output"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[PriorPredictiveResult] */
        Sourced_PriorPredictiveResult_: {
            value: components["schemas"]["PriorPredictiveResult"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[RawDataData] */
        Sourced_RawDataData_: {
            value: components["schemas"]["RawDataData"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[SimulationReport] */
        Sourced_SimulationReport_: {
            value: components["schemas"]["SimulationReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[SpecificationReport] */
        Sourced_SpecificationReport_: {
            value: components["schemas"]["SpecificationReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[ValidationReportArtifact] */
        Sourced_ValidationReportArtifact_: {
            value: components["schemas"]["ValidationReportArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[tuple[StructuralItemDisposition, ...]] */
        Sourced_tuple_StructuralItemDisposition__________: {
            /** Value */
            value: components["schemas"]["StructuralItemDisposition"][];
            source: components["schemas"]["FactSource"];
        };
        /**
         * SpecificationFinding
         * @description One model-only check and its current evaluation status.
         */
        SpecificationFinding: Domain.SpecificationFinding;
        /**
         * SpecificationReport
         * @description Model-only findings; data compatibility has its own paired provenance.
         */
        SpecificationReport: Domain.SpecificationReport;
        /**
         * StateEquation
         * @description A continuous-time state equation rendered from declared scientific mechanisms.
         */
        StateEquation: Domain.StateEquation;
        /**
         * StateExpression
         * @description A construct's state or declared known input, referenced by identity.
         */
        StateExpression: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            kind: "state";
            construct_id: components["schemas"]["ConstructId"];
        };
        /**
         * StructuralDisposition
         * @description A structural disposition classifies how compilation uses or excludes an authored model
         *     entity.
         * @enum {string}
         */
        StructuralDisposition: Domain.StructuralDisposition;
        /**
         * StructuralItemDisposition
         * @description An item disposition explains the compilation decision for one identified authored
         *     entity.
         */
        StructuralItemDisposition: Domain.StructuralItemDisposition;
        /**
         * TemporalStatus
         * @description Temporal status states whether a construct varies within the individual over time.
         * @enum {string}
         */
        TemporalStatus: "time_varying" | "time_invariant";
        /**
         * TransitionRef
         * @description An immutable entry in the workspace transition journal.
         */
        TransitionRef: Domain.TransitionRef;
        /** ValidationError */
        ValidationError: {
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
            /** Input */
            input?: unknown;
            /** Context */
            ctx?: Record<string, never>;
        };
        /**
         * ValidationIssue
         * @description A validation issue explains a data problem and its severity for an indicator or the
         *     dataset.
         */
        ValidationIssue: Domain.ValidationIssue;
        /**
         * ValidationReportArtifact
         * @description A validation report summarizes whether extracted measurements satisfy the required data
         *     checks.
         */
        ValidationReportArtifact: Domain.ValidationReportArtifact;
        /** @description Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values. */
        WindowExpression: string;
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    execute_scientific_action_api_episodes__workspace_id__actions_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["EditModelRequest"] | components["schemas"]["PrepareDataRequest"] | components["schemas"]["FitRequest"] | components["schemas"]["SimulateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MoveOutcome"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_snapshot_api_episodes__workspace_id__model_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelSnapshot"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_model_api_episodes__workspace_id__model_put: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ModelUpdateBody"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelSnapshot"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_revisions_api_episodes__workspace_id__revisions_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RevisionCatalog"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    read_model_revision_api_episodes__workspace_id__revisions_model__version__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
                version: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelSpec-Output"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    compare_model_revisions_api_episodes__workspace_id__revisions_compare_get: {
        parameters: {
            query: {
                before: number;
                after: number;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelComparison"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    read_data_profile_api_episodes__workspace_id__revisions_data_profile__panel_version__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
                panel_version: number;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DataProfileArtifact"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_definition_api_episodes__workspace_id__model_definition_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Sourced_ModelSpec_"] | null;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_inference_report_api_episodes__workspace_id__model_inference_report_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Sourced_InferenceReport_"] | null;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_constructs_api_episodes__workspace_id__model_constructs_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ConstructSpec-Output"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_edges_api_episodes__workspace_id__model_edges_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CausalEdgeSpec-Output"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_indicators_api_episodes__workspace_id__model_indicators_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IndicatorSpec-Output"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_parameters_api_episodes__workspace_id__model_parameters_get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ParameterSpec"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_model_view_api_episodes__workspace_id__model_views__artifact_id__get: {
        parameters: {
            query?: {
                at_seq?: number | null;
            };
            header?: never;
            path: {
                workspace_id: string;
                artifact_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ArtifactViewResponse"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
