/** AUTO-GENERATED from FastAPI OpenAPI. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
    "/api/studies/{workspace_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Study
         * @description Current study state: the single read to poll while navigating.
         *
         *     Returns the four scientific action names and per-artifact existence,
         *     freshness and revision from the selected Git branch snapshot, and the
         *     attempt the study's Temporal workflow is executing on any branch, if any.
         */
        get: operations["get_study_api_studies__workspace_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model": {
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
         *     Omit `at` for the selected branch head, or pass an exact Git commit ID.
         *     Use `context.commit_id` to pin subsequent reads. Failed attempts retain logs without advancing scientific state.
         */
        get: operations["get_model_snapshot_api_studies__workspace_id__model_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model-diff": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
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
        get: operations["model_diff"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/data-diff": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
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
        post: operations["data_diff"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/data-diff/{commit_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Data Diff
         * @description Read a comparison's saved evidence; timeline records contain no report payload.
         */
        get: operations["get_data_diff_api_studies__workspace_id__data_diff__commit_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/definition": {
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
        get: operations["get_model_definition_api_studies__workspace_id__model_definition_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/inference-report": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Inference Report
         * @description Read the fit report associated with the selected model revision.
         */
        get: operations["get_model_inference_report_api_studies__workspace_id__model_inference_report_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/visuals/observations/{indicator_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Observation History
         * @description All prepared observations on their recorded temporal support.
         */
        get: operations["get_observation_history_api_studies__workspace_id__model_visuals_observations__indicator_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/visuals/predictive/{indicator_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Predictive History
         * @description Saved predictive paths on the exact schedule of their pinned inputs.
         */
        get: operations["get_predictive_history_api_studies__workspace_id__model_visuals_predictive__indicator_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/visuals/simulation": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Simulation Paths
         * @description A contiguous page of original simulation draws, without time thinning.
         */
        get: operations["get_simulation_paths_api_studies__workspace_id__model_visuals_simulation_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/visuals/parameters": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Parameter Draws
         * @description All coordinates and all draws of the retained joint posterior.
         */
        get: operations["get_parameter_draws_api_studies__workspace_id__model_visuals_parameters_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/visuals/mechanism": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Get Mechanism Curves
         * @description Read conditional drift curves using the exact model equations; creates no scientific action.
         */
        post: operations["get_mechanism_curves_api_studies__workspace_id__model_visuals_mechanism_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/constructs": {
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
        get: operations["get_model_constructs_api_studies__workspace_id__model_constructs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/edges": {
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
        get: operations["get_model_edges_api_studies__workspace_id__model_edges_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/indicators": {
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
        get: operations["get_model_indicators_api_studies__workspace_id__model_indicators_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/parameters": {
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
        get: operations["get_model_parameters_api_studies__workspace_id__model_parameters_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/model/views/{artifact_id}": {
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
        get: operations["get_model_view_api_studies__workspace_id__model_views__artifact_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/timeline": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
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
        get: operations["get_timeline_api_studies__workspace_id__timeline_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/studies/{workspace_id}/events": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Events
         * @description Live progress of one attempt: data-preparation step status and extraction fan-out.
         *
         *     Pass the last-seen event cursor as `after` to page forward. Progress is disposable
         *     and never saved with the attempt; its record and traces are authoritative.
         */
        get: operations["get_events_api_studies__workspace_id__events_get"];
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
        ActionId: Domain.ActionId;
        /**
         * ActionMessage
         * @description One label emitted by an action; scientific measurements belong in its body.
         */
        ActionMessage: Domain.ActionMessage;
        /**
         * ActionReceipt
         * @description A durable dispatch acknowledgment, without a scientific result body.
         */
        ActionReceipt: Domain.ActionReceipt;
        /** @enum {string} */
        AggregationFunction: Domain.AggregationFunction;
        /** ArtifactFreshness */
        ArtifactFreshness: Domain.ArtifactFreshness;
        /** @enum {string} */
        ArtifactId: Domain.ArtifactId;
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
        ArtifactRecord: Domain.ArtifactRecord;
        /**
         * ArtifactViewResponse
         * @description One available artifact projection returned by the model view endpoint.
         */
        ArtifactViewResponse: Domain.ArtifactViewResponse;
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        "BinaryExpression-Output": Domain.BinaryExpression;
        /** @enum {string} */
        BinaryOperator: Domain.BinaryOperator;
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        "CallExpression-Output": Domain.CallExpression;
        /**
         * CategoryProbabilitySummary
         * @description Predictive probabilities for each declared level; unobserved anchors are null.
         */
        CategoryProbabilitySummary: Domain.CategoryProbabilitySummary;
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        "CausalEdgeSpec-Output": Domain.CausalEdgeSpec;
        /**
         * CausalEffectResult
         * @description Causal effects and realized trajectories under the enclosing report's design.
         */
        CausalEffectResult: Domain.CausalEffectResult;
        /** @enum {string} */
        CheckGroup: Domain.CheckGroup;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        CoefficientExpression: Domain.CoefficientExpression;
        /** @enum {string} */
        CoefficientRole: Domain.CoefficientRole;
        /**
         * ComparisonConnection
         * @description Endpoint references and description for one side of a causal edge comparison.
         */
        ComparisonConnection: Domain.ComparisonConnection;
        /**
         * ConstructComparison
         * @description A construct's presence and time-slice topology in two model revisions.
         */
        ConstructComparison: Domain.ConstructComparison;
        ConstructId: string;
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        ConstructRef: Domain.ConstructRef;
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        "ConstructSpec-Output": Domain.ConstructSpec;
        /**
         * DataDiffReport
         * @description Comparisons of existing data, preserving each history's immutable source reference.
         */
        DataDiffReport: Domain.DataDiffReport;
        /**
         * DataDiffRequest
         * @description Compare two immutable data selections, each containing one or more histories.
         */
        DataDiffRequest: {
            /**
             * Action
             * @default data_diff
             * @constant
             */
            action?: "data_diff";
            left: components["schemas"]["DataSelection"];
            right: components["schemas"]["DataSelection"];
        };
        /**
         * DataPoint
         * @description An observed anchor and support; dates are synthetic for a calendar-free series.
         */
        DataPoint: Domain.DataPoint;
        /**
         * DataPointChange
         * @description An added, removed or revised measurement in a single-history comparison.
         */
        DataPointChange: Domain.DataPointChange;
        /**
         * DataPreparationSpec
         * @description A versioned data definition supplied directly to prepare_data.
         */
        DataPreparationSpec: Domain.DataPreparationSpec;
        /**
         * DataProfileArtifact
         * @description Model-independent empirical measurements and data-quality findings.
         */
        DataProfileArtifact: Domain.DataProfileArtifact;
        /**
         * DataRef
         * @description A saved panel or simulation; omitting replicate selects every saved simulation draw.
         */
        DataRef: {
            /**
             * Kind
             * @enum {string}
             */
            kind: "panel" | "simulation";
            revision: components["schemas"]["GitOid"];
            /** Replicate */
            replicate?: number | null;
        };
        /** @description A data selection identifies one or more saved observation histories. */
        DataSelection: components["schemas"]["DataRef"] | components["schemas"]["DataRef"][];
        /**
         * DataSeries
         * @description One variable's recorded measurements in one history; no pooling across replicas.
         */
        DataSeries: Domain.DataSeries;
        /** @description Uploaded sources or one recorded simulation replicate. */
        DataSourceRef: Domain.DataSourceRef;
        /**
         * DataStatisticComparison
         * @description The same descriptive statistic measured independently in every selected history.
         */
        DataStatisticComparison: Domain.DataStatisticComparison;
        /**
         * DataVariableDiff
         * @description Definitions, histories and comparisons for one persistent observation identity.
         */
        DataVariableDiff: Domain.DataVariableDiff;
        /**
         * DataVariableSpec
         * @description How to produce one observed variable, without any causal or latent model.
         */
        DataVariableSpec: Domain.DataVariableSpec;
        /**
         * DensityPoint
         * @description A plotting coordinate evaluated from the native prior's log density.
         */
        DensityPoint: Domain.DensityPoint;
        /** @description A native law whose membership is defined by the model's scientific quantities. */
        DistributionId: Domain.DistributionId;
        /**
         * DynamicsMechanismSpec
         * @description A symbolic specification of an additive drift term or a node potential.
         *
         *     A node potential contributes its negative gradient to the drift.
         */
        "DynamicsMechanismSpec-Output": Domain.DynamicsMechanismSpec;
        /**
         * EdgeComparison
         * @description An explicit causal edge's presence and endpoints in two model revisions.
         */
        EdgeComparison: Domain.EdgeComparison;
        EdgeId: Domain.EdgeId;
        /**
         * EdgeRef
         * @description An edge reference identifies a causal relationship independently of edits to its
         *     definition.
         */
        EdgeRef: Domain.EdgeRef;
        /**
         * EffectSummary
         * @description An effect summary reports posterior location, uncertainty, and sign probability.
         */
        EffectSummary: Domain.EffectSummary;
        /**
         * EffectTrajectoryPoint
         * @description A certified paired contrast and 95% interval at one absolute model time.
         */
        EffectTrajectoryPoint: Domain.EffectTrajectoryPoint;
        /** EmpiricalPoint */
        EmpiricalPoint: Domain.EmpiricalPoint;
        /**
         * EventsResponse
         * @description An events response pages one running attempt's live progress.
         */
        EventsResponse: Domain.EventsResponse;
        "Expression-Output": Domain.Expression;
        /** @enum {string} */
        ExpressionFunction: Domain.ExpressionFunction;
        /**
         * ExtractionPlanEvent
         * @description The extraction fan-out plan.
         */
        ExtractionPlanEvent: Domain.ExtractionPlanEvent;
        /**
         * ExtractionSnapshotEvent
         * @description Aggregate extraction worker counts.
         */
        ExtractionSnapshotEvent: Domain.ExtractionSnapshotEvent;
        /**
         * ExtractionWorkerEvent
         * @description One extraction worker's state; a worker reports its LLM calls when it finishes.
         */
        ExtractionWorkerEvent: Domain.ExtractionWorkerEvent;
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact revision and records its freshness.
         */
        FactSource: Domain.FactSource;
        /**
         * FileSourceRef
         * @description Explicit uploaded filenames, relative to this study's input directory.
         */
        FileSourceRef: Domain.FileSourceRef;
        /** @enum {string} */
        FitReliability: Domain.FitReliability;
        /**
         * FitSummary
         * @description A fit read contains the inference report summary and server-composed display findings.
         *
         *     Per-draw diagnostics load separately from the inference report endpoint.
         */
        FitSummary: Domain.FitSummary;
        GitOid: string;
        /**
         * GitRef
         * @description An exact file in a study's Git object database: repository, object, and path.
         */
        GitRef: Domain.GitRef;
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
        IndicatorId: Domain.IndicatorId;
        /**
         * IndicatorPolarity
         * @description Indicator polarity states whether a measurement increases or decreases with its
         *     construct.
         * @enum {string}
         */
        IndicatorPolarity: Domain.IndicatorPolarity;
        /**
         * IndicatorRef
         * @description An indicator reference identifies a measurement definition independently of its name or
         *     revision.
         */
        IndicatorRef: Domain.IndicatorRef;
        /**
         * IndicatorSpec
         * @description Bind an observed-variable ID to a construct and an emission likelihood.
         *
         *     Extraction instructions belong to DataPreparationSpec. The shared observation
         *     schema also permits generative models before any observations have been collected.
         */
        "IndicatorSpec-Output": Domain.IndicatorSpec;
        /**
         * InferenceMetadata
         * @description Inference metadata records the sampling method, sample count, and run duration.
         */
        InferenceMetadata: Domain.InferenceMetadata;
        /**
         * InferenceReport
         * @description Display findings recorded by a fit, separate from ModelSpec.
         */
        InferenceReport: Domain.InferenceReport;
        /**
         * InterventionSpec
         * @description Set a latent state at one model time, then let its dynamics resume.
         */
        InterventionSpec: Domain.InterventionSpec;
        /** @enum {string} */
        JournalStatus: Domain.JournalStatus;
        "JsonArray-Output": Domain.JsonArray;
        "JsonObject-Output": Domain.JsonObject;
        JsonScalar: Domain.JsonScalar;
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
        "LikelihoodSpec-Output": Domain.LikelihoodSpec;
        /**
         * LiteralExpression
         * @description A finite scalar constant in a model equation.
         */
        LiteralExpression: Domain.LiteralExpression;
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        LiteratureSource: Domain.LiteratureSource;
        /** @enum {string} */
        MeasurementDtype: Domain.MeasurementDtype;
        /**
         * MeasurementsData
         * @description Counts and representative observations read directly from one panel revision.
         */
        MeasurementsData: Domain.MeasurementsData;
        /**
         * MechanismCurves
         * @description Exact conditional drift contributions, not marginal or total causal effects.
         */
        MechanismCurves: Domain.MechanismCurves;
        MechanismId: Domain.MechanismId;
        /** MechanismViewRequest */
        MechanismViewRequest: {
            /** Owner Id */
            owner_id: string;
            axis?: components["schemas"]["ConstructId"] | null;
            /**
             * Lower
             * @default -3
             */
            lower?: number;
            /**
             * Upper
             * @default 3
             */
            upper?: number;
            /** Held */
            held?: {
                [key: string]: number;
            };
            moderator?: components["schemas"]["ConstructId"] | null;
            /**
             * Levels
             * @default [
             *       -1,
             *       0,
             *       1
             *     ]
             */
            levels?: number[];
            /**
             * Start
             * @default 0
             */
            start?: number;
            /**
             * Count
             * @default 24
             */
            count?: number;
            /**
             * Points
             * @default 201
             */
            points?: number;
        };
        /**
         * ModelCheckReport
         * @description Checks selected by their consumed inputs, retained with the study snapshot.
         */
        ModelCheckReport: Domain.ModelCheckReport;
        /**
         * ModelData
         * @description Observed evidence paired with its source versions.
         */
        ModelData: Domain.ModelData;
        /**
         * ModelDefinitionChange
         * @description One changed field in identity-keyed scientific model definitions.
         */
        ModelDefinitionChange: Domain.ModelDefinitionChange;
        /**
         * ModelDiagnostics
         * @description Server-derived equations and comparisons with pinned observations.
         */
        ModelDiagnostics: Domain.ModelDiagnostics;
        /**
         * ModelDiffReport
         * @description A model diff joins definition changes and evidence at two model revisions or checkpoints.
         */
        ModelDiffReport: Domain.ModelDiffReport;
        /**
         * ModelFindings
         * @description ModelSpec findings collect identification, validation, and fitted results with their input references.
         */
        ModelFindings: Domain.ModelFindings;
        /**
         * ModelGraphComparison
         * @description Identity-aligned topology changes, excluding laws and other entity attributes.
         */
        ModelGraphComparison: Domain.ModelGraphComparison;
        /**
         * ModelGraphView
         * @description Scientific entity identities selected for the graph at this authoring checkpoint.
         */
        ModelGraphView: Domain.ModelGraphView;
        /**
         * ModelPredictiveReport
         * @description One automatic, reproducible battery over the full model's current laws.
         */
        ModelPredictiveReport: Domain.ModelPredictiveReport;
        /**
         * ModelSnapshot
         * @description The canonical scientific definition with independently sourced inputs and findings.
         */
        ModelSnapshot: Domain.ModelSnapshot;
        /**
         * ModelSpec
         * @description An evolving research question and connected causal graph with owned scientific detail.
         */
        "ModelSpec-Output": Domain.ModelSpec;
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        NonIdentifiableTreatmentStatus: Domain.NonIdentifiableTreatmentStatus;
        "NumPyroDistribution-Output": Domain.NumPyroDistribution;
        /**
         * ObservationHistory
         * @description All prepared observations, their true anchors and their measurement support.
         */
        ObservationHistory: Domain.ObservationHistory;
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
         * ObservationSpec
         * @description A stable observed variable, reusable across scientific model definitions.
         */
        ObservationSpec: Domain.ObservationSpec;
        /**
         * PPCOverlay
         * @description A predictive overlay sets one indicator's observed values against simulated ones.
         *
         *     It carries the predictive median and a few individual replicated series, the
         *     spaghetti plot of a visual predictive check.
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
        /**
         * ParameterChange
         * @description A parameter change compares one parameter's law across model revisions.
         */
        ParameterChange: Domain.ParameterChange;
        /**
         * ParameterConvergenceFailure
         * @description A failed criterion on one retained scalar parameter element.
         */
        ParameterConvergenceFailure: Domain.ParameterConvergenceFailure;
        /**
         * ParameterConvergenceReport
         * @description Recorded-chain checks cover parameters, not latent-path mixing.
         */
        ParameterConvergenceReport: Domain.ParameterConvergenceReport;
        /** ParameterDrawColumn */
        ParameterDrawColumn: Domain.ParameterDrawColumn;
        /**
         * ParameterDraws
         * @description Every retained parameter coordinate, without thinning or pair selection.
         */
        ParameterDraws: Domain.ParameterDraws;
        ParameterElementId: Domain.ParameterElementId;
        ParameterId: Domain.ParameterId;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        ParameterRef: Domain.ParameterRef;
        /**
         * ParameterSpec
         * @description A named uncertain quantity; fixed coefficients are literals in component slots.
         */
        ParameterSpec: Domain.ParameterSpec;
        /** PathSeries */
        PathSeries: Domain.PathSeries;
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
         * PredictiveCheckFinding
         * @description One measured simulation check, independent of its authoring or simulation context.
         */
        PredictiveCheckFinding: Domain.PredictiveCheckFinding;
        /** @enum {string} */
        PredictiveCheckReason: Domain.PredictiveCheckReason;
        /**
         * PredictiveHistory
         * @description A saved check on the exact schedule and scale used to evaluate it.
         */
        PredictiveHistory: Domain.PredictiveHistory;
        /**
         * PredictiveLawProvenance
         * @description Known conditioning history, independently of a probability law's family.
         */
        PredictiveLawProvenance: Domain.PredictiveLawProvenance;
        PredictiveSummary: Domain.PredictiveSummary;
        /**
         * PreparedDataMetadata
         * @description Self-contained semantics and provenance of one prepared observation table.
         */
        PreparedDataMetadata: Domain.PreparedDataMetadata;
        ProgressEvent: Domain.ProgressEvent;
        /** @enum {string} */
        ProgressStep: Domain.ProgressStep;
        /**
         * RawDataColumnDescription
         * @description A stored column's physical type and authored interpretation.
         */
        RawDataColumnDescription: Domain.RawDataColumnDescription;
        /**
         * RawDataData
         * @description Profile and representative rows from one uploaded table revision.
         */
        RawDataData: Domain.RawDataData;
        /**
         * RawDataDateRange
         * @description Observed date bounds of the uploaded table, when it contains a date column.
         */
        RawDataDateRange: Domain.RawDataDateRange;
        /**
         * RecordDependency
         * @description A journal record used an output of an earlier record.
         */
        RecordDependency: Domain.RecordDependency;
        /** RecordedPath */
        RecordedPath: Domain.RecordedPath;
        /** ResponseCurve */
        ResponseCurve: Domain.ResponseCurve;
        /**
         * RetractedArtifact
         * @description A current artifact removed by an action, with the finding that caused it.
         */
        RetractedArtifact: Domain.RetractedArtifact;
        /**
         * Role
         * @description A construct role states whether the variable is modeled as endogenous or treated as
         *     exogenous.
         * @enum {string}
         */
        Role: Domain.Role;
        /**
         * RunningAction
         * @description The attempt a study is executing, with the labels it has emitted so far.
         */
        RunningAction: Domain.RunningAction;
        /** @enum {string} */
        ScientificActionId: Domain.ScientificActionId;
        /**
         * SimulationObservationLayout
         * @description Saved observation semantics and coordinates; generation truths remain separate.
         */
        SimulationObservationLayout: Domain.SimulationObservationLayout;
        /**
         * SimulationPaths
         * @description Contiguous pages of original draws, with every recorded time point intact.
         */
        SimulationPaths: Domain.SimulationPaths;
        /**
         * SimulationPredictiveReport
         * @description Model implications, independently of whether a causal contrast is certified.
         */
        SimulationPredictiveReport: Domain.SimulationPredictiveReport;
        /**
         * SimulationReplicateRef
         * @description One replicate from a recorded, applied simulation in this study.
         */
        SimulationReplicateRef: Domain.SimulationReplicateRef;
        /**
         * SimulationReport
         * @description Generated histories and derived findings with their resolved execution coordinates.
         */
        SimulationReport: Domain.SimulationReport;
        /**
         * SimulationSeriesSummary
         * @description One state's or indicator's generated distribution in each simulated arm.
         */
        SimulationSeriesSummary: Domain.SimulationSeriesSummary;
        /**
         * SimulationSpec
         * @description Generate through end, optionally starting earlier and applying dated interventions.
         */
        SimulationSpec: Domain.SimulationSpec;
        /**
         * SnapshotContext
         * @description A snapshot context identifies the selected Git commit and its artifact versions.
         */
        SnapshotContext: Domain.SnapshotContext;
        /**
         * SnapshotState
         * @description A snapshot state lists the artifact revisions current at the selected commit.
         *
         *     Recorded checks appear once, as the specification and predictive findings.
         */
        SnapshotState: Domain.SnapshotState;
        /**
         * SourceValidity
         * @description Source validity records whether a fact still matches its pinned inputs.
         * @enum {string}
         */
        SourceValidity: Domain.SourceValidity;
        /** Sourced[DataProfileArtifact] */
        Sourced_DataProfileArtifact_: {
            value: components["schemas"]["DataProfileArtifact"];
            source: components["schemas"]["FactSource"];
        };
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
        /** Sourced[ModelPredictiveReport] */
        Sourced_ModelPredictiveReport_: {
            value: components["schemas"]["ModelPredictiveReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[ModelSpec] */
        Sourced_ModelSpec_: {
            value: components["schemas"]["ModelSpec-Output"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[PreparedDataMetadata] */
        Sourced_PreparedDataMetadata_: {
            value: components["schemas"]["PreparedDataMetadata"];
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
         * @description Model-only findings; data compatibility has its own paired input references.
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
        StateExpression: Domain.StateExpression;
        /**
         * StepError
         * @description The error type and message of a failed step.
         */
        StepError: Domain.StepError;
        /**
         * StepEvent
         * @description A data-preparation step changed status.
         */
        StepEvent: Domain.StepEvent;
        /** @enum {string} */
        StepStatus: Domain.StepStatus;
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
         * StudyRevision
         * @description One Git commit's parent links and its action log.
         */
        StudyRevision: Domain.StudyRevision;
        /**
         * StudyState
         * @description Study state projects the artifact trees selected by one Git commit.
         *
         *     ``current`` maps artifact id → the revision info that is *current* for the
         *     study. Absent key = the artifact does not exist (either never produced,
         *     or produced-when-nonempty semantics withheld it).
         */
        StudyState: Domain.StudyState;
        /**
         * StudyStatus
         * @description Study status reports committed artifacts, their freshness, actions, and any running one.
         */
        StudyStatus: Domain.StudyStatus;
        /**
         * TemporalStatus
         * @description Temporal status states whether a construct varies within the individual over time.
         * @enum {string}
         */
        TemporalStatus: Domain.TemporalStatus;
        /**
         * TimelineResponse
         * @description Typed attempt journal returned by the study read plane.
         */
        TimelineResponse: Domain.TimelineResponse;
        /**
         * TrajectorySummary
         * @description Pointwise means and fixed 95% quantiles across generated numeric draws.
         */
        TrajectorySummary: Domain.TrajectorySummary;
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
         * @description Measurement findings augmented with model-dependent execution checks.
         */
        ValidationReportArtifact: Domain.ValidationReportArtifact;
        /** @description Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values. */
        WindowExpression: Domain.WindowExpression;
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    get_study_api_studies__workspace_id__get: {
        parameters: {
            query?: {
                branch?: string;
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
                    "application/json": components["schemas"]["StudyStatus"];
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
    get_model_snapshot_api_studies__workspace_id__model_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    model_diff: {
        parameters: {
            query: {
                before: components["schemas"]["GitOid"];
                after: components["schemas"]["GitOid"];
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
                    "application/json": components["schemas"]["ModelDiffReport"];
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
    data_diff: {
        parameters: {
            query?: {
                branch?: string;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["DataDiffRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ActionReceipt"];
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
    get_data_diff_api_studies__workspace_id__data_diff__commit_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                workspace_id: string;
                commit_id: components["schemas"]["GitOid"];
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
                    "application/json": components["schemas"]["DataDiffReport"];
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
    get_model_definition_api_studies__workspace_id__model_definition_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_model_inference_report_api_studies__workspace_id__model_inference_report_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_observation_history_api_studies__workspace_id__model_visuals_observations__indicator_id__get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
            };
            header?: never;
            path: {
                indicator_id: components["schemas"]["IndicatorId"];
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
                    "application/json": components["schemas"]["ObservationHistory"] | null;
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
    get_predictive_history_api_studies__workspace_id__model_visuals_predictive__indicator_id__get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
            };
            header?: never;
            path: {
                indicator_id: components["schemas"]["IndicatorId"];
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
                    "application/json": components["schemas"]["PredictiveHistory"] | null;
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
    get_simulation_paths_api_studies__workspace_id__model_visuals_simulation_get: {
        parameters: {
            query?: {
                start?: number;
                count?: number;
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
                    "application/json": components["schemas"]["SimulationPaths"] | null;
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
    get_parameter_draws_api_studies__workspace_id__model_visuals_parameters_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
                    "application/json": components["schemas"]["ParameterDraws"];
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
    get_mechanism_curves_api_studies__workspace_id__model_visuals_mechanism_post: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
            };
            header?: never;
            path: {
                workspace_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MechanismViewRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MechanismCurves"];
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
    get_model_constructs_api_studies__workspace_id__model_constructs_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_model_edges_api_studies__workspace_id__model_edges_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_model_indicators_api_studies__workspace_id__model_indicators_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_model_parameters_api_studies__workspace_id__model_parameters_get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_model_view_api_studies__workspace_id__model_views__artifact_id__get: {
        parameters: {
            query?: {
                branch?: string;
                at?: components["schemas"]["GitOid"] | null;
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
    get_timeline_api_studies__workspace_id__timeline_get: {
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
                    "application/json": components["schemas"]["TimelineResponse"];
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
    get_events_api_studies__workspace_id__events_get: {
        parameters: {
            query: {
                attempt_id: string;
                after?: string | null;
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
                    "application/json": components["schemas"]["EventsResponse"];
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
