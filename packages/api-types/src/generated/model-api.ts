/** AUTO-GENERATED from FastAPI OpenAPI. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
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
        AggregationFunction: Domain.AggregationFunction;
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
        "BinaryExpression-Input": Domain.BinaryExpression;
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
        "CallExpression-Input": Domain.CallExpression;
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        "CallExpression-Output": Domain.CallExpression;
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        "CausalEdgeSpec-Input": Domain.CausalEdgeSpec;
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        "CausalEdgeSpec-Output": Domain.CausalEdgeSpec;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        CoefficientExpression: Domain.CoefficientExpression;
        /** @enum {string} */
        CoefficientRole: Domain.CoefficientRole;
        ConstructId: Domain.ConstructId;
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
        "ConstructSpec-Input": Domain.ConstructSpec;
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        "ConstructSpec-Output": Domain.ConstructSpec;
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
        "DynamicsMechanismSpec-Input": Domain.DynamicsMechanismSpec;
        /**
         * DynamicsMechanismSpec
         * @description A symbolic specification of an additive drift term or a node potential.
         *
         *     A node potential contributes its negative gradient to the drift.
         */
        "DynamicsMechanismSpec-Output": Domain.DynamicsMechanismSpec;
        EdgeId: Domain.EdgeId;
        /**
         * EdgeRef
         * @description An edge reference identifies a causal relationship independently of edits to its
         *     definition.
         */
        EdgeRef: Domain.EdgeRef;
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
        "Expression-Input": Domain.Expression;
        "Expression-Output": Domain.Expression;
        /** @enum {string} */
        ExpressionFunction: Domain.ExpressionFunction;
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact version and records its freshness.
         */
        FactSource: Domain.FactSource;
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
         * @description A specification of a construct's observed measurement and how to extract it.
         */
        "IndicatorSpec-Input": Domain.IndicatorSpec;
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
        "JsonArray-Input": Domain.JsonArray;
        "JsonArray-Output": Domain.JsonArray;
        "JsonObject-Input": Domain.JsonObject;
        "JsonObject-Output": Domain.JsonObject;
        JsonScalar: Domain.JsonScalar;
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
        "LikelihoodSpec-Input": Domain.LikelihoodSpec;
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
         * @description Counts and representative observations read directly from one panel version.
         */
        MeasurementsData: Domain.MeasurementsData;
        MechanismId: Domain.MechanismId;
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
         * ModelSnapshot
         * @description The canonical scientific definition with independently sourced inputs and findings.
         */
        ModelSnapshot: Domain.ModelSnapshot;
        /**
         * ModelSpec
         * @description An evolving research question and connected causal graph with owned scientific detail.
         */
        "ModelSpec-Input": Domain.ModelSpec;
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
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        NonIdentifiableTreatmentStatus: Domain.NonIdentifiableTreatmentStatus;
        "NumPyroDistribution-Input": Domain.NumPyroDistribution;
        "NumPyroDistribution-Output": Domain.NumPyroDistribution;
        /**
         * ObservationLawSpec
         * @description A symbolic specification of an indicator's conditional observation distribution.
         */
        "ObservationLawSpec-Input": Domain.ObservationLawSpec;
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
        ParameterElementId: Domain.ParameterElementId;
        ParameterId: Domain.ParameterId;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        ParameterRef: Domain.ParameterRef;
        /**
         * ParameterSpec
         * @description A named quantity's current uncertainty; component slots define its meaning.
         */
        ParameterSpec: Domain.ParameterSpec;
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
         * Role
         * @description A construct role states whether the variable is modeled as endogenous or treated as
         *     exogenous.
         * @enum {string}
         */
        Role: Domain.Role;
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
        TemporalStatus: Domain.TemporalStatus;
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
