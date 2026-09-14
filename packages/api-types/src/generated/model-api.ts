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
        /**
         * AdmissionReport
         * @description Prior research and admission findings pinned to the model that was checked.
         */
        AdmissionReport: Domain.AdmissionReport;
        /** @enum {string} */
        AggregationFunction: Domain.AggregationFunction;
        /**
         * AnchorCertificate
         * @description Compiler proof that one retained latent has location and scale anchors.
         */
        AnchorCertificate: Domain.AnchorCertificate;
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
         * BaselineReportArtifact
         * @description A baseline report collects treatment effects and explicitly retained simulations from the fitted model.
         */
        BaselineReportArtifact: Domain.BaselineReportArtifact;
        /** BaselineReportVisualization */
        BaselineReportVisualization: Domain.BaselineReportVisualization;
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
         * CausalEdge
         * @description A causal edge declares a directed causal relationship between two constructs.
         */
        "CausalEdge-Input": Domain.CausalEdge;
        /**
         * CausalEdge
         * @description A causal edge declares a directed causal relationship between two constructs.
         */
        "CausalEdge-Output": Domain.CausalEdge;
        "Coefficient-Input": Domain.Coefficient;
        "Coefficient-Output": Domain.Coefficient;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        "CoefficientExpression-Input": Domain.CoefficientExpression;
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        "CoefficientExpression-Output": Domain.CoefficientExpression;
        /** @enum {string} */
        CoefficientRole: Domain.CoefficientRole;
        /**
         * Construct
         * @description A construct represents a theoretical entity in the scientific causal model.
         */
        "Construct-Input": Domain.Construct;
        /**
         * Construct
         * @description A construct represents a theoretical entity in the scientific causal model.
         */
        "Construct-Output": Domain.Construct;
        ConstructId: Domain.ConstructId;
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        ConstructRef: Domain.ConstructRef;
        "ConstructUsage-Input": Domain.ConstructUsage;
        "ConstructUsage-Output": Domain.ConstructUsage;
        /**
         * DensityPoint
         * @description A plotting coordinate evaluated from the native prior's log density.
         */
        DensityPoint: Domain.DensityPoint;
        /** @description A shared native law whose membership is defined by the model's scientific quantities. */
        DistributionId: Domain.DistributionId;
        /**
         * DynamicsMechanism
         * @description An additive drift term or node potential whose negative gradient enters the drift.
         */
        "DynamicsMechanism-Input": Domain.DynamicsMechanism;
        /**
         * DynamicsMechanism
         * @description An additive drift term or node potential whose negative gradient enters the drift.
         */
        "DynamicsMechanism-Output": Domain.DynamicsMechanism;
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
         * @description An effect trajectory point records a causal delta at one elapsed rollout time.
         */
        EffectTrajectoryPoint: Domain.EffectTrajectoryPoint;
        EntityRef: Domain.EntityRef;
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
        /**
         * ExecutionReadiness
         * @description Current model requirements and latent anchors, computed without a stored receipt.
         */
        ExecutionReadiness: Domain.ExecutionReadiness;
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
        /**
         * FixedCoefficient
         * @description A coefficient held at a specified value on the continuous-time model scale.
         */
        FixedCoefficient: Domain.FixedCoefficient;
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
         * IdentifiabilityStatus
         * @description Status of causal effect identifiability.
         */
        IdentifiabilityStatus: Domain.IdentifiabilityStatus;
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
         * Indicator
         * @description An indicator defines an observed measurement of a construct and how to extract it.
         */
        "Indicator-Input": Domain.Indicator;
        /**
         * Indicator
         * @description An indicator defines an observed measurement of a construct and how to extract it.
         */
        "Indicator-Output": Domain.Indicator;
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
         * IndicatorValidation
         * @description Indicator validation records the outcomes and issues from checks on one extracted
         *     indicator.
         */
        IndicatorValidation: Domain.IndicatorValidation;
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
        /**
         * InitialStateSpec
         * @description Initial location, marginal scale and correlations, including shared baseline factors.
         */
        "InitialStateSpec-Input": Domain.InitialStateSpec;
        /**
         * InitialStateSpec
         * @description Initial location, marginal scale and correlations, including shared baseline factors.
         */
        "InitialStateSpec-Output": Domain.InitialStateSpec;
        /**
         * InnovationSpec
         * @description Continuous-time driving noise, including conditional loadings from common causes.
         */
        "InnovationSpec-Input": Domain.InnovationSpec;
        /**
         * InnovationSpec
         * @description Continuous-time driving noise, including conditional loadings from common causes.
         */
        "InnovationSpec-Output": Domain.InnovationSpec;
        "JsonArray-Input": Domain.JsonArray;
        "JsonArray-Output": Domain.JsonArray;
        "JsonObject-Input": Domain.JsonObject;
        "JsonObject-Output": Domain.JsonObject;
        JsonScalar: Domain.JsonScalar;
        "JsonValue-Input": Domain.JsonValue;
        "JsonValue-Output": Domain.JsonValue;
        /**
         * KnownInput
         * @description An observed-input declaration binds a construct to its measured driver trajectory.
         */
        KnownInput: Domain.KnownInput;
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
         * MechanismRef
         * @description A particular additive term, independently of its position or coefficient values.
         */
        MechanismRef: Domain.MechanismRef;
        /**
         * ModelData
         * @description ModelSpec data pairs the causal question and observed evidence with their source versions.
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
         * ModelRef
         * @description A model reference identifies the workspace that owns the scientific model.
         */
        ModelRef: Domain.ModelRef;
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
         * @description One connected causal graph whose endpoints and relationships gain scientific detail.
         */
        "ModelSpec-Input": Domain.ModelSpec;
        /**
         * ModelSpec
         * @description One connected causal graph whose endpoints and relationships gain scientific detail.
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
         * ObservationLaw
         * @description A native probability constructor applied to model-dependent expressions.
         */
        "ObservationLaw-Input": Domain.ObservationLaw;
        /**
         * ObservationLaw
         * @description A native probability constructor applied to model-dependent expressions.
         */
        "ObservationLaw-Output": Domain.ObservationLaw;
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
        /**
         * ParameterCoefficient
         * @description A slot referencing a scientific parameter, whether fixed or estimated.
         */
        ParameterCoefficient: Domain.ParameterCoefficient;
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
        "ParameterSpec-Input": Domain.ParameterSpec;
        /**
         * ParameterSpec
         * @description A named quantity's current uncertainty; component slots define its meaning.
         */
        "ParameterSpec-Output": Domain.ParameterSpec;
        /**
         * PosteriorAssessment
         * @description Predictive assessments of a fitted posterior.
         */
        PosteriorAssessment: Domain.PosteriorAssessment;
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
         * @description A prior predictive diagnostic records the result of one exact model-admission check.
         */
        PriorPredictiveDiagnostic: Domain.PriorPredictiveDiagnostic;
        /** @enum {string} */
        Provenance: Domain.Provenance;
        /**
         * QuestionArtifact
         * @description A research question states the observational causal question under investigation.
         */
        QuestionArtifact: Domain.QuestionArtifact;
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
         * ScenarioClamp
         * @description A do-operator on one latent variable over a time window.
         *
         *     The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
         *     the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
         *     value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
         *     interpolates across the window, and ``trajectory`` tracks a list of values across it.
         */
        ScenarioClamp: Domain.ScenarioClamp;
        /** ScenarioQueryInput */
        ScenarioQueryInput: Domain.ScenarioQueryInput;
        /**
         * ScenarioRequest
         * @description One reusable request for an on-demand simulation of a fitted model.
         */
        ScenarioRequest: Domain.ScenarioRequest;
        /**
         * ScenarioStartInput
         * @description Where the forward rollout begins (replaces the rung-2/rung-3 split).
         */
        ScenarioStartInput: Domain.ScenarioStartInput;
        /**
         * ScientificOnlyConstruct
         * @description A scientific-only declaration excludes an identified construct from executable states.
         */
        ScientificOnlyConstruct: Domain.ScientificOnlyConstruct;
        /**
         * SimulationProvenance
         * @description The retained fit and actual numerical settings used by this response.
         */
        SimulationProvenance: Domain.SimulationProvenance;
        /**
         * SimulationResult
         * @description Ephemeral response, retained only when explicitly included in a report.
         *
         *     This engine integrates the true nonlinear drift for each posterior draw.
         *     It does not include future process noise or claim the mean of the SDE.
         */
        SimulationResult: Domain.SimulationResult;
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
        /** Sourced[AdmissionReport] */
        Sourced_AdmissionReport_: {
            value: components["schemas"]["AdmissionReport"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[BaselineReportArtifact] */
        Sourced_BaselineReportArtifact_: {
            value: components["schemas"]["BaselineReportArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[ExecutionReadiness] */
        Sourced_ExecutionReadiness_: {
            value: components["schemas"]["ExecutionReadiness"];
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
        /** Sourced[ModelSpec] */
        Sourced_ModelSpec_: {
            value: components["schemas"]["ModelSpec-Output"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[QuestionArtifact] */
        Sourced_QuestionArtifact_: {
            value: components["schemas"]["QuestionArtifact"];
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
         * StateCoupling
         * @description A coefficient connecting an owned component to another construct.
         */
        "StateCoupling-Input": Domain.StateCoupling;
        /**
         * StateCoupling
         * @description A coefficient connecting an owned component to another construct.
         */
        "StateCoupling-Output": Domain.StateCoupling;
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
         * TemporalEffect
         * @description A temporal effect summarizes a trajectory at requested horizons and its absolute peak.
         */
        TemporalEffect: Domain.TemporalEffect;
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
        /**
         * TreatmentEffect
         * @description A treatment effect stores posterior effect draws and optional temporal or observed-scale
         *     summaries.
         */
        TreatmentEffect: Domain.TreatmentEffect;
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
                    "application/json": components["schemas"]["Construct-Output"][];
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
                    "application/json": components["schemas"]["CausalEdge-Output"][];
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
                    "application/json": components["schemas"]["Indicator-Output"][];
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
                    "application/json": components["schemas"]["ParameterSpec-Output"][];
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
