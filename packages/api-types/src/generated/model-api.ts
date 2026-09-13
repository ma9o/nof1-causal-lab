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
         *     Use the returned `seq` for subsequent aggregate or collection reads at the same revision.
         */
        get: operations["get_model_snapshot_api_episodes__workspace_id__model_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/latent-structure": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Latent Structure
         * @description Canonical latent structure, including its default outcome, at the selected revision.
         */
        get: operations["get_model_latent_structure_api_episodes__workspace_id__model_latent_structure_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/measurement-structure": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Measurement Structure
         * @description Measurement definitions, clock, and declarations with their shared provenance.
         */
        get: operations["get_model_measurement_structure_api_episodes__workspace_id__model_measurement_structure_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/specification": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Specification
         * @description Canonical specification with prior results compatible with the selected compiler.
         */
        get: operations["get_model_specification_api_episodes__workspace_id__model_specification_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/episodes/{workspace_id}/model/posterior": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Model Posterior
         * @description Canonical posterior with compatible parameter coordinates and its own fit assessment.
         */
        get: operations["get_model_posterior_api_episodes__workspace_id__model_posterior_get"];
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
         * @description Scientific parameter definitions from the selected compiler, without inference execution.
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
         * BaselineReportArtifact
         * @description A baseline report collects treatment effects and saved scenarios from the fitted model.
         */
        BaselineReportArtifact: Domain.BaselineReportArtifact;
        /** BaselineReportVisualization */
        BaselineReportVisualization: Domain.BaselineReportVisualization;
        /**
         * CausalDesign
         * @description Scientific causal design before executable structural compilation.
         */
        CausalDesign: Domain.CausalDesign;
        /**
         * CausalDesignRef
         * @description The workspace and version of the scientific design supporting an inference.
         */
        CausalDesignRef: Domain.CausalDesignRef;
        /**
         * CausalEdge
         * @description A causal edge declares a directed causal relationship between two constructs.
         */
        CausalEdge: Domain.CausalEdge;
        /**
         * ConstantDriftMechanism
         * @description An additive continuous-time forcing of a state.
         */
        ConstantDriftMechanism: Domain.ConstantDriftMechanism;
        /**
         * Construct
         * @description A construct represents a theoretical entity in the scientific causal model.
         */
        Construct: Domain.Construct;
        ConstructId: Domain.ConstructId;
        /**
         * ConstructRef
         * @description A construct reference identifies a construct independently of its current name or
         *     revision.
         */
        ConstructRef: Domain.ConstructRef;
        /**
         * DensityPoint
         * @description A density point stores one coordinate of a prior density curve for plotting.
         */
        DensityPoint: Domain.DensityPoint;
        /**
         * DistributionFamily
         * @description This enumeration identifies the probability family used to model an observed variable.
         * @enum {string}
         */
        DistributionFamily: Domain.DistributionFamily;
        DynamicsMechanism: Domain.DynamicsMechanism;
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
        /**
         * EnergyDiagnostics
         * @description Energy diagnostics assess Hamiltonian sampling through energy distributions and mixing
         *     measures.
         */
        EnergyDiagnostics: Domain.EnergyDiagnostics;
        /**
         * EnergyHistogram
         * @description An energy histogram supplies bin centers and densities for a sampler energy plot.
         */
        EnergyHistogram: Domain.EnergyHistogram;
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
         * EstimatedCoefficient
         * @description A free coefficient referencing its scientific parameter definition.
         */
        EstimatedCoefficient: Domain.EstimatedCoefficient;
        /**
         * FactSource
         * @description A fact source locates supporting content within an artifact version and records its freshness.
         */
        FactSource: Domain.FactSource;
        /**
         * FitSummary
         * @description A fit read contains the canonical posterior and server-composed display findings.
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
         * HillEdgeMechanism
         * @description A directed saturating effect of a state through the native Hill response.
         */
        HillEdgeMechanism: Domain.HillEdgeMechanism;
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
         * IdentifiedTreatmentStatus
         * @description Details on how a treatment effect is identified.
         */
        IdentifiedTreatmentStatus: Domain.IdentifiedTreatmentStatus;
        /**
         * Indicator
         * @description An indicator defines an observed measurement of a construct and how to extract it.
         */
        Indicator: Domain.Indicator;
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
         * InitializationPolicy
         * @description This policy selects stationary-derived or freely estimated initial conditions for
         *     dynamic states.
         * @enum {string}
         */
        InitializationPolicy: Domain.InitializationPolicy;
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
         * LatentStructure
         * @description A latent structure defines the scientific causal graph of constructs and directed
         *     relationships.
         */
        LatentStructure: Domain.LatentStructure;
        /**
         * LatentStructureArtifact
         * @description This artifact stores the authored causal structure proposed for the research question.
         */
        LatentStructureArtifact: Domain.LatentStructureArtifact;
        /**
         * LikelihoodSpec
         * @description A likelihood specification defines how an indicator's observed values follow from the
         *     model.
         */
        LikelihoodSpec: Domain.LikelihoodSpec;
        /**
         * LinearEdgeMechanism
         * @description A directed effect proportional to the source state or known input.
         */
        LinearEdgeMechanism: Domain.LinearEdgeMechanism;
        /**
         * LinkFunction
         * @description A link function connects an observation distribution to the model's predictor.
         * @enum {string}
         */
        LinkFunction: Domain.LinkFunction;
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        LiteratureSource: Domain.LiteratureSource;
        /**
         * MCMCDiagnostics
         * @description MCMC diagnostics add parameter-owned convergence checks and plots to the sampler summary.
         */
        MCMCDiagnostics: Domain.MCMCDiagnostics;
        /**
         * MCMCParamDiagnostic
         * @description These diagnostics assess convergence and sampling precision for one model parameter.
         */
        MCMCParamDiagnostic: Domain.MCMCParamDiagnostic;
        /** @enum {string} */
        MeasurementDtype: Domain.MeasurementDtype;
        /**
         * MeasurementStructure
         * @description A measurement structure defines the indicators and common clock used to observe
         *     constructs.
         */
        MeasurementStructure: Domain.MeasurementStructure;
        /**
         * MeasurementStructureArtifact
         * @description This artifact stores measurement definitions and declarations that shape the executable
         *     model.
         */
        MeasurementStructureArtifact: Domain.MeasurementStructureArtifact;
        /**
         * MeasurementStructureViewData
         * @description Measurement definitions with their corresponding causal design and structural plan.
         */
        MeasurementStructureViewData: Domain.MeasurementStructureViewData;
        /**
         * MeasurementsData
         * @description Worker outcomes and panel counts derived from the same extraction revision.
         */
        MeasurementsData: Domain.MeasurementsData;
        MechanismCoefficient: Domain.MechanismCoefficient;
        /**
         * ModelRef
         * @description A model reference identifies the workspace that owns the scientific model.
         */
        ModelRef: Domain.ModelRef;
        /**
         * ModelSnapshot
         * @description A model snapshot batches independently sourced aggregates at one committed revision.
         *
         *     Authored structure, measurement declarations, specification, and posterior retain their
         *     canonical hierarchy. Optional reads represent partial models; each source preserves its
         *     own version and freshness. Only cross-artifact ownership and provenance belong here.
         */
        ModelSnapshot: Domain.ModelSnapshot;
        /**
         * ModelSpecLikelihoodDiagnostics
         * @description Observed values and validation profile for one likelihood's pinned panel.
         */
        ModelSpecLikelihoodDiagnostics: Domain.ModelSpecLikelihoodDiagnostics;
        /**
         * NodePotentialMechanism
         * @description Restoring drift -stiffness * (x - center) - quartic * (x - center)^3.
         */
        NodePotentialMechanism: Domain.NodePotentialMechanism;
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        NonIdentifiableTreatmentStatus: Domain.NonIdentifiableTreatmentStatus;
        /**
         * ObservationInterceptPolicy
         * @description This policy determines whether eligible observation intercepts are fixed or freely
         *     estimated.
         * @enum {string}
         */
        ObservationInterceptPolicy: Domain.ObservationInterceptPolicy;
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
         * ParameterConstraint
         * @description A parameter constraint specifies the permitted range of a model parameter.
         * @enum {string}
         */
        ParameterConstraint: Domain.ParameterConstraint;
        ParameterElementId: Domain.ParameterElementId;
        ParameterId: Domain.ParameterId;
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        ParameterRef: Domain.ParameterRef;
        /**
         * ParameterRole
         * @description A parameter role identifies which part of the statistical model a parameter controls.
         * @enum {string}
         */
        ParameterRole: Domain.ParameterRole;
        /**
         * ParameterSpec
         * @description A parameter specification declares a named model quantity, its role, and its allowed
         *     values.
         */
        ParameterSpec: Domain.ParameterSpec;
        /**
         * PosteriorArtifact
         * @description A joint posterior's draw axes, exact provenance, summaries, and separate assessment.
         */
        PosteriorArtifact: Domain.PosteriorArtifact;
        /**
         * PosteriorAssessment
         * @description Sampling and predictive assessments of a fitted posterior.
         */
        PosteriorAssessment: Domain.PosteriorAssessment;
        /**
         * PosteriorDrawsInfo
         * @description Axes of aligned joint draws stored in the posterior's fitted payload.
         */
        PosteriorDrawsInfo: Domain.PosteriorDrawsInfo;
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
         * PosteriorProvenance
         * @description Exact model and observation versions defining the posterior distribution.
         */
        PosteriorProvenance: Domain.PosteriorProvenance;
        /**
         * PriorAuthoringTransform
         * @description How an authored semantic prior is transformed before site attachment.
         * @enum {string}
         */
        PriorAuthoringTransform: Domain.PriorAuthoringTransform;
        /**
         * PriorDistributionFamily
         * @description This enumeration identifies the probability families permitted in authored prior
         *     proposals.
         * @enum {string}
         */
        PriorDistributionFamily: Domain.PriorDistributionFamily;
        /**
         * PriorPredictiveDiagnostic
         * @description A prior predictive diagnostic records the result of one exact model-admission check.
         */
        PriorPredictiveDiagnostic: Domain.PriorPredictiveDiagnostic;
        /**
         * PriorProposal
         * @description A prior proposal specifies a parameter's prior distribution with its rationale and
         *     supporting evidence.
         */
        PriorProposal: Domain.PriorProposal;
        /**
         * PriorSource
         * @description A prior source records literature evidence used to justify a parameter's prior
         *     distribution.
         */
        PriorSource: Domain.PriorSource;
        /** @enum {string} */
        Provenance: Domain.Provenance;
        /**
         * QuestionArtifact
         * @description A research question states the observational causal question under investigation.
         */
        QuestionArtifact: Domain.QuestionArtifact;
        /**
         * RankHistogram
         * @description A rank histogram compares parameter ranks across chains to assess mixing.
         */
        RankHistogram: Domain.RankHistogram;
        /**
         * RankHistogramChain
         * @description This histogram stores one chain's rank counts for a parameter mixing plot.
         */
        RankHistogramChain: Domain.RankHistogramChain;
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
         * SMCDiagnostics
         * @description SMC diagnostics track particle sampling through its tempering schedule, effective sample
         *     sizes, and acceptance rates.
         */
        SMCDiagnostics: Domain.SMCDiagnostics;
        /**
         * SavedScenario
         * @description A saved scenario preserves a labeled causal query and its optional narrative summary.
         */
        SavedScenario: Domain.SavedScenario;
        /**
         * SavedScenariosArtifact
         * @description Saved scenarios preserve the selected queries for a fitted model.
         */
        SavedScenariosArtifact: Domain.SavedScenariosArtifact;
        /**
         * ScenarioClamp
         * @description A resolved clamp binds its transport label to a persistent construct identity.
         */
        ScenarioClamp: Domain.ScenarioClamp;
        /**
         * ScenarioEvaluation
         * @description An evaluation binds a scientific query to one model and exact posterior version.
         */
        ScenarioEvaluation: Domain.ScenarioEvaluation;
        ScenarioEvaluationId: Domain.ScenarioEvaluationId;
        /**
         * ScenarioEvaluationResult
         * @description One posterior-specific evaluation and its matching computed outputs.
         */
        ScenarioEvaluationResult: Domain.ScenarioEvaluationResult;
        /**
         * ScenarioQuery
         * @description A scientific query keeps its identity across model and posterior revisions.
         */
        ScenarioQuery: Domain.ScenarioQuery;
        ScenarioQueryId: Domain.ScenarioQueryId;
        /** ScenarioQueryInput */
        ScenarioQueryInput: Domain.ScenarioQueryInput;
        /**
         * ScenarioResult
         * @description Computed outputs reference the evaluation that fixes their query and posterior.
         */
        ScenarioResult: Domain.ScenarioResult;
        /**
         * ScenarioStartInput
         * @description Where the forward rollout begins (replaces the rung-2/rung-3 split).
         */
        ScenarioStartInput: Domain.ScenarioStartInput;
        /** ScenarioStartResult */
        ScenarioStartResult: Domain.ScenarioStartResult;
        /**
         * ScientificOnlyConstruct
         * @description A scientific-only declaration excludes an identified construct from executable states.
         */
        ScientificOnlyConstruct: Domain.ScientificOnlyConstruct;
        /**
         * SiteKind
         * @description Semantic role for each sample site.
         * @enum {string}
         */
        SiteKind: Domain.SiteKind;
        /**
         * SourceValidity
         * @description Source validity records whether a fact still matches its pinned inputs.
         * @enum {string}
         */
        SourceValidity: Domain.SourceValidity;
        /** Sourced[BaselineReportArtifact] */
        Sourced_BaselineReportArtifact_: {
            value: components["schemas"]["BaselineReportArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[FitSummary] */
        Sourced_FitSummary_: {
            value: components["schemas"]["FitSummary"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[IdentifiabilityStatus] */
        Sourced_IdentifiabilityStatus_: {
            value: components["schemas"]["IdentifiabilityStatus"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[LatentStructure] */
        Sourced_LatentStructure_: {
            value: components["schemas"]["LatentStructure"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[MeasurementStructureArtifact] */
        Sourced_MeasurementStructureArtifact_: {
            value: components["schemas"]["MeasurementStructureArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[MeasurementsData] */
        Sourced_MeasurementsData_: {
            value: components["schemas"]["MeasurementsData"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[PosteriorArtifact] */
        Sourced_PosteriorArtifact_: {
            value: components["schemas"]["PosteriorArtifact"];
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
        /** Sourced[SavedScenariosArtifact] */
        Sourced_SavedScenariosArtifact_: {
            value: components["schemas"]["SavedScenariosArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[StatisticalModelSpecArtifact] */
        Sourced_StatisticalModelSpecArtifact_: {
            value: components["schemas"]["StatisticalModelSpecArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[ValidationReportArtifact] */
        Sourced_ValidationReportArtifact_: {
            value: components["schemas"]["ValidationReportArtifact"];
            source: components["schemas"]["FactSource"];
        };
        /** Sourced[tuple[ParameterSpec, ...]] */
        Sourced_tuple_ParameterSpec__________: {
            /** Value */
            value: components["schemas"]["ParameterSpec"][];
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
         * StatisticalModelSpec
         * @description A statistical model specification defines likelihoods, parameter roles, and estimation
         *     policies.
         */
        StatisticalModelSpec: Domain.StatisticalModelSpec;
        /**
         * StatisticalModelSpecArtifact
         * @description This artifact combines the statistical specification with prior proposals and admission
         *     diagnostics.
         */
        StatisticalModelSpecArtifact: Domain.StatisticalModelSpecArtifact;
        /**
         * StatisticalModelSpecData
         * @description A specification with observed likelihood diagnostics from its pinned inputs.
         */
        StatisticalModelSpecData: Domain.StatisticalModelSpecData;
        /**
         * StructuralDisposition
         * @description A structural disposition classifies how compilation uses or excludes an authored model
         *     entity.
         * @enum {string}
         */
        StructuralDisposition: Domain.StructuralDisposition;
        /**
         * StructuralEdge
         * @description A structural edge connects retained states or known inputs in the executable model.
         */
        StructuralEdge: Domain.StructuralEdge;
        /**
         * StructuralInducedDependency
         * @description An induced dependency records dependence created by projecting explicit latent root
         *     confounders.
         */
        StructuralInducedDependency: Domain.StructuralInducedDependency;
        /**
         * StructuralItemDisposition
         * @description An item disposition explains the compilation decision for one identified authored
         *     entity.
         */
        StructuralItemDisposition: Domain.StructuralItemDisposition;
        /**
         * StructuralKnownInput
         * @description A structural known input binds an observed indicator to a driver of the executable
         *     dynamics.
         */
        StructuralKnownInput: Domain.StructuralKnownInput;
        /**
         * StructuralPlan
         * @description A structural plan translates the scientific causal design into the topology used by
         *     model compilation.
         */
        StructuralPlan: Domain.StructuralPlan;
        /**
         * StructuralSemanticCatalog
         * @description The semantic catalog preserves authored definitions under the IDs used by the executable
         *     plan.
         */
        StructuralSemanticCatalog: Domain.StructuralSemanticCatalog;
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
         * TraceChain
         * @description A trace chain stores a thinned sequence of parameter draws from one sampling chain.
         */
        TraceChain: Domain.TraceChain;
        /**
         * TraceData
         * @description Trace data groups a parameter's sampled paths across chains for visual inspection.
         */
        TraceData: Domain.TraceData;
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
        /**
         * WorkerStatus
         * @description A worker status reports extraction progress, produced measurements, and any failure for
         *     one worker.
         */
        WorkerStatus: Domain.WorkerStatus;
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
    get_model_latent_structure_api_episodes__workspace_id__model_latent_structure_get: {
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
                    "application/json": components["schemas"]["Sourced_LatentStructure_"] | null;
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
    get_model_measurement_structure_api_episodes__workspace_id__model_measurement_structure_get: {
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
                    "application/json": components["schemas"]["Sourced_MeasurementStructureArtifact_"] | null;
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
    get_model_specification_api_episodes__workspace_id__model_specification_get: {
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
                    "application/json": components["schemas"]["Sourced_StatisticalModelSpecArtifact_"] | null;
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
    get_model_posterior_api_episodes__workspace_id__model_posterior_get: {
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
                    "application/json": components["schemas"]["Sourced_PosteriorArtifact_"] | null;
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
                    "application/json": components["schemas"]["Construct"][];
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
                    "application/json": components["schemas"]["CausalEdge"][];
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
                    "application/json": components["schemas"]["Indicator"][];
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
