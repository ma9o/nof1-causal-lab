/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python Pydantic models via:
 *   cd apps/data-pipeline && uv run python -m scripts.codegen.export_api
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py
 * plus facade API models exported from apps/data-pipeline/src/nof1_causal_lab/study_api.py
 */
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructId".
 */
export type ConstructId = `construct:${string}`;
/**
 * Uploaded sources or one recorded simulation replicate.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataSourceRef".
 */
export type DataSourceRef = FileSourceRef | SimulationReplicateRef;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "GitOid".
 */
export type GitOid = string;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionSpec".
 */
export type ExtractionSpec = ComputedExtractionSpec | SemanticExtractionSpec;
/**
 * Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WindowExpression".
 */
export type WindowExpression = string;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeId".
 */
export type EdgeId = `edge:${string}`;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismId".
 */
export type MechanismId = `mechanism:${string}`;
/**
 * A scalar expression composes supported arithmetic with scientific state and coefficient references.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Expression".
 */
export type Expression = LiteralExpression | StateExpression | CoefficientExpression | BinaryExpression | CallExpression;
/**
 * A coefficient role identifies an expression operand’s scientific quantity and support.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CoefficientRole".
 */
export type CoefficientRole = "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
/**
 * A scientific parameter identity connects component coefficients to one parameter definition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterId".
 */
export type ParameterId = `parameter:${string}`;
/**
 * A binary operator combines two scalar expression operands.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BinaryOperator".
 */
export type BinaryOperator = "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
/**
 * An expression function transforms scalar operands or constructs structured observation arguments.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExpressionFunction".
 */
export type ExpressionFunction = "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationLawSpec".
 */
export type ObservationLawSpec = DeltaLawSpec<Expression> | NormalLawSpec<Expression> | StudentTLawSpec<Expression> | PoissonLawSpec<Expression> | GammaLawSpec<Expression> | BernoulliLogitsLawSpec<Expression> | BernoulliProbsLawSpec<Expression> | NegativeBinomial2LawSpec<Expression> | BetaLawSpec<Expression> | OrderedLogisticLawSpec<Expression> | CategoricalLawSpec<Expression>;
/**
 * Indicator polarity states whether a measurement increases or decreases with its
 * construct.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorPolarity".
 */
export type IndicatorPolarity = "positive" | "negative";
/**
 * A dynamics mechanism declares one contribution to continuous-time drift.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DynamicsMechanismSpec".
 */
export type DynamicsMechanismSpec = DriftMechanismSpec | PotentialMechanismSpec;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DistributionId".
 */
export type DistributionId = `distribution:${string}`;
/**
 * A construct role states whether the variable is modeled as endogenous or treated as
 * exogenous.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Role".
 */
export type Role = "endogenous" | "exogenous";
/**
 * Temporal status states whether a construct varies within the individual over time.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TemporalStatus".
 */
export type TemporalStatus = "time_varying" | "time_invariant";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterTransformSpec".
 */
export type ParameterTransformSpec = IdentityTransformSpec | PersistenceTransformSpec | IntervalEffectTransformSpec | InitialCorrelationTransformSpec;
/**
 * A JSON value transports a scalar or a recursive array or object.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JsonValue".
 */
export type JsonValue = JsonScalar | JsonArray | JsonObject;
/**
 * A JSON scalar transports a string, number, boolean, or null.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JsonScalar".
 */
export type JsonScalar = boolean | number | string | null;
/**
 * A JSON array transports an ordered collection of recursively typed values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JsonArray".
 */
export type JsonArray = readonly JsonValue[];
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorId".
 */
export type IndicatorId = `indicator:${string}`;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SpecificationAssessment".
 */
export type SpecificationAssessment = Evaluated<string, string> | NotEvaluated<string>;
/**
 * A closed action attempt pairs its request with only that action's successful result or failure outcome.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionAttempt".
 */
export type ActionAttempt = Attempt<"set_question", SetQuestionRequest, null> | Attempt<"edit_model", EditModelRequest, null> | Attempt<"prepare_data", PrepareDataRequest, DataPreparationResult> | Attempt<"fit", FitRequest, ModelFitResult> | Attempt<"simulate", SimulateRequest, ModelSimulationResult> | Attempt<"data_diff", DataDiffRequest, DataComparisonResult>;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactId".
 */
export type ArtifactId = "question" | "raw_data" | "model" | "identification_report" | "panel" | "data_profile" | "validation_report";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QuestionAssessment".
 */
export type QuestionAssessment = Evaluated<QuestionSubject, string> | NotEvaluated<QuestionSubject>;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveLawProvenance".
 */
export type PredictiveLawProvenance = AuthoredLawProvenance | FittedLawProvenance | MixedLawProvenance | UnknownLawProvenance;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelPredictiveEvaluation".
 */
export type ModelPredictiveEvaluation = EvaluatedPredictiveChecks | UnavailablePredictiveChecks;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveAssessment".
 */
export type PredictiveAssessment = Evaluated<PredictiveSubject, readonly NumericCriterionEvidence[]> | NotEvaluated<PredictiveSubject>;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveCheckReason".
 */
export type PredictiveCheckReason = "MODEL_INCOMPLETE" | "MODEL_NOT_EXECUTABLE" | "NO_COMPATIBLE_PANEL" | "INSUFFICIENT_OBSERVATION_TIMES" | "SIMULATION_UNSUPPORTED" | "ARCHIVED_MEASUREMENT_NOT_RETAINED";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CheckGroup".
 */
export type CheckGroup = "specification" | "identification" | "compatibility" | "question";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionId".
 */
export type ActionId = ScientificActionId | "data_diff";
/**
 * A scientific action identity selects setting the question, model editing, data preparation, fitting, or simulation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScientificActionId".
 */
export type ScientificActionId = "set_question" | "edit_model" | "prepare_data" | "fit" | "simulate";
/**
 * A poll is either running labels or a completed typed attempt with its optional publication identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionPoll".
 */
export type ActionPoll = RunningPoll | CompletedPoll;
/**
 * An artifact's presence and freshness are derived from the selected journal revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactFreshness".
 */
export type ArtifactFreshness = Missing | Present;
/**
 * Whether a selected artifact still matches its pinned inputs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SourceValidity".
 */
export type SourceValidity = "fresh" | "stale";
/**
 * A parameter element identity identifies a logical scalar component across model revisions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterElementId".
 */
export type ParameterElementId = `element:${string}`;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConvergenceAssessmentSubject".
 */
export type ConvergenceAssessmentSubject = ConvergenceSubject | "recorded_parameter_chains";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConvergenceCriterion".
 */
export type ConvergenceCriterion = "r_hat" | "ess_bulk" | "ess_tail";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DSMCLeafProposal".
 */
export type DSMCLeafProposal = "amala_exact" | "paid_mix";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataRef".
 */
export type DataRef = PanelRef | SimulationRef;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataStatistic".
 */
export type DataStatistic = "observed_count" | "missing_count" | "mean" | "sd" | "min" | "max" | "proportion";
/**
 * A predictive comparison selects one reference history or records why no reference comparison applies.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveComparisonResult".
 */
export type PredictiveComparisonResult = PredictiveComparison | Unavailable | NotApplicable;
/**
 * A data selection identifies one or more saved observation histories.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataSelection".
 */
export type DataSelection = DataRef | readonly [
    DataRef,
    ...readonly DataRef[]
];
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionWorkerResult".
 */
export type ExtractionWorkerResult = CompletedExtractionWorker | FailedExtractionChunk;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EntityRef".
 */
export type EntityRef = ConstructRef | EdgeRef | IndicatorRef | MechanismRef;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitReliability".
 */
export type FitReliability = "not_fitted" | "converged" | "unconverged" | "unknown";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementDtype".
 */
export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";
/**
 * A structural disposition classifies how compilation uses or excludes an authored model
 * entity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralDisposition".
 */
export type StructuralDisposition = "retained_state" | "marginalized" | "identification_only" | "retained_edge" | "projected_edge" | "manifest" | "excluded_indicator" | "unsupported";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NotEvaluatedReason".
 */
export type NotEvaluatedReason = PredictiveCheckReason | ("NONFINITE_EMISSION_MEAN" | "INSUFFICIENT_TIMES" | "NO_RELAXATION_TERM" | "EDGE_CONTRASTS_EXPLICIT" | "NO_OBSERVATION_SUPPORT" | "NO_OBSERVATIONS" | "STATIC_CONSTRUCT" | "INSUFFICIENT_OBSERVATIONS" | "ZERO_RESIDUAL_VARIANCE" | "ZERO_OBSERVED_VARIANCE" | "NONFINITE_PATHS" | "NONFINITE_SIGNAL" | "COMPARISON_INPUTS_MISSING" | "INSUFFICIENT_CHAIN_SAMPLES" | "NO_RETAINED_CHAINS" | "ARCHIVED_ENGINE_NOT_RETAINED" | "NO_OUTCOME" | "CONSTRUCT_UNDEFINED" | "NO_PANEL" | "STATE_NOT_RECORDED");
/**
 * Every retained parameter coordinate is available without thinning or pair selection, or has an explicit unavailable reason.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterDraws".
 */
export type ParameterDraws = Available<readonly ParameterDrawColumn[]> | Unavailable;
/**
 * A progress event records one running attempt's step status or extraction telemetry.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ProgressEvent".
 */
export type ProgressEvent = StepEvent | ExtractionPlanEvent | ExtractionWorkerEvent | ExtractionSnapshotEvent;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ProgressStep".
 */
export type ProgressStep = "ingestion" | "extraction";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StepStatus".
 */
export type StepStatus = "running" | "completed" | "failed";
/**
 * A query name labels one contrast of the study question.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QueryName".
 */
export type QueryName = string;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QuestionSubject".
 */
export type QuestionSubject = OutcomeSubject | QueryTargetSubject | QueryWindowSubject;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RejectionReason".
 */
export type RejectionReason = "revision_conflict" | "input_unavailable" | "scientific_inputs" | "recorded_rejection";
/**
 * A summary operator specifies how values within a measurement window produce one
 * observation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SummaryOperator".
 */
export type SummaryOperator = "first" | "last" | "sum" | "count" | "mean" | "std";
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Assessment".
 */
export type Assessment<Subject, Evidence> = Evaluated<Subject, Evidence> | NotEvaluated<Subject>;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Change".
 */
export type Change<PayloadT> = Added<PayloadT> | Removed<PayloadT> | Revised<PayloadT>;
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Evaluation".
 */
export type Evaluation<PayloadT> = Available<PayloadT> | Unavailable | NotApplicable;
/**
 * Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.
 */
export interface CausalSSMContracts {
    readonly question?: QuestionSpec;
    readonly panel?: PreparedDataMetadata;
    readonly model?: ModelSpec;
    readonly identification_report?: IdentificationReport;
    readonly data_profile?: DataProfileArtifact;
    readonly validation_report?: ValidationReportArtifact;
}
/**
 * What the study asks: the user's words, the outcome, and named contrasts.
 *
 * Each query is a contrast of its interventions against the recorded course.
 * Constructs are named by identity before a model defines them.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QuestionSpec".
 */
export interface QuestionSpec {
    /**
     * The user's question in their own words.
     */
    readonly text: string;
    /**
     * The construct whose course answers the question.
     */
    readonly outcome: ConstructId | null;
    /**
     * Named contrasts against the recorded course, each with at least one intervention.
     */
    readonly queries: Readonly<Partial<Record<QueryName, SimulationSpec>>>;
}
/**
 * Generate from a calendar day over a horizon, with interventions placed after the start.
 *
 * The start is the only absolute time. A record's origin places it in model days;
 * without a record, the start is model day zero.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationSpec".
 */
export interface SimulationSpec {
    /**
     * Calendar day the window starts, at 00:00 UTC.
     */
    readonly start: string;
    /**
     * How long the window lasts, such as 9w or 61d.
     */
    readonly horizon: string;
    readonly interventions: readonly InterventionSpec[];
}
/**
 * Set a state some time after the design's start, then let its dynamics resume.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InterventionSpec".
 */
export interface InterventionSpec {
    readonly target: ConstructId;
    /**
     * Offset from the design's start; omitted means at the start.
     */
    readonly after: string | null;
    readonly value: number;
}
/**
 * Self-contained semantics and provenance of one prepared observation table.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PreparedDataMetadata".
 */
export interface PreparedDataMetadata {
    readonly source: DataSourceRef;
    /**
     * @minItems 1
     */
    readonly variables: readonly [
        ObservationSpec<string>,
        ...readonly ObservationSpec<string>[]
    ];
    readonly preparation: DataPreparationSpec | null;
    /**
     * Calendar instant of model day zero; null denotes a calendar-free history.
     */
    readonly time_origin: string | null;
}
/**
 * Explicit uploaded filenames, relative to this study's input directory.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FileSourceRef".
 */
export interface FileSourceRef {
    /**
     * @minItems 1
     */
    readonly files: readonly [
        string,
        ...readonly string[]
    ];
    /**
     * Inclusive UTC source-coverage date.
     */
    readonly start: string | null;
    /**
     * Exclusive UTC source-coverage date.
     */
    readonly end: string | null;
}
/**
 * One replicate from a recorded, applied simulation in this study.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationReplicateRef".
 */
export interface SimulationReplicateRef {
    readonly revision: GitOid;
    readonly replicate: number;
}
/**
 * A versioned data definition supplied directly to prepare_data.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataPreparationSpec".
 */
export interface DataPreparationSpec {
    readonly default_window: string;
    /**
     * @minItems 1
     */
    readonly variables: readonly [
        DataVariableSpec,
        ...readonly DataVariableSpec[]
    ];
    /**
     * Optional context for interpreting the source data.
     */
    readonly context: string;
}
/**
 * Compose an observed variable with its data-owned extraction instructions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataVariableSpec".
 */
export interface DataVariableSpec {
    readonly observation: ObservationSpec<string | null>;
    readonly extraction: ExtractionSpec;
}
/**
 * Compute a deterministic support-window measurement from source columns.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ComputedExtractionSpec".
 */
export interface ComputedExtractionSpec {
    readonly kind: "computed";
    /**
     * Description of the deterministic measurement.
     */
    readonly how_to_measure: string;
    /**
     * @minItems 1
     */
    readonly source_columns: readonly [
        string,
        ...readonly string[]
    ];
    /**
     * Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation.
     */
    readonly computed_rule: WindowExpression | null;
    /**
     * Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
     */
    readonly fill_null: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
    /**
     * Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
     */
    readonly fill_null_limit: number | null;
}
/**
 * Interpret source records using an explicit measurement rubric.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SemanticExtractionSpec".
 */
export interface SemanticExtractionSpec {
    readonly kind: "semantic";
    /**
     * Scoring rubric and extraction instructions.
     */
    readonly how_to_measure: string;
    /**
     * Source columns exposed to the extraction worker.
     */
    readonly source_columns: readonly string[];
}
/**
 * A connected causal graph with owned scientific detail, built to answer the study question.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSpec".
 */
export interface ModelSpec {
    readonly edges: readonly CausalEdgeSpec[];
    readonly parameters: readonly ParameterSpec[];
    /**
     * All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements.
     */
    readonly distributions: Readonly<Partial<Record<DistributionId, NumPyroDistribution>>>;
    readonly time_points: readonly number[];
    readonly measurement_clock: string | null;
}
/**
 * A specification of a directed causal relationship between two constructs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalEdgeSpec".
 */
export interface CausalEdgeSpec {
    readonly id: EdgeId;
    readonly mechanisms: readonly DriftMechanismSpec[];
    /**
     * Cause construct; shared endpoints have one identity.
     */
    readonly cause: ConstructSpec | ConstructRef;
    /**
     * Effect construct; shared endpoints have one identity.
     */
    readonly effect: ConstructSpec | ConstructRef;
    /**
     * Theoretical justification for this causal link
     */
    readonly description: string;
    /**
     * Literature sources supporting this causal link
     */
    readonly sources: readonly LiteratureSource[];
}
/**
 * An additive drift contribution on a construct or directed edge.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DriftMechanismSpec".
 */
export interface DriftMechanismSpec {
    readonly id: MechanismId;
    readonly expression: Expression;
    readonly kind: "drift";
}
/**
 * A finite scalar constant in a model equation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LiteralExpression".
 */
export interface LiteralExpression {
    readonly kind: "literal";
    readonly value: number;
}
/**
 * A construct's state or declared known input, referenced by identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StateExpression".
 */
export interface StateExpression {
    readonly kind: "state";
    readonly construct_id: ConstructId;
}
/**
 * A scientifically typed coefficient operand, literal or parameter reference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CoefficientExpression".
 */
export interface CoefficientExpression {
    readonly kind: "coefficient";
    readonly role: CoefficientRole;
    /**
     * Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
     */
    readonly value: number | ParameterId | null;
    /**
     * Additional constructs participating in this coefficient use.
     */
    readonly construct_ids: readonly ConstructId[];
}
/**
 * A supported scalar operation composing two expressions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BinaryExpression".
 */
export interface BinaryExpression {
    readonly kind: "binary";
    readonly operator: BinaryOperator;
    readonly left: Expression;
    readonly right: Expression;
}
/**
 * A supported mathematical function, including explicit discrete contrasts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CallExpression".
 */
export interface CallExpression {
    readonly kind: "call";
    readonly function: ExpressionFunction;
    readonly arguments: readonly Expression[];
}
/**
 * A specification of a theoretical entity in the scientific causal model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructSpec".
 */
export interface ConstructSpec {
    readonly id: ConstructId;
    /**
     * Construct name (e.g., 'stress', 'sleep_quality')
     */
    readonly name: string;
    /**
     * What this theoretical construct represents
     */
    readonly description: string;
    readonly indicators: readonly IndicatorSpec[];
    readonly dynamics: readonly DynamicsMechanismSpec[];
    readonly coefficients: readonly CoefficientExpression[];
    readonly innovation_family: "gaussian" | "student_t";
    /**
     * Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points.
     */
    readonly distribution: DistributionId | null;
    readonly role: Role;
    readonly temporal_status: TemporalStatus;
}
/**
 * Bind an observed-variable ID to a construct and an emission likelihood.
 *
 * Extraction instructions belong to DataPreparationSpec. The shared observation
 * schema also permits generative models before any observations have been collected.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorSpec".
 */
export interface IndicatorSpec {
    readonly observation: ObservationSpec<string | null>;
    readonly likelihood: LikelihoodSpec | null;
    readonly construct_polarity: IndicatorPolarity;
}
/**
 * An indicator's conditional probability law and its scientific justification.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodSpec".
 */
export interface LikelihoodSpec {
    readonly law: ObservationLawSpec;
    /**
     * Whether observations are mean-centered and scaled before fitting.
     */
    readonly standardized: boolean;
    /**
     * Why this conditional law was chosen for the indicator
     */
    readonly reasoning: string;
    readonly sources: readonly LiteratureSource[];
}
/**
 * A literature source records cited evidence supporting a scientific modeling decision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LiteratureSource".
 */
export interface LiteratureSource {
    /**
     * Title of the source (paper, meta-analysis, textbook, etc.)
     */
    readonly title: string;
    /**
     * URL of the source if available
     */
    readonly url: string | null;
    /**
     * Relevant excerpt or paraphrase from the source
     */
    readonly snippet: string;
}
/**
 * A construct potential whose negative gradient contributes to its drift.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PotentialMechanismSpec".
 */
export interface PotentialMechanismSpec {
    readonly id: MechanismId;
    readonly expression: Expression;
    readonly kind: "potential";
}
/**
 * A construct reference identifies a construct independently of its current name or
 * revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructRef".
 */
export interface ConstructRef {
    readonly kind: "construct";
    readonly id: ConstructId;
}
/**
 * A named uncertain quantity; fixed coefficients are literals in component slots.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterSpec".
 */
export interface ParameterSpec {
    readonly id: ParameterId;
    /**
     * Authored parameter label; relationships use its persistent ID
     */
    readonly name: string;
    /**
     * Human-readable description of what this parameter represents
     */
    readonly description: string;
    readonly transform: ParameterTransformSpec;
    /**
     * Membership in a native law in ModelSpec.distributions; may be joint. None means the law has not been assigned yet.
     */
    readonly distribution: DistributionId | null;
    /**
     * Why the authored prior law fits this quantity, and where its values come from.
     */
    readonly reasoning: string | null;
    /**
     * Evidence behind the authored prior law.
     */
    readonly sources: readonly LiteratureSource[];
}
/**
 * Keep the authored probability law on the scientific quantity's native scale.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentityTransformSpec".
 */
export interface IdentityTransformSpec {
    readonly kind: "identity";
}
/**
 * Map persistence p to -log(p) divided by its explicit interval in days.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PersistenceTransformSpec".
 */
export interface PersistenceTransformSpec {
    readonly kind: "dt_persistence_to_ct_decay";
    readonly interval_days: number | "model_clock";
}
/**
 * Divide an interval effect by its explicit duration in days.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IntervalEffectTransformSpec".
 */
export interface IntervalEffectTransformSpec {
    readonly kind: "dt_effect_to_ct_rate";
    readonly interval_days: number | "model_clock";
}
/**
 * Constrain an initial-state correlation to its scientific support [-1, 1].
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InitialCorrelationTransformSpec".
 */
export interface InitialCorrelationTransformSpec {
    readonly kind: "initial_state_correlation";
}
/**
 * A native NumPyro probability distribution serialized by its constructor tree.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NumPyroDistribution".
 */
export interface NumPyroDistribution {
    readonly distribution: string;
    readonly params: {
        readonly [k: string]: JsonValue | undefined;
    };
}
/**
 * A JSON object transports string-keyed recursively typed values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JsonObject".
 */
export interface JsonObject {
    readonly [k: string]: JsonValue | undefined;
}
/**
 * Positive and negative causal identification findings for the study question's outcome.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentificationReport".
 */
export interface IdentificationReport {
    readonly outcome: ConstructId | null;
    /**
     * One tagged identification result per treatment, including its supporting evidence
     */
    readonly treatments: Readonly<Partial<Record<ConstructId, (IdentifiedTreatmentStatus | NonIdentifiableTreatmentStatus)>>>;
}
/**
 * Details on how a treatment effect is identified.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentifiedTreatmentStatus".
 */
export interface IdentifiedTreatmentStatus {
    readonly status: "identified";
    /**
     * Nonparametric identification; linear-IV arguments do not certify ModelSpec.
     */
    readonly method: "do_calculus";
    /**
     * Nonparametric estimand returned by do-calculus
     */
    readonly estimand: string;
    /**
     * Unobserved confounders the estimand integrates out
     */
    readonly marginalized_confounders: readonly ConstructId[];
    /**
     * Instrument constructs appearing in the nonparametric identification argument
     */
    readonly instruments: readonly ConstructId[];
}
/**
 * Context on why a treatment effect is not identifiable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NonIdentifiableTreatmentStatus".
 */
export interface NonIdentifiableTreatmentStatus {
    readonly status: "not_identified";
    /**
     * Unobserved constructs blocking identification
     */
    readonly confounders: readonly ConstructId[];
    /**
     * Optional explanation if confounders cannot be enumerated
     */
    readonly notes: string | null;
}
/**
 * Model-independent empirical measurements and data-quality findings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataProfileArtifact".
 */
export interface DataProfileArtifact {
    readonly indicators: Readonly<Partial<Record<IndicatorId, IndicatorAudit>>>;
    readonly dataset_issues: readonly ValidationIssue[];
    /**
     * Whether the data findings contain no errors, independent of any model.
     */
    readonly is_valid: boolean;
}
/**
 * An indicator audit combines its empirical data profile with the results of validation
 * checks.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorAudit".
 */
export interface IndicatorAudit {
    readonly profile: IndicatorEmpiricalProfile | null;
    readonly issues: readonly ValidationIssue[];
    readonly checks: {
        readonly [k: string]: ("ok" | "warning" | "error" | "not_evaluated") | undefined;
    };
}
/**
 * An empirical profile summarizes an indicator's observed values, coverage, and data-
 * quality signals.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorEmpiricalProfile".
 */
export interface IndicatorEmpiricalProfile {
    readonly measurement_dtype: string | null;
    readonly n_obs: number;
    readonly mean: number | null;
    readonly std: number | null;
    readonly min: number | null;
    readonly max: number | null;
    readonly q25: number | null;
    readonly q50: number | null;
    readonly q75: number | null;
    readonly variance: number | null;
    readonly time_coverage_ratio: number | null;
    readonly max_gap_ratio: number | null;
    readonly dtype_violations: number | null;
    readonly duplicate_pct: number | null;
    readonly arithmetic_sequence_detected: boolean;
    readonly n_unparseable_timestamps: number | null;
    readonly zero_fraction: number | null;
    readonly is_nonnegative: boolean | null;
    readonly is_unit_interval: boolean | null;
    readonly looks_integer_valued: boolean | null;
    readonly variance_to_mean_ratio: number | null;
}
/**
 * A validation issue explains a data problem and its severity for an indicator or the
 * dataset.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationIssue".
 */
export interface ValidationIssue {
    /**
     * Affected indicator; null for a dataset-wide issue.
     */
    readonly indicator_id: IndicatorId | null;
    readonly issue_type: string;
    readonly severity: "error" | "warning" | "info";
    readonly message: string;
}
/**
 * Data findings composed with model-dependent execution checks.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationReportArtifact".
 */
export interface ValidationReportArtifact {
    readonly data: DataProfileArtifact;
    readonly preflight: readonly SpecificationAssessment[];
    /**
     * Whether both the data findings and model preflight contain no failures.
     */
    readonly is_valid: boolean;
}
/**
 * What an executed action did to the store: the workflow installs this.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionEffects".
 */
export interface ActionEffects {
    readonly produced: readonly ArtifactRecord[];
    readonly retracted: readonly RetractedArtifact[];
    readonly checks: ModelCheckReport | null;
}
/**
 * Artifact revision metadata records how a stored artifact was produced and which inputs it
 * used.
 *
 * ``derived_from`` pins the exact input versions the payload was computed
 * from. For initial model revisions it is empty. ``created_at`` is
 * stamped by the activity that produced the revision — never inside workflow
 * code, where wall-clock time is non-deterministic.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactRecord".
 */
export interface ArtifactRecord {
    readonly artifact_id: ArtifactId;
    readonly revision: GitOid;
    readonly derived_from: Readonly<Partial<Record<ArtifactId, GitOid>>>;
    readonly model_inputs: {
        readonly [k: string]: string | undefined;
    };
    readonly consumed_model_inputs: {
        readonly [k: string]: string | undefined;
    };
    readonly produced_by: string | null;
    readonly created_at: string;
}
/**
 * A current artifact removed by an action, with the finding that caused it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RetractedArtifact".
 */
export interface RetractedArtifact {
    readonly artifact_id: ArtifactId;
    readonly reason_ref: string;
}
/**
 * Checks selected by their consumed inputs, retained with the study snapshot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelCheckReport".
 */
export interface ModelCheckReport {
    readonly input_keys: Readonly<Partial<Record<CheckGroup, string>>>;
    readonly specification: readonly SpecificationAssessment[];
    readonly question: QuestionCheckReport | null;
    readonly predictive: ModelPredictiveReport | null;
    readonly reused: readonly (CheckGroup | "predictive")[];
}
/**
 * The study question checked against the model and, once prepared, the record.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QuestionCheckReport".
 */
export interface QuestionCheckReport {
    readonly question_revision: GitOid;
    readonly panel_revision: GitOid | null;
    readonly findings: readonly QuestionAssessment[];
}
/**
 * One automatic, reproducible battery over the full model's current laws.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelPredictiveReport".
 */
export interface ModelPredictiveReport {
    readonly input_key: string;
    readonly model_revision: GitOid;
    readonly panel_revision: GitOid | null;
    readonly draws: number;
    readonly seed: number;
    readonly law: PredictiveLawProvenance;
    readonly evaluation: ModelPredictiveEvaluation;
    readonly status: "passed" | "failed" | "not_evaluated";
}
/**
 * The current laws have authored ancestry without retained fitting.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AuthoredLawProvenance".
 */
export interface AuthoredLawProvenance {
    readonly kind: "authored";
    readonly interpretation: "prior_predictive";
}
/**
 * All current laws retain one committed fit's model and observation panel.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FittedLawProvenance".
 */
export interface FittedLawProvenance {
    readonly kind: "fitted";
    readonly fitted_panel_revision: GitOid;
    readonly fitted_model_revision: GitOid;
    readonly interpretation: "in_sample_posterior_predictive" | "posterior_predictive";
}
/**
 * Some laws retain a committed fit and others have different ancestry.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MixedLawProvenance".
 */
export interface MixedLawProvenance {
    readonly kind: "mixed";
    readonly fitted_panel_revision: GitOid;
    readonly fitted_model_revision: GitOid;
    readonly interpretation: "mixed";
}
/**
 * Imported laws do not establish a conditioning history.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "UnknownLawProvenance".
 */
export interface UnknownLawProvenance {
    readonly kind: "unknown";
    readonly interpretation: "unknown";
}
/**
 * An evaluated run may retain failed and partially unavailable scientific evidence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EvaluatedPredictiveChecks".
 */
export interface EvaluatedPredictiveChecks {
    readonly kind: "evaluated";
    readonly findings: readonly PredictiveAssessment[];
    readonly predictive_checks: PosteriorPredictiveChecks | null;
}
/**
 * Posterior predictive checks report exact-model checks and their supporting plot data.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorPredictiveChecks".
 */
export interface PosteriorPredictiveChecks {
    readonly per_variable_warnings: readonly Assessment<IndicatorCheckSubject, NumericCriterionEvidence>[];
    readonly checked: boolean;
    readonly n_subsample: number;
    readonly overlays: readonly PPCOverlay[];
    readonly test_stats: readonly PPCTestStat[];
}
/**
 * A predictive overlay sets one indicator's observed values against simulated ones.
 *
 * It carries the predictive median and a few individual replicated series, the
 * spaghetti plot of a visual predictive check.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCOverlay".
 */
export interface PPCOverlay {
    readonly times: readonly number[];
    readonly time_origin: string | null;
    readonly standardized: boolean;
    readonly indicator_id: IndicatorId;
    readonly observed: readonly (number | null)[];
    readonly median: readonly (number | null)[];
    readonly spaghetti_draws: readonly (readonly (number | null)[])[];
}
/**
 * A predictive test statistic compares an observed summary with its distribution under
 * replicated data.
 *
 * Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
 * with a vertical line at T(y_observed).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCTestStat".
 */
export interface PPCTestStat {
    readonly indicator_id: IndicatorId;
    readonly stat_name: "mean" | "sd" | "min" | "max";
    readonly observed_value: number;
    readonly rep_values: readonly number[];
    readonly p_value: number | null;
    readonly histogram: readonly HistogramBin[];
}
/**
 * A histogram bin gives its interval, center, and number of posterior draws.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "HistogramBin".
 */
export interface HistogramBin {
    readonly bin_center: number;
    readonly bin_start: number;
    readonly bin_end: number;
    readonly count: number;
}
/**
 * The run could not evaluate its scientific battery.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "UnavailablePredictiveChecks".
 */
export interface UnavailablePredictiveChecks {
    readonly kind: "unavailable";
    readonly reason: PredictiveCheckReason;
    readonly detail: string | null;
}
/**
 * A label emitted by an attempt; measurements belong in its scientific result.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionMessage".
 */
export interface ActionMessage {
    readonly timestamp: string;
    readonly level: "debug" | "info" | "warn" | "error";
    readonly label: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RunningPoll".
 */
export interface RunningPoll {
    readonly kind: "running";
    readonly messages: readonly ActionMessage[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompletedPoll".
 */
export interface CompletedPoll {
    readonly kind: "completed";
    readonly commit_id: GitOid | null;
    readonly attempt: ActionAttempt;
    readonly messages: readonly ActionMessage[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionReceipt".
 */
export interface ActionReceipt {
    readonly attempt_id: string;
}
/**
 * An artifact envelope delivers a stored payload with its revision and file
 * list.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactEnvelope".
 */
export interface ArtifactEnvelope {
    readonly workspace_id: string;
    readonly meta: ArtifactRecord;
    readonly payload: JsonObject;
    readonly binary_files: readonly string[];
}
/**
 * An artifact absent from the selected state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Missing".
 */
export interface Missing {
    readonly kind: "missing";
    readonly artifact_id: ArtifactId;
}
/**
 * The selected artifact record and its input validity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Present".
 */
export interface Present {
    readonly kind: "present";
    readonly record: ArtifactRecord;
    readonly validity: SourceValidity;
}
/**
 * Stored inside the Git object, with no self-referential publication ID.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AttemptRecord".
 */
export interface AttemptRecord {
    readonly seq: number;
    readonly attempt_id: string | null;
    readonly branch: string;
    readonly ts: string;
    readonly messages: readonly ActionMessage[];
    readonly trace_ids: readonly string[];
    readonly attempt: ActionAttempt;
}
/**
 * Promoted traces identified by their committed execution sequence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AttemptTraceIndex".
 */
export interface AttemptTraceIndex {
    readonly commit_id: GitOid;
    readonly trace_ids: readonly string[];
}
/**
 * Predictive probabilities for each declared level; unobserved anchors are null.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CategoryProbabilitySummary".
 */
export interface CategoryProbabilitySummary {
    readonly kind: "categorical";
    readonly probabilities: {
        readonly [k: string]: readonly (number | null)[] | undefined;
    };
    readonly n_draws: readonly number[];
}
/**
 * Causal effects and realized trajectories under the enclosing report's design.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalEffectResult".
 */
export interface CausalEffectResult {
    readonly outcome: ConstructId;
    readonly labels: Readonly<Partial<Record<ConstructId, string>>>;
    readonly warnings: readonly string[];
}
/**
 * Compact retained-chain measurements; plot series compose the report detail.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ChainDiagnostics".
 */
export interface ChainDiagnostics {
    readonly num_chains: number;
    readonly num_samples: number;
    readonly per_parameter: readonly ParameterDiagnostics[];
    readonly num_divergences: number | null;
    readonly divergence_rate: number | null;
    readonly tree_depth_mean: number | null;
    readonly tree_depth_max: number | null;
    readonly accept_prob_mean: number | null;
    readonly latent_accept_prob_mean: number | null;
    readonly parameter_accept_prob_mean: number | null;
    readonly energy: EnergyDiagnostics | null;
}
/**
 * Measurements on one scientifically identified retained scalar chain.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterDiagnostics".
 */
export interface ParameterDiagnostics {
    readonly parameter: string;
    readonly subject: ParameterRef;
    readonly r_hat: number | null;
    readonly ess_bulk: number | null;
    readonly ess_tail: number | null;
    readonly mcse_mean: number | null;
}
/**
 * A scalar finding identifies its scientific parameter and declared logical component.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterRef".
 */
export interface ParameterRef {
    readonly parameter_id: ParameterId;
    readonly element_id: ParameterElementId;
}
/**
 * Energy distributions and the producer's chain-specific BFMI values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EnergyDiagnostics".
 */
export interface EnergyDiagnostics {
    readonly energy_hist: DensityCurve;
    readonly energy_transition_hist: DensityCurve;
    readonly bfmi: readonly number[];
}
/**
 * Aligned density ordinates; the owning field distinguishes PDF samples from histogram heights.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DensityCurve".
 */
export interface DensityCurve {
    readonly x: readonly number[];
    readonly density: readonly number[];
}
/**
 * Retained measurements from a completed worker; no failure field exists.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompletedExtractionWorker".
 */
export interface CompletedExtractionWorker {
    readonly worker_id: number;
    readonly n_extractions: number;
    readonly n_windows: number;
    readonly n_llm_calls: number | null;
    readonly reused: boolean | null;
    readonly status: "completed";
}
/**
 * A convergence criterion on one stable scientific scalar.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConvergenceSubject".
 */
export interface ConvergenceSubject {
    readonly parameter: ParameterRef;
    readonly criterion: ConvergenceCriterion;
    readonly label: string;
}
/**
 * A retained data comparison; it never installs scientific artifacts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataComparisonResult".
 */
export interface DataComparisonResult {
    readonly action: "data_diff";
    readonly report: DataDiffReport;
}
/**
 * Comparisons of existing data, preserving each history's immutable source reference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataDiffReport".
 */
export interface DataDiffReport {
    readonly left: readonly DataRef[];
    readonly right: readonly DataRef[];
    readonly variables: readonly DataVariableDiff[];
}
/**
 * An immutable observed panel with its own calendar history.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PanelRef".
 */
export interface PanelRef {
    readonly kind: "panel";
    readonly revision: GitOid;
}
/**
 * A saved simulation; a null replicate selects all its recorded draws.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationRef".
 */
export interface SimulationRef {
    readonly kind: "simulation";
    readonly revision: GitOid;
    readonly replicate: number | null;
}
/**
 * Definitions, histories and comparisons for one persistent observation identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataVariableDiff".
 */
export interface DataVariableDiff {
    readonly indicator_id: IndicatorId;
    readonly left: readonly DataSeries[];
    readonly right: readonly DataSeries[];
    readonly changes: readonly Change<DataPoint>[];
    readonly statistics: readonly DataStatisticComparison[];
    readonly comparison_issues: readonly string[];
    readonly predictive: PredictiveComparisonResult;
}
/**
 * One variable's recorded measurements in one history; no pooling across replicas.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataSeries".
 */
export interface DataSeries {
    readonly variable: ObservationSpec<string> | null;
    /**
     * Recorded calendar binding; null means the point dates are serialization coordinates, not real dates.
     */
    readonly time_origin: string | null;
    readonly points: readonly DataPoint[];
}
/**
 * An observed anchor and support; dates are synthetic for a calendar-free series.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataPoint".
 */
export interface DataPoint {
    readonly anchor_time: string;
    readonly support_start: string | null;
    readonly support_end: string | null;
    readonly value: number | null;
}
/**
 * The same descriptive statistic measured independently in every selected history.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataStatisticComparison".
 */
export interface DataStatisticComparison {
    readonly statistic: DataStatistic;
    readonly level: string | null;
    readonly left: readonly (number | null)[];
    readonly right: readonly (number | null)[];
    readonly left_histogram: readonly HistogramBin[];
    readonly right_histogram: readonly HistogramBin[];
}
/**
 * A selected reference history retains its role even when checks are unavailable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveComparison".
 */
export interface PredictiveComparison {
    readonly kind: "comparison";
    readonly reference_side: "left" | "right";
    readonly evaluation: Evaluation<PosteriorPredictiveChecks>;
}
/**
 * An applicable result could not be produced, for an explicit reason.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Unavailable".
 */
export interface Unavailable {
    readonly kind: "unavailable";
    readonly reason: string;
}
/**
 * The selected operation does not call for this result.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NotApplicable".
 */
export interface NotApplicable {
    readonly kind: "not_applicable";
    readonly reason: string;
}
/**
 * Compare two immutable data selections, each containing one or more histories.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataDiffRequest".
 */
export interface DataDiffRequest {
    readonly action: "data_diff";
    readonly left: DataSelection;
    readonly right: DataSelection;
}
/**
 * Preparation artifacts and the measurements actually retained by extraction.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DataPreparationResult".
 */
export interface DataPreparationResult {
    readonly action: "prepare_data";
    readonly raw_data: GitRef | null;
    readonly model: GitRef | null;
    readonly simulation_source: SimulationReplicateRef | null;
    readonly n_observations: number | null;
    readonly workers: readonly ExtractionWorkerResult[];
    readonly ingestion_reused: boolean | null;
    readonly extraction_reused: number | null;
}
/**
 * An exact file in a study's Git object database: repository, object, and path.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "GitRef".
 */
export interface GitRef {
    readonly workspace_id: string;
    readonly revision: GitOid;
    readonly path: string;
}
/**
 * An extraction failure with its error and no usable result-file reference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FailedExtractionChunk".
 */
export interface FailedExtractionChunk {
    readonly worker_id: number;
    readonly n_extractions: number;
    readonly n_windows: number;
    readonly n_llm_calls: number | null;
    readonly reused: boolean | null;
    readonly status: "failed";
    readonly error: string;
}
/**
 * An edge reference identifies a causal relationship independently of edits to its
 * definition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeRef".
 */
export interface EdgeRef {
    readonly kind: "edge";
    readonly id: EdgeId;
}
/**
 * Replace one named base revision with a validated scientific definition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EditModelRequest".
 */
export interface EditModelRequest {
    readonly action: "edit_model";
    readonly expected_revision: GitOid | null;
    readonly model: ModelSpec;
}
/**
 * An effect summary reports posterior location, uncertainty, and sign probability.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EffectSummary".
 */
export interface EffectSummary {
    readonly mean: number;
    readonly median: number;
    readonly lower_95: number;
    readonly upper_95: number;
    readonly prob_positive: number;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EmpiricalPoint".
 */
export interface EmpiricalPoint {
    readonly value: number;
    readonly probability: number;
    readonly count: number;
}
/**
 * An indicator reference identifies a measurement definition independently of its name or
 * revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorRef".
 */
export interface IndicatorRef {
    readonly kind: "indicator";
    readonly id: IndicatorId;
}
/**
 * A particular additive term, independently of its position or coefficient values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismRef".
 */
export interface MechanismRef {
    readonly kind: "mechanism";
    readonly id: MechanismId;
}
/**
 * The extraction fan-out plan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionPlanEvent".
 */
export interface ExtractionPlanEvent {
    readonly attempt_id: string;
    readonly cursor: string;
    readonly event: "nof1-causal-lab.extraction.plan";
    readonly total_workers: number;
    readonly max_concurrent_workers: number | null;
}
/**
 * Aggregate extraction worker counts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionSnapshotEvent".
 */
export interface ExtractionSnapshotEvent {
    readonly attempt_id: string;
    readonly cursor: string;
    readonly event: "nof1-causal-lab.extraction.snapshot";
    readonly total_workers: number;
    readonly pending_workers: number;
    readonly running_workers: number;
    readonly completed_workers: number;
    readonly failed_workers: number;
}
/**
 * One extraction worker's state; a worker reports its LLM calls when it finishes.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionWorkerEvent".
 */
export interface ExtractionWorkerEvent {
    readonly attempt_id: string;
    readonly cursor: string;
    readonly event: "nof1-causal-lab.extraction.worker";
    readonly worker_id: number;
    readonly state: "pending" | "running" | "completed" | "failed";
    readonly n_windows: number;
    readonly n_extractions: number | null;
    readonly n_llm_calls: number | null;
    readonly error: string | null;
}
/**
 * A fact source locates supporting content within an artifact revision and records its freshness.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FactSource".
 */
export interface FactSource {
    readonly ref: GitRef;
    readonly pointer: string;
    readonly validity: SourceValidity;
}
/**
 * Uploaded sources and the complete recipe for preparing their observations.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FilePreparationSpec".
 */
export interface FilePreparationSpec {
    readonly source: FileSourceRef;
    readonly definition: DataPreparationSpec;
}
/**
 * Condition explicitly selected model and observation revisions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitRequest".
 */
export interface FitRequest {
    readonly action: "fit";
    readonly model_revision: GitOid;
    readonly panel_revision: GitOid;
    readonly settings: FitSettingsSpec;
}
/**
 * Optional numerical controls applied to the configured particle sampler.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitSettingsSpec".
 */
export interface FitSettingsSpec {
    readonly num_samples: number | null;
    readonly num_warmup: number | null;
    readonly num_chains: number | null;
    readonly n_particles: number | null;
    readonly seed: number | null;
}
/**
 * A fit read contains the inference report summary and server-composed display findings.
 *
 * Per-draw diagnostics load separately from the inference report endpoint.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitSummary".
 */
export interface FitSummary {
    readonly report: InferenceReportCore;
    readonly edge_estimates: Readonly<Partial<Record<EdgeId, ParameterRef>>>;
    readonly decay_estimates: Readonly<Partial<Record<ConstructId, ParameterRef>>>;
    /**
     * Conditioned input laws of the fitted parameters, on their posterior marginals' quantity scale; absent where the current compiler cannot place the input model.
     */
    readonly prior_densities: Readonly<Partial<Record<ParameterId, DensityCurve>>>;
}
/**
 * Compact scientific report shared by snapshots and the full report.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceReportCore".
 */
export interface InferenceReportCore {
    readonly time_origin: string | null;
    readonly inference_metadata: InferenceMetadata;
    readonly engine: Assessment<string, ParticleMCMCEvidence>;
    readonly inference_diagnostics: ChainDiagnostics | null;
    readonly sampler_diagnostics: ParticleSamplerDiagnostics | null;
    readonly convergence: ParameterConvergenceReport;
    readonly loo_diagnostics: LOODiagnostics | null;
    readonly posterior_marginals: readonly PosteriorMarginal[] | null;
}
/**
 * Inference metadata records the sampling method, sample count, and run duration.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceMetadata".
 */
export interface InferenceMetadata {
    readonly method: string;
    readonly n_samples: number;
    readonly duration_seconds: number;
}
/**
 * Typed exact-sampler settings and transition telemetry from the native producer.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParticleSamplerDiagnostics".
 */
export interface ParticleSamplerDiagnostics {
    readonly settings: SamplerSpec;
    readonly latent_kernel: string;
    readonly latent_smoother: string;
    readonly latent_smoother_algorithm: string;
    readonly latent_smoother_family: string;
    readonly latent_smoother_selection: string;
    readonly latent_smoother_parallel: boolean;
    readonly parameter_kernel: string;
    readonly mcmc_phase_seconds: number;
    readonly latent_backward_sampling: boolean;
    readonly amala_delta_adapted: boolean;
    readonly dsmc_leaf_proposal: string;
    readonly latent_transition_kind: string;
    readonly diagnostic_metrics: readonly string[];
    readonly param_target_accept: number;
    readonly parameter_preconditioned: boolean;
    readonly diagnostic_summary_phase: string;
    readonly parameter_accept_rate: number;
    readonly latent_update_fraction: number;
    readonly latent_frozen_fraction: number;
    readonly latent_block_coords: number | null;
    readonly initial_param_step_size: readonly number[];
    readonly final_param_step_size: readonly number[];
    readonly latent_init_method: string;
    readonly latent_sign_flip_moves: boolean | null;
    readonly chain_post_warmup_complete_log_posterior_mean: readonly number[];
    readonly latent_move_rms_mean: number | null;
    readonly parameter_jump_rms_mean: number | null;
    readonly reference_path_hit_rate_mean: number | null;
    readonly selected_particle_unique_count_mean: number | null;
    readonly amala_grad_norm_mean: number | null;
    readonly amala_grad_norm_max: number | null;
    readonly parameter_warmup: ParameterWarmupDiagnostics;
}
/**
 * Fully resolved controls for the exact particle sampler.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SamplerSpec".
 */
export interface SamplerSpec {
    readonly num_warmup: number;
    readonly num_samples: number;
    readonly num_chains: number;
    readonly seed: number;
    readonly n_particles: number;
    readonly retain_latent_paths: boolean;
    readonly marginal_particle_gibbs: MarginalParticleGibbsSpec;
}
/**
 * Marginalized Particle Gibbs inference settings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MarginalParticleGibbsSpec".
 */
export interface MarginalParticleGibbsSpec {
    readonly n_parameter_particles: number;
    readonly latent_smoother: "dsmc";
    readonly latent_delta: number;
    readonly parameter_proposal: "random_walk" | "pseudo_langevin";
    readonly amala_delta_init: number;
    readonly amala_delta_min: number;
    readonly amala_delta_max: number;
    readonly amala_target_accept: number;
    readonly amala_adaptation_window: number;
    readonly amala_adaptation_tolerance: number;
    readonly amala_adaptation_rho: number;
    readonly amala_adaptation_rho_min: number;
    readonly amala_adaptation_gamma: number;
    readonly amala_kappa: number;
    readonly amala_grad_clip: number | "infinity";
    readonly dsmc_leaf_proposal: DSMCLeafProposal;
    readonly latent_block_coords: number | null;
    readonly paid_mix_z_weight: number;
    readonly paid_mix_pilot_weight: number;
    readonly paid_mix_pilot_var_scale: number;
    readonly paid_mix_wide_mult: number;
    readonly diagnostic_metrics_all: boolean;
    readonly diagnostic_metrics: readonly string[];
    readonly param_step_size: number;
    readonly param_step_size_min: number;
    readonly param_step_size_max: number;
    readonly param_target_accept: number;
    readonly adaptation_rate: number;
    readonly adaptation_scheme: "simple" | "dual_averaging";
    readonly init_method: "random" | "pathfinder";
    readonly latent_init_method: "predictive";
    readonly pathfinder_num_elbo_samples: number;
    readonly pathfinder_maxiter: number;
    readonly n_pathfinder_starts: number;
    readonly pathfinder_parallel_workers: number | null;
    readonly pathfinder_init_scale: number | null;
    readonly auto_preconditioner_method: "map" | "none" | "pathfinder";
    readonly auto_preconditioner_maxiter: number;
    readonly init_scale: number;
    readonly compute_latent_posterior_summary: boolean;
    readonly n_ieks_iters: number;
}
/**
 * Realized initialization and preconditioning, with the complete Pathfinder evidence once.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterWarmupDiagnostics".
 */
export interface ParameterWarmupDiagnostics {
    readonly pathfinder: PathfinderDiagnostics | null;
    readonly pathfinder_run_count: number;
    readonly pathfinder_consumers: readonly string[];
    readonly init_source: string;
    readonly preconditioner_source: string;
    readonly preconditioner_device: string | null;
    readonly dim: number;
    readonly duration_seconds: number;
    readonly pathfinder_sampling_mode: string | null;
    readonly pathfinder_init_scale: number | null;
    readonly prior_released_site_names: readonly string[];
    readonly prior_released_site_indices: readonly number[];
    readonly prior_release_scale: number;
}
/**
 * Retained native initialization measurements; never posterior evidence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PathfinderDiagnostics".
 */
export interface PathfinderDiagnostics {
    readonly n_pathfinder_starts: number;
    readonly n_pathfinder_starts_finite: number;
    readonly pathfinder_parallel_workers: number;
    readonly pathfinder_setup_seconds: number;
    readonly pathfinder_jax_compile_seconds: number;
    readonly pathfinder_jax_compile_batch_sizes: readonly number[];
    readonly pathfinder_runtime_seconds: number;
    readonly pathfinder_total_seconds: number;
    readonly best_pathfinder_elbo: number;
    readonly pathfinder_elbo_min: number;
    readonly pathfinder_elbo_max: number;
    readonly pathfinder_elbo_spread: number;
    readonly pathfinder_elbos: readonly number[];
    readonly pathfinder_maxiter: number;
    readonly pathfinder_lbfgs_memory: number;
    readonly pathfinder_elbo_samples: number;
    readonly pathfinder_elbo_screen_samples: number;
    readonly pathfinder_elbo_refine_candidates: number;
    readonly pathfinder_elbo_candidate_batch_size: number;
    readonly pathfinder_per_start: readonly PathfinderStartDiagnostics[];
}
/**
 * Retained native initialization measurements; never posterior evidence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PathfinderStartDiagnostics".
 */
export interface PathfinderStartDiagnostics {
    readonly n_elbo_batch_evaluations: number;
    readonly n_elbo_screen_candidates: number;
    readonly n_elbo_refine_candidates: number;
    readonly best_elbo_candidate_index: number;
    readonly start_idx: number;
    readonly n_trajectory_points: number;
    readonly n_valid_iterates: number;
    readonly n_elbo_candidates: number;
    readonly n_lbfgs_iterations: number;
    readonly final_log_posterior: number;
    readonly best_elbo_this_start: number | null;
    readonly scipy_success: boolean;
    readonly scipy_status: number;
}
/**
 * Recorded-chain criteria cover parameters, not latent-path mixing.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterConvergenceReport".
 */
export interface ParameterConvergenceReport {
    readonly scope: "recorded_parameter_chains";
    readonly assessments: readonly Assessment<ConvergenceAssessmentSubject, NumericCriterionEvidence>[];
    readonly checked: number;
    readonly status: "passed" | "failed" | "not_evaluated";
    readonly messages: readonly string[];
}
/**
 * Exact-emission leave-one-measurement-row-out interpolation diagnostics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LOODiagnostics".
 */
export interface LOODiagnostics {
    readonly elpd_loo: number;
    readonly p_loo: number;
    readonly se: number;
    readonly n_data_points: number;
    readonly observation_unit: "measurement_row";
    readonly prediction_task: "interpolation_given_other_measurements";
    readonly likelihood_source: "exact_emission_on_joint_particle_draws";
    readonly n_bad_k: number | null;
    readonly n_warn_k: number | null;
    readonly pareto_warning_limit: number;
    readonly pareto_failure_limit: number;
}
/**
 * One parameter's posterior interval, scale and density plot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorMarginal".
 */
export interface PosteriorMarginal {
    readonly parameter: string;
    readonly subject: ParameterRef;
    readonly density_curve: DensityCurve;
    readonly mean: number;
    readonly lower: number;
    readonly upper: number;
    readonly interval_kind: "hdi" | "equal_tail";
    readonly interval_mass: number;
    readonly sd: number;
}
/**
 * The indicator and criterion remain present when evaluation is unavailable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorCheckSubject".
 */
export interface IndicatorCheckSubject {
    readonly target: IndicatorRef;
    readonly check: "calibration" | "autocorrelation" | "variance";
}
/**
 * The compact core composed with retained detail, without filtering or re-parsing.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceReport".
 */
export interface InferenceReport {
    readonly core: InferenceReportCore;
    readonly detail: InferenceReportDetail;
}
/**
 * Retained plot series served in full by the report endpoint.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceReportDetail".
 */
export interface InferenceReportDetail {
    readonly tempering: TemperingDiagnostics | null;
    readonly trace_data: readonly TraceSeries[];
    readonly rank_histograms: readonly RankHistogram[];
    readonly pareto_k: readonly ParetoKPoint[];
    readonly loo_pit: readonly LOOPITPoint[];
    readonly posterior_pairs: readonly (readonly [
        ParameterRef,
        ParameterRef
    ])[];
    readonly divergent: readonly boolean[] | null;
    readonly initial_latent_delta: readonly (readonly number[])[] | null;
    readonly final_latent_delta: readonly (readonly number[])[] | null;
}
/**
 * Retained tempering telemetry, separate from evidence of a production engine.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TemperingDiagnostics".
 */
export interface TemperingDiagnostics {
    readonly n_levels: number;
    readonly n_particles: number;
    readonly accept_rates: readonly number[];
    readonly beta_schedule: readonly number[];
    readonly ess_history: readonly number[];
}
/**
 * Every retained draw, grouped in original chain order.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceSeries".
 */
export interface TraceSeries {
    readonly parameter: string;
    readonly subject: ParameterRef;
    readonly chains: readonly (readonly number[])[];
}
/**
 * Pooled-rank bin counts, grouped in original chain order.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RankHistogram".
 */
export interface RankHistogram {
    readonly parameter: string;
    readonly subject: ParameterRef;
    readonly n_bins: number;
    readonly expected_per_bin: number;
    readonly chains: readonly (readonly number[])[];
}
/**
 * One PSIS influence measurement with its original row and scientific class.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParetoKPoint".
 */
export interface ParetoKPoint {
    readonly rank: number;
    readonly timestep: number;
    readonly k: number | ("infinity" | "-infinity" | "undefined");
    readonly status: "passed" | "warning" | "failed" | "not_evaluated";
}
/**
 * A retained PIT value and its empirical and reference cumulative probabilities.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LOOPITPoint".
 */
export interface LOOPITPoint {
    readonly pit: number;
    readonly ecdf: number;
    readonly uniform: number;
}
/**
 * An LLM trace records a conversation, its model, elapsed time, and token usage.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LLMTrace".
 */
export interface LLMTrace {
    readonly messages: readonly TraceMessage[];
    readonly model: string;
    readonly total_time_seconds: number;
    readonly usage: TraceUsage;
}
/**
 * A trace message records one conversational step, including any reasoning or tool
 * interaction.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceMessage".
 */
export interface TraceMessage {
    readonly role: string;
    readonly content: string;
    readonly reasoning: string | null;
    readonly tool_calls: readonly TraceToolCall[] | null;
    readonly tool_call_id: string | null;
    readonly tool_name: string | null;
    readonly tool_result: string | null;
    readonly tool_is_error: boolean;
}
/**
 * A function invocation with the call identity used to match its result.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceToolCall".
 */
export interface TraceToolCall {
    readonly id: string;
    readonly type: "function";
    readonly name: string;
    readonly arguments: string;
}
/**
 * Trace usage records the input, output, and reasoning tokens consumed by a conversation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceUsage".
 */
export interface TraceUsage {
    readonly input_tokens: number;
    readonly output_tokens: number;
    readonly reasoning_tokens: number | null;
}
/**
 * Counts and representative observations read directly from one panel revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementsData".
 */
export interface MeasurementsData {
    readonly n_observations: number;
    readonly per_indicator_counts: Readonly<Partial<Record<IndicatorId, number>>>;
    readonly combined_extractions_sample: readonly ObservationRecord[];
}
/**
 * Canonical serialized extraction observation row.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationRecord".
 */
export interface ObservationRecord {
    readonly indicator_id: IndicatorId;
    readonly value: string | number | boolean | null;
    readonly anchor_time: string | null;
    readonly support_kind: string | null;
    readonly summary_operator: string | null;
    readonly anchor_policy: string | null;
    readonly observation_window: string | null;
    readonly support_start: string | null;
    readonly support_end: string | null;
}
/**
 * Exact conditional drift contributions, not marginal or total causal effects.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismCurves".
 */
export interface MechanismCurves {
    readonly axis: ConstructId;
    readonly axis_label: string;
    readonly target_label: string;
    readonly states: Readonly<Partial<Record<ConstructId, string>>>;
    readonly held: Readonly<Partial<Record<ConstructId, number>>>;
    readonly moderator: ConstructId | null;
    readonly x: readonly number[];
    readonly curves: readonly ResponseCurve[];
    readonly law: "retained" | "sampled" | "fixed";
    readonly total_draws: number;
    readonly start: number;
    readonly count: number;
    readonly nonfinite: number;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ResponseCurve".
 */
export interface ResponseCurve {
    readonly draw: number;
    readonly values: readonly (number | null)[];
    readonly level: number | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismViewRequest".
 */
export interface MechanismViewRequest {
    readonly owner_id: string;
    readonly axis: ConstructId | null;
    readonly lower: number;
    readonly upper: number;
    readonly held: Readonly<Partial<Record<ConstructId, number>>>;
    readonly moderator: ConstructId | null;
    /**
     * @minItems 1
     * @maxItems 5
     */
    readonly levels: readonly [
        number
    ] | readonly [
        number,
        number
    ] | readonly [
        number,
        number,
        number
    ] | readonly [
        number,
        number,
        number,
        number
    ] | readonly [
        number,
        number,
        number,
        number,
        number
    ];
    readonly start: number;
    readonly count: number;
    readonly points: number;
}
/**
 * A model diff joins typed entity comparisons and evidence at two model revisions or checkpoints.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelDiffReport".
 */
export interface ModelDiffReport {
    readonly before: GitRef;
    readonly after: GitRef;
    readonly parameters: readonly Change<ParameterSpec>[];
    readonly constructs: readonly (Change<ConstructRef> | Unchanged<ConstructRef>)[];
    readonly edges: readonly (Change<EdgeRef> | Unchanged<EdgeRef>)[];
    readonly before_dispositions: readonly StructuralItemDisposition[];
    readonly after_dispositions: readonly StructuralItemDisposition[];
    readonly before_dynamic_construct_ids: readonly ConstructId[];
    readonly after_dynamic_construct_ids: readonly ConstructId[];
    readonly changed_inputs: readonly string[];
    readonly before_checks: readonly SpecificationAssessment[];
    readonly after_checks: readonly SpecificationAssessment[];
    readonly before_fit: InferenceReportCore | null;
    readonly after_fit: InferenceReportCore | null;
    readonly before_simulation: SimulationReport | null;
    readonly after_simulation: SimulationReport | null;
}
/**
 * An item disposition explains the compilation decision for one identified authored
 * entity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralItemDisposition".
 */
export interface StructuralItemDisposition {
    readonly target: ConstructRef | EdgeRef | IndicatorRef;
    readonly disposition: StructuralDisposition;
    readonly reason: string;
}
/**
 * A simulation report records forward histories, resolved execution settings, and certified effects when supported.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationReport".
 */
export interface SimulationReport {
    readonly model: GitRef;
    readonly design: SimulationSpec;
    /**
     * Calendar instant of model day zero.
     */
    readonly time_origin: string;
    /**
     * The design's interventions in model days.
     */
    readonly assignments: readonly StateAssignment[];
    /**
     * @minItems 2
     */
    readonly times: readonly [
        number,
        number,
        ...readonly number[]
    ];
    readonly draws: number;
    readonly seed: number;
    /**
     * Panel that supplied the time origin: the fit's panel for fitted laws, otherwise the current panel when present.
     */
    readonly origin_panel_revision: GitOid | null;
    readonly state_ids: readonly ConstructId[];
    readonly parameter_draws: {
        readonly [k: string]: string | undefined;
    };
    readonly latent_paths: string;
    readonly observations: string;
    readonly observation_layout: SimulationObservationLayout;
    readonly law: PredictiveLawProvenance | null;
    readonly reference_latent_paths: string | null;
    readonly reference_observations: string | null;
    readonly findings: readonly PredictiveAssessment[];
    readonly fit_reliability: FitReliability;
    readonly causal: Evaluation<CausalEffectResult>;
}
/**
 * A state set at one model time: a resolved intervention or a replayed input reading.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StateAssignment".
 */
export interface StateAssignment {
    readonly target: ConstructId;
    /**
     * Absolute time in model days.
     */
    readonly time: number;
    readonly value: number;
}
/**
 * Saved observation semantics and coordinates; generation truths remain separate.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationObservationLayout".
 */
export interface SimulationObservationLayout {
    readonly variables: readonly ObservationSpec<string>[];
    readonly support_start_times: string;
    readonly support_end_times: string;
    readonly mask: string;
}
/**
 * One retained fit report, with the exact inputs and truthful retention state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelFitResult".
 */
export interface ModelFitResult {
    readonly action: "fit";
    readonly model: GitRef;
    readonly panel: GitRef;
    readonly report: InferenceReport;
    readonly retention: "joint" | "report_only";
}
/**
 * Scientific entity identities selected for the graph at this authoring checkpoint.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelGraphView".
 */
export interface ModelGraphView {
    readonly construct_ids: readonly ConstructId[];
    readonly edge_ids: readonly EdgeId[];
    readonly dynamic_construct_ids: readonly ConstructId[];
    readonly status: Readonly<Partial<Record<ConstructId, ("observed" | "marginalized" | "blocking")>>>;
}
/**
 * The report owns its model reference; the selected panel is separately pinned.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSimulationResult".
 */
export interface ModelSimulationResult {
    readonly action: "simulate";
    readonly panel: GitRef | null;
    readonly report: SimulationReport;
}
/**
 * The canonical scientific definition with independently sourced inputs and findings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSnapshot".
 */
export interface ModelSnapshot {
    readonly question: Sourced<QuestionSpec> | null;
    readonly model: Sourced<ModelSpec> | null;
    readonly workspace_id: string;
    readonly branch: string;
    readonly commit_id: GitOid;
    readonly selected_seq: number;
    readonly can_simulate: boolean;
    readonly state: StudyState;
    readonly raw_data: Sourced<RawDataData> | null;
    readonly measurements: Sourced<MeasurementsData> | null;
    readonly metadata: Sourced<PreparedDataMetadata> | null;
    readonly profile: Sourced<DataProfileArtifact> | null;
    readonly identification: Sourced<IdentificationReport> | null;
    readonly dispositions: Sourced<readonly StructuralItemDisposition[]> | null;
    readonly graph: ModelGraphView;
    readonly entity_failures: {
        readonly [k: string]: readonly string[] | undefined;
    };
    readonly validation_report: Sourced<ValidationReportArtifact> | null;
    readonly confounder_equations: Readonly<Partial<Record<ConstructId, string>>>;
    readonly state_equations: Readonly<Partial<Record<ConstructId, string>>>;
    readonly observation_equations: Readonly<Partial<Record<IndicatorId, string>>>;
    readonly likelihood_diagnostics: Readonly<Partial<Record<IndicatorId, readonly HistogramBin[]>>>;
    readonly authoring_prior_densities: Readonly<Partial<Record<ParameterId, DensityCurve>>>;
    readonly fit: Sourced<FitSummary> | null;
    readonly specification: Sourced<readonly SpecificationAssessment[]> | null;
    readonly question_checks: Sourced<QuestionCheckReport> | null;
    readonly simulation: Sourced<SimulationReport> | null;
    readonly predictive: Sourced<ModelPredictiveReport> | null;
}
/**
 * Study state projects the artifact trees selected by one Git commit.
 *
 * ``current`` maps artifact id → the revision info that is *current* for the
 * study. Absent key = the artifact does not exist (either never produced,
 * or produced-when-nonempty semantics withheld it).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StudyState".
 */
export interface StudyState {
    readonly current: Readonly<Partial<Record<ArtifactId, ArtifactRecord>>>;
    readonly checks: ModelCheckReport | null;
}
/**
 * A measured scalar and the producer's numerical acceptance region.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NumericCriterionEvidence".
 */
export interface NumericCriterionEvidence {
    readonly criterion: string;
    readonly value: number;
    readonly lower: number | null;
    readonly upper: number | null;
    readonly lower_inclusive: boolean;
    readonly upper_inclusive: boolean;
    readonly note: string;
    readonly display_value: string;
    readonly band_label: string;
}
/**
 * All prepared observations, their true anchors and their measurement support.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationHistory".
 */
export interface ObservationHistory {
    readonly indicator_id: IndicatorId;
    readonly label: string;
    readonly times: readonly number[];
    readonly values: readonly (number | null)[];
    readonly support_start: readonly (number | null)[];
    readonly support_end: readonly (number | null)[];
    readonly time_origin: string | null;
    readonly levels: readonly string[] | null;
    readonly empirical: readonly EmpiricalPoint[];
}
/**
 * Whether the model defines the question's outcome as a measured, modeled course.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "OutcomeSubject".
 */
export interface OutcomeSubject {
    readonly check: "outcome";
    readonly outcome: ConstructRef;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterDrawColumn".
 */
export interface ParameterDrawColumn {
    readonly label: string;
    readonly subject: ParameterRef;
    readonly values: readonly number[];
    readonly empirical: readonly EmpiricalPoint[];
}
/**
 * The production particle-MCMC target and its exact latent transition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParticleMCMCEvidence".
 */
export interface ParticleMCMCEvidence {
    readonly engine: "marginal_particle_gibbs";
    readonly latent_transition: "euler_maruyama";
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PathSeries".
 */
export interface PathSeries {
    readonly label: string;
    readonly action: readonly RecordedPath[];
    readonly reference: readonly RecordedPath[];
    readonly levels: readonly string[] | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RecordedPath".
 */
export interface RecordedPath {
    readonly draw: number;
    readonly values: readonly (number | null)[];
}
/**
 * One named check and its stable target in a construct's scientific context.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PredictiveSubject".
 */
export interface PredictiveSubject {
    readonly check: string;
    readonly construct_id: ConstructId | null;
    readonly target: EntityRef | ("whole_model" | "observations");
}
/**
 * Prepare uploaded sources or a simulation replicate without a model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PrepareDataRequest".
 */
export interface PrepareDataRequest {
    readonly action: "prepare_data";
    readonly input: FilePreparationSpec | SimulationReplicateRef;
}
/**
 * A data-preparation step changed status.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StepEvent".
 */
export interface StepEvent {
    readonly attempt_id: string;
    readonly cursor: string;
    readonly event: "nof1-causal-lab.step";
    readonly step: ProgressStep;
    readonly status: StepStatus;
    readonly error: StepError | null;
}
/**
 * The error type and message of a failed step.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StepError".
 */
export interface StepError {
    readonly type: string;
    readonly message: string;
}
/**
 * One query's intervention target: defined, identified, or set inside the record.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QueryTargetSubject".
 */
export interface QueryTargetSubject {
    readonly check: "target" | "identification" | "range";
    readonly query: string;
    readonly target: ConstructRef;
}
/**
 * Whether the record supports one query's window.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QueryWindowSubject".
 */
export interface QueryWindowSubject {
    readonly check: "window";
    readonly query: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Raised".
 */
export interface Raised {
    readonly status: "raised";
    readonly error_type: string;
    readonly error_message: string;
    readonly details: readonly string[];
}
/**
 * A stored column's physical type and authored interpretation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataColumnDescription".
 */
export interface RawDataColumnDescription {
    readonly name: string;
    readonly dtype: string;
    readonly description: string;
}
/**
 * Profile and representative rows from one uploaded table revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataData".
 */
export interface RawDataData {
    readonly n_records: number;
    readonly n_columns: number;
    readonly date_range: RawDataDateRange | null;
    readonly sample: readonly {
        readonly [k: string]: (string | null) | undefined;
    }[];
    readonly column_descriptions: readonly RawDataColumnDescription[];
}
/**
 * Observed date bounds of the uploaded table, when it contains a date column.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataDateRange".
 */
export interface RawDataDateRange {
    readonly start: string;
    readonly end: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RecordDependency".
 */
export interface RecordDependency {
    readonly seq: number;
    readonly source_seq: number;
    readonly argument: string;
    /**
     * Only the action's checks read the output; its request did not name it.
     */
    readonly check: boolean;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Rejected".
 */
export interface Rejected {
    readonly status: "rejected";
    readonly reason: RejectionReason;
    readonly detail: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RunningAction".
 */
export interface RunningAction {
    readonly attempt_id: string;
    readonly action: ActionId;
    readonly branch: string;
    readonly messages: readonly ActionMessage[];
}
/**
 * Set the study question; it is the first action of every study.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SetQuestionRequest".
 */
export interface SetQuestionRequest {
    readonly action: "set_question";
    readonly question: QuestionSpec;
}
/**
 * Generate a dated window with optional interventions; compare saved data with data_diff.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulateRequest".
 */
export interface SimulateRequest {
    /**
     * Calendar day the window starts, at 00:00 UTC.
     */
    readonly start: string;
    /**
     * How long the window lasts, such as 9w or 61d.
     */
    readonly horizon: string;
    readonly interventions: readonly InterventionSpec[];
    readonly action: "simulate";
    readonly model_revision: GitOid;
}
/**
 * Contiguous pages of original draws, with every recorded time point intact.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationPaths".
 */
export interface SimulationPaths {
    readonly times: readonly number[];
    readonly time_origin: string | null;
    readonly total_draws: number;
    readonly start: number;
    readonly count: number;
    readonly states: Readonly<Partial<Record<ConstructId, PathSeries>>>;
    readonly indicators: Readonly<Partial<Record<IndicatorId, PathSeries>>>;
    readonly effect: PathSeries | null;
    readonly effect_summary: EffectSummary | null;
    readonly reference_mean: number | null;
    readonly manifest_effects: Readonly<Partial<Record<IndicatorId, number>>>;
    readonly action_category_probabilities: Readonly<Partial<Record<IndicatorId, CategoryProbabilitySummary>>>;
    readonly reference_category_probabilities: Readonly<Partial<Record<IndicatorId, CategoryProbabilitySummary>>>;
}
/**
 * Git publication wraps its already-owned record, without copying its fields.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StudyRevision".
 */
export interface StudyRevision {
    readonly commit_id: GitOid;
    readonly parent_ids: readonly GitOid[];
    readonly record: AttemptRecord;
}
/**
 * Study status reports committed artifacts, their freshness, actions, and any running one.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StudyStatus".
 */
export interface StudyStatus {
    readonly workspace_id: string;
    readonly branch: string;
    readonly commit_id: GitOid | null;
    readonly seq: number;
    readonly state: StudyState;
    readonly artifacts: readonly ArtifactFreshness[];
    readonly actions: readonly ScientificActionId[];
    readonly running: RunningAction | null;
}
/**
 * Typed attempt journal returned by the study read plane.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TimelineResponse".
 */
export interface TimelineResponse {
    readonly workspace_id: string;
    readonly attempts: readonly StudyRevision[];
    readonly branches: {
        readonly [k: string]: GitOid | undefined;
    };
    readonly dependencies: readonly RecordDependency[];
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Added".
 */
export interface Added<PayloadT> {
    readonly kind: "added";
    readonly after: PayloadT;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Applied".
 */
export interface Applied<ResultT> {
    readonly status: "applied";
    readonly result: ResultT;
    readonly effects: ActionEffects;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Attempt".
 */
export interface Attempt<ActionT, RequestT, ResultT> {
    readonly action: ActionT;
    readonly request: RequestT | null;
    readonly outcome: Applied<ResultT> | Rejected | Raised;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Available".
 */
export interface Available<PayloadT> {
    readonly kind: "available";
    readonly value: PayloadT;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BernoulliLogitsLawSpec".
 */
export interface BernoulliLogitsLawSpec<A> {
    readonly distribution: "BernoulliLogits";
    readonly logits: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BernoulliProbsLawSpec".
 */
export interface BernoulliProbsLawSpec<A> {
    readonly distribution: "BernoulliProbs";
    readonly probs: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BetaLawSpec".
 */
export interface BetaLawSpec<A> {
    readonly distribution: "Beta";
    readonly concentration1: A;
    readonly concentration0: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CategoricalLawSpec".
 */
export interface CategoricalLawSpec<A> {
    readonly distribution: "Categorical";
    readonly logits: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DeltaLawSpec".
 */
export interface DeltaLawSpec<A> {
    readonly distribution: "Delta";
    readonly v: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Evaluated".
 */
export interface Evaluated<Subject, Evidence> {
    readonly kind: "evaluated";
    readonly subject: Subject;
    readonly outcome: "passed" | "failed" | "warning" | "error";
    readonly evidence: Evidence;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "GammaLawSpec".
 */
export interface GammaLawSpec<A> {
    readonly distribution: "Gamma";
    readonly concentration: A;
    readonly rate: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NegativeBinomial2LawSpec".
 */
export interface NegativeBinomial2LawSpec<A> {
    readonly distribution: "NegativeBinomial2";
    readonly mean: A;
    readonly concentration: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NormalLawSpec".
 */
export interface NormalLawSpec<A> {
    readonly distribution: "Normal";
    readonly loc: A;
    readonly scale: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NotEvaluated".
 */
export interface NotEvaluated<Subject> {
    readonly kind: "not_evaluated";
    readonly subject: Subject;
    readonly reason: NotEvaluatedReason;
    readonly detail: string;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationSpec".
 */
export interface ObservationSpec<WindowT> {
    readonly id: IndicatorId;
    /**
     * Indicator name (e.g., 'hrv', 'self_reported_stress')
     */
    readonly name: string;
    readonly measurement_dtype: MeasurementDtype;
    readonly aggregation: SummaryOperator;
    /**
     * Optional duration string describing the support window summarized by this indicator, in positive fixed units s, m, h, d or w (for example '2w'). Resolved by the preparation window or the generative model clock.
     */
    readonly observation_window: WindowT;
    /**
     * Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
     */
    readonly ordinal_levels: readonly string[] | null;
    /**
     * Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
     */
    readonly categorical_levels: readonly string[] | null;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "OrderedLogisticLawSpec".
 */
export interface OrderedLogisticLawSpec<A> {
    readonly distribution: "OrderedLogistic";
    readonly predictor: A;
    readonly cutpoints: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PoissonLawSpec".
 */
export interface PoissonLawSpec<A> {
    readonly distribution: "Poisson";
    readonly rate: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Removed".
 */
export interface Removed<PayloadT> {
    readonly kind: "removed";
    readonly before: PayloadT;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Revised".
 */
export interface Revised<PayloadT> {
    readonly kind: "revised";
    readonly before: PayloadT;
    readonly after: PayloadT;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced".
 */
export interface Sourced<T> {
    readonly value: T;
    readonly source: FactSource;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StudentTLawSpec".
 */
export interface StudentTLawSpec<A> {
    readonly distribution: "StudentT";
    readonly df: A;
    readonly loc: A;
    readonly scale: A;
}
/**
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Unchanged".
 */
export interface Unchanged<PayloadT> {
    readonly kind: "unchanged";
    readonly before: PayloadT;
    readonly after: PayloadT;
}
