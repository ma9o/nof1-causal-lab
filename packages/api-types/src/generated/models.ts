/** AUTO-GENERATED from Python's OpenAPI graph. Run bun run codegen. */
import type { components } from "./model-api";
export type ActionAttempt = components["schemas"]["ActionAttempt"];
export type ActionEffects = components["schemas"]["ActionEffects"];
export type ActionId = components["schemas"]["ActionId"];
export type ActionMessage = components["schemas"]["ActionMessage"];
export type ActionPoll = components["schemas"]["ActionPoll"];
export type ArtifactId = components["schemas"]["ArtifactId"];
export type ArtifactRecord = components["schemas"]["ArtifactRecord"];
export type AttemptRecord = components["schemas"]["AttemptRecord"];
export type AuthoredLawProvenance = components["schemas"]["AuthoredLawProvenance"];
export type BinaryExpression = components["schemas"]["BinaryExpression-Output"];
export type BinaryOperator = components["schemas"]["BinaryOperator"];
export type CallExpression = components["schemas"]["CallExpression-Output"];
export type CategoryProbabilitySummary = components["schemas"]["CategoryProbabilitySummary"];
export type CausalEdgeSpec = components["schemas"]["CausalEdgeSpec-Output"];
export type CausalEffectResult = components["schemas"]["CausalEffectResult"];
export type ChainDiagnostics = components["schemas"]["ChainDiagnostics"];
export type CheckGroup = components["schemas"]["CheckGroup"];
export type CoefficientExpression = components["schemas"]["CoefficientExpression-Output"];
export type CoefficientRole = components["schemas"]["CoefficientRole"];
export type CompletedExtractionWorker = components["schemas"]["CompletedExtractionWorker"];
export type CompletedPoll = components["schemas"]["CompletedPoll"];
export type ComputedExtractionSpec = components["schemas"]["ComputedExtractionSpec-Output"];
export type ConstructId = components["schemas"]["ConstructId-Output"];
export type ConstructRef = components["schemas"]["ConstructRef-Output"];
export type ConstructSpec = components["schemas"]["ConstructSpec-Output"];
export type ConvergenceAssessmentSubject = components["schemas"]["ConvergenceAssessmentSubject"];
export type ConvergenceCriterion = components["schemas"]["ConvergenceCriterion"];
export type ConvergenceSubject = components["schemas"]["ConvergenceSubject"];
export type DSMCLeafProposal = components["schemas"]["DSMCLeafProposal"];
export type DataDiffReport = components["schemas"]["DataDiffReport"];
export type DataDiffRequest = components["schemas"]["DataDiffRequest-Output"];
export type DataPoint = components["schemas"]["DataPoint"];
export type DataPreparationResult = components["schemas"]["DataPreparationResult"];
export type DataPreparationSpec = components["schemas"]["DataPreparationSpec-Output"];
export type DataProfileArtifact = components["schemas"]["DataProfileArtifact"];
export type DataRef = components["schemas"]["DataRef-Output"];
export type DataSelection = components["schemas"]["DataSelection-Output"];
export type DataSeries = components["schemas"]["DataSeries"];
export type DataStatistic = components["schemas"]["DataStatistic"];
export type DataStatisticComparison = components["schemas"]["DataStatisticComparison"];
export type DataVariableDiff = components["schemas"]["DataVariableDiff"];
export type DataVariableSpec = components["schemas"]["DataVariableSpec-Output"];
export type DensityCurve = components["schemas"]["DensityCurve"];
export type DistributionId = components["schemas"]["DistributionId-Output"];
export type DriftMechanismSpec = components["schemas"]["DriftMechanismSpec-Output"];
export type DynamicsMechanismSpec = components["schemas"]["DynamicsMechanismSpec-Output"];
export type EdgeId = components["schemas"]["EdgeId-Output"];
export type EdgeRef = components["schemas"]["EdgeRef"];
export type EditModelRequest = components["schemas"]["EditModelRequest-Output"];
export type EffectSummary = components["schemas"]["EffectSummary"];
export type EmpiricalPoint = components["schemas"]["EmpiricalPoint"];
export type EnergyDiagnostics = components["schemas"]["EnergyDiagnostics"];
export type EntityRef = components["schemas"]["EntityRef"];
export type EvaluatedPredictiveChecks = components["schemas"]["EvaluatedPredictiveChecks"];
export type Expression = components["schemas"]["Expression-Output"];
export type ExpressionFunction = components["schemas"]["ExpressionFunction"];
export type ExtractionPlanEvent = components["schemas"]["ExtractionPlanEvent"];
export type ExtractionSnapshotEvent = components["schemas"]["ExtractionSnapshotEvent"];
export type ExtractionSpec = components["schemas"]["ExtractionSpec-Output"];
export type ExtractionWorkerEvent = components["schemas"]["ExtractionWorkerEvent"];
export type ExtractionWorkerResult = components["schemas"]["ExtractionWorkerResult"];
export type FactSource = components["schemas"]["FactSource"];
export type FailedExtractionChunk = components["schemas"]["FailedExtractionChunk"];
export type FilePreparationSpec = components["schemas"]["FilePreparationSpec-Output"];
export type FilePreparedDataMetadata = components["schemas"]["FilePreparedDataMetadata"];
export type FileSourceRef = components["schemas"]["FileSourceRef-Output"];
export type FitReliability = components["schemas"]["FitReliability"];
export type FitRequest = components["schemas"]["FitRequest-Output"];
export type FitSettingsSpec = components["schemas"]["FitSettingsSpec-Output"];
export type FitSummary = components["schemas"]["FitSummary"];
export type FittedLawProvenance = components["schemas"]["FittedLawProvenance"];
export type GitOid = components["schemas"]["GitOid-Output"];
export type GitRef = components["schemas"]["GitRef"];
export type HistogramBin = components["schemas"]["HistogramBin"];
export type IdentificationReport = components["schemas"]["IdentificationReport"];
export type IdentifiedTreatmentStatus = components["schemas"]["IdentifiedTreatmentStatus"];
export type IdentityTransformSpec = components["schemas"]["IdentityTransformSpec-Output"];
export type IndicatorAudit = components["schemas"]["IndicatorAudit"];
export type IndicatorCheck = components["schemas"]["IndicatorCheck"];
export type IndicatorCheckSubject = components["schemas"]["IndicatorCheckSubject"];
export type IndicatorEmpiricalProfile = components["schemas"]["IndicatorEmpiricalProfile"];
export type IndicatorId = components["schemas"]["IndicatorId-Output"];
export type IndicatorPolarity = components["schemas"]["IndicatorPolarity"];
export type IndicatorRef = components["schemas"]["IndicatorRef"];
export type IndicatorSpec = components["schemas"]["IndicatorSpec-Output"];
export type InferenceEvidence = components["schemas"]["InferenceEvidence"];
export type InferenceMetadata = components["schemas"]["InferenceMetadata"];
export type InferenceReport = components["schemas"]["InferenceReport"];
export type InferenceReportCore = components["schemas"]["InferenceReportCore"];
export type InferenceReportDetail = components["schemas"]["InferenceReportDetail"];
export type InitialCorrelationTransformSpec = components["schemas"]["InitialCorrelationTransformSpec-Output"];
export type IntervalEffectTransformSpec = components["schemas"]["IntervalEffectTransformSpec-Output"];
export type InterventionSpec = components["schemas"]["InterventionSpec-Output"];
export type JointLawLayout = components["schemas"]["JointLawLayout-Output"];
export type JsonArray = readonly JsonValue[];
export type JsonObject = {
    readonly [key: string]: JsonValue;
};
export type JsonScalar = boolean | number | string | null;
export type JsonValue = JsonScalar | JsonArray | JsonObject;
export type LLMTrace = components["schemas"]["LLMTrace"];
export type LOODiagnostics = components["schemas"]["LOODiagnostics"];
export type LOOPITPoint = components["schemas"]["LOOPITPoint"];
export type LikelihoodSpec = components["schemas"]["LikelihoodSpec-Output"];
export type LiteralExpression = components["schemas"]["LiteralExpression-Output"];
export type LiteratureSource = components["schemas"]["LiteratureSource-Output"];
export type MarginalParticleGibbsSpec = components["schemas"]["MarginalParticleGibbsSpec"];
export type MeasurementDtype = components["schemas"]["MeasurementDtype"];
export type MeasurementsData = components["schemas"]["MeasurementsData"];
export type MechanismId = components["schemas"]["MechanismId-Output"];
export type MechanismRef = components["schemas"]["MechanismRef"];
export type MixedLawProvenance = components["schemas"]["MixedLawProvenance"];
export type ModelCheckReport = components["schemas"]["ModelCheckReport"];
export type ModelDiffReport = components["schemas"]["ModelDiffReport"];
export type ModelDiffRequest = components["schemas"]["ModelDiffRequest-Output"];
export type ModelFitResult = components["schemas"]["ModelFitResult"];
export type ModelGraphView = components["schemas"]["ModelGraphView"];
export type ModelPredictiveEvaluation = components["schemas"]["ModelPredictiveEvaluation"];
export type ModelPredictiveReport = components["schemas"]["ModelPredictiveReport"];
export type ModelSimulationResult = components["schemas"]["ModelSimulationResult"];
export type ModelSnapshot = components["schemas"]["ModelSnapshot"];
export type ModelSpec = components["schemas"]["ModelSpec-Output"];
export type NonIdentifiableTreatmentStatus = components["schemas"]["NonIdentifiableTreatmentStatus"];
export type NotApplicable = components["schemas"]["NotApplicable"];
export type NotEvaluatedReason = components["schemas"]["NotEvaluatedReason"];
export type NumPyroDistribution = components["schemas"]["NumPyroDistribution-Output"];
export type NumericCriterionEvidence = components["schemas"]["NumericCriterionEvidence"];
export type ObservationHistory = components["schemas"]["ObservationHistory"];
export type ObservationLawSpec = components["schemas"]["ObservationLawSpec-Output"];
export type ObservationRecord = components["schemas"]["ObservationRecord"];
export type OutcomeSubject = components["schemas"]["OutcomeSubject"];
export type PPCOverlay = components["schemas"]["PPCOverlay"];
export type PPCTestStat = components["schemas"]["PPCTestStat"];
export type PanelRef = components["schemas"]["PanelRef-Output"];
export type ParameterConvergenceReport = components["schemas"]["ParameterConvergenceReport"];
export type ParameterDiagnostics = components["schemas"]["ParameterDiagnostics"];
export type ParameterDrawColumn = components["schemas"]["ParameterDrawColumn"];
export type ParameterDraws = components["schemas"]["ParameterDraws"];
export type ParameterElementId = components["schemas"]["ParameterElementId-Output"];
export type ParameterId = components["schemas"]["ParameterId-Output"];
export type ParameterRef = components["schemas"]["ParameterRef"];
export type ParameterSpec = components["schemas"]["ParameterSpec-Output"];
export type ParameterTransformSpec = components["schemas"]["ParameterTransformSpec-Output"];
export type ParameterWarmupDiagnostics = components["schemas"]["ParameterWarmupDiagnostics"];
export type ParetoKPoint = components["schemas"]["ParetoKPoint"];
export type ParticleMCMCEvidence = components["schemas"]["ParticleMCMCEvidence"];
export type ParticleSamplerDiagnostics = components["schemas"]["ParticleSamplerDiagnostics"];
export type PathSeries = components["schemas"]["PathSeries"];
export type PathfinderDiagnostics = components["schemas"]["PathfinderDiagnostics"];
export type PathfinderStartDiagnostics = components["schemas"]["PathfinderStartDiagnostics"];
export type PersistenceTransformSpec = components["schemas"]["PersistenceTransformSpec-Output"];
export type PosteriorMarginal = components["schemas"]["PosteriorMarginal"];
export type PosteriorPredictiveChecks = components["schemas"]["PosteriorPredictiveChecks"];
export type PotentialMechanismSpec = components["schemas"]["PotentialMechanismSpec-Output"];
export type PredictiveAssessment = components["schemas"]["PredictiveAssessment"];
export type PredictiveCheckReason = components["schemas"]["PredictiveCheckReason"];
export type PredictiveComparison = components["schemas"]["PredictiveComparison"];
export type PredictiveComparisonResult = components["schemas"]["PredictiveComparisonResult"];
export type PredictiveLawProvenance = components["schemas"]["PredictiveLawProvenance"];
export type PredictiveSubject = components["schemas"]["PredictiveSubject"];
export type PrepareDataRequest = components["schemas"]["PrepareDataRequest-Output"];
export type PreparedDataMetadata = components["schemas"]["PreparedDataMetadata"];
export type ProgressEvent = components["schemas"]["ProgressEvent"];
export type ProgressStep = components["schemas"]["ProgressStep"];
export type QueryName = components["schemas"]["QueryName"];
export type QueryTargetSubject = components["schemas"]["QueryTargetSubject"];
export type QueryWindowSubject = components["schemas"]["QueryWindowSubject"];
export type QuestionAssessment = components["schemas"]["QuestionAssessment"];
export type QuestionCheckReport = components["schemas"]["QuestionCheckReport"];
export type QuestionSpec = components["schemas"]["QuestionSpec-Output"];
export type QuestionSubject = components["schemas"]["QuestionSubject"];
export type Raised = components["schemas"]["Raised"];
export type RankHistogram = components["schemas"]["RankHistogram"];
export type RawDataColumnDescription = components["schemas"]["RawDataColumnDescription"];
export type RawDataData = components["schemas"]["RawDataData"];
export type RawDataDateRange = components["schemas"]["RawDataDateRange"];
export type RecordDependency = components["schemas"]["RecordDependency"];
export type RecordedPath = components["schemas"]["RecordedPath"];
export type Rejected = components["schemas"]["Rejected"];
export type RejectionReason = components["schemas"]["RejectionReason"];
export type RetractedArtifact = components["schemas"]["RetractedArtifact"];
export type Role = components["schemas"]["Role"];
export type RunningAction = components["schemas"]["RunningAction"];
export type RunningPoll = components["schemas"]["RunningPoll"];
export type SamplerSpec = components["schemas"]["SamplerSpec"];
export type ScientificActionId = components["schemas"]["ScientificActionId"];
export type ScientificActionRequest = components["schemas"]["ScientificActionRequest"];
export type SemanticExtractionSpec = components["schemas"]["SemanticExtractionSpec-Output"];
export type SetQuestionRequest = components["schemas"]["SetQuestionRequest-Output"];
export type SimulateRequest = components["schemas"]["SimulateRequest-Output"];
export type SimulationEvidence = components["schemas"]["SimulationEvidence"];
export type SimulationObservationLayout = components["schemas"]["SimulationObservationLayout"];
export type SimulationPaths = components["schemas"]["SimulationPaths"];
export type SimulationPreparedDataMetadata = components["schemas"]["SimulationPreparedDataMetadata"];
export type SimulationRef = components["schemas"]["SimulationRef-Output"];
export type SimulationReplicateRef = components["schemas"]["SimulationReplicateRef-Output"];
export type SimulationReport = components["schemas"]["SimulationReport"];
export type SimulationSpec = components["schemas"]["SimulationSpec-Output"];
export type SourceValidity = components["schemas"]["SourceValidity"];
export type SpecificationAssessment = components["schemas"]["SpecificationAssessment"];
export type StateAssignment = components["schemas"]["StateAssignment"];
export type StateExpression = components["schemas"]["StateExpression-Output"];
export type StepError = components["schemas"]["StepError"];
export type StepEvent = components["schemas"]["StepEvent"];
export type StepStatus = components["schemas"]["StepStatus"];
export type StructuralDisposition = components["schemas"]["StructuralDisposition"];
export type StructuralItemDisposition = components["schemas"]["StructuralItemDisposition"];
export type StudyRevision = components["schemas"]["StudyRevision"];
export type StudyState = components["schemas"]["StudyState"];
export type SummaryOperator = components["schemas"]["SummaryOperator"];
export type TemporalStatus = components["schemas"]["TemporalStatus"];
export type TimelineRecord = components["schemas"]["TimelineRecord"];
export type TimelineResponse = components["schemas"]["TimelineResponse"];
export type TimelineRevision = components["schemas"]["TimelineRevision"];
export type TraceMessage = components["schemas"]["TraceMessage"];
export type TraceSeries = components["schemas"]["TraceSeries"];
export type TraceToolCall = components["schemas"]["TraceToolCall"];
export type TraceUsage = components["schemas"]["TraceUsage"];
export type Unavailable = components["schemas"]["Unavailable"];
export type UnavailablePredictiveChecks = components["schemas"]["UnavailablePredictiveChecks"];
export type UnknownLawProvenance = components["schemas"]["UnknownLawProvenance"];
export type ValidationIssue = components["schemas"]["ValidationIssue"];
export type ValidationReportArtifact = components["schemas"]["ValidationReportArtifact"];
export type WindowExpression = components["schemas"]["WindowExpression"];
export type Added<PayloadT> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "added";
    /** After */
    readonly after: PayloadT;
};
export type Applied<ResultT> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly status: "applied";
    /** Result */
    readonly result: ResultT;
    readonly effects: ActionEffects;
};
export type Assessment<Subject, Evidence> = Evaluated<Subject, Evidence> | NotEvaluated<Subject>;
export type Attempt<ActionT, RequestT, ResultT> = {
    /** Action */
    readonly action: ActionT;
    /**
     * Request
     * @description Parsed arguments, or null for a historical attempt whose arguments were not retained
     */
    readonly request: RequestT | null;
    /** Outcome */
    readonly outcome: Applied<ResultT> | Rejected | Raised;
};
export type Available<PayloadT> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "available";
    /** Value */
    readonly value: PayloadT;
};
export type BernoulliLogitsLawSpec<A> = {
    /**
     * Distribution
     * @default BernoulliLogits
     * @constant
     */
    readonly distribution: "BernoulliLogits";
    /** Logits */
    readonly logits: A;
};
export type BernoulliProbsLawSpec<A> = {
    /**
     * Distribution
     * @default BernoulliProbs
     * @constant
     */
    readonly distribution: "BernoulliProbs";
    /** Probs */
    readonly probs: A;
};
export type BetaLawSpec<A> = {
    /**
     * Distribution
     * @default Beta
     * @constant
     */
    readonly distribution: "Beta";
    /** Concentration1 */
    readonly concentration1: A;
    /** Concentration0 */
    readonly concentration0: A;
};
export type CategoricalLawSpec<A> = {
    /**
     * Distribution
     * @default Categorical
     * @constant
     */
    readonly distribution: "Categorical";
    /** Logits */
    readonly logits: A;
};
export type Change<PayloadT> = Added<PayloadT> | Removed<PayloadT> | Revised<PayloadT>;
export type DeltaLawSpec<A> = {
    /**
     * Distribution
     * @default Delta
     * @constant
     */
    readonly distribution: "Delta";
    /** V */
    readonly v: A;
};
export type Evaluated<Subject, Evidence> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "evaluated";
    /** Subject */
    readonly subject: Subject;
    /**
     * Outcome
     * @enum {string}
     */
    readonly outcome: "passed" | "failed" | "warning" | "error";
    /** Evidence */
    readonly evidence: Evidence;
};
export type Evaluation<PayloadT> = Available<PayloadT> | Unavailable | NotApplicable;
export type GammaLawSpec<A> = {
    /**
     * Distribution
     * @default Gamma
     * @constant
     */
    readonly distribution: "Gamma";
    /** Concentration */
    readonly concentration: A;
    /** Rate */
    readonly rate: A;
};
export type NegativeBinomial2LawSpec<A> = {
    /**
     * Distribution
     * @default NegativeBinomial2
     * @constant
     */
    readonly distribution: "NegativeBinomial2";
    /** Mean */
    readonly mean: A;
    /** Concentration */
    readonly concentration: A;
};
export type NormalLawSpec<A> = {
    /**
     * Distribution
     * @default Normal
     * @constant
     */
    readonly distribution: "Normal";
    /** Loc */
    readonly loc: A;
    /** Scale */
    readonly scale: A;
};
export type NotEvaluated<Subject> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "not_evaluated";
    /** Subject */
    readonly subject: Subject;
    readonly reason: NotEvaluatedReason;
    /**
     * Detail
     * @default
     */
    readonly detail: string;
};
export type ObservationSpec<WindowT> = {
    /** @description Persistent identity. Preserve when revising or renaming. */
    readonly id: IndicatorId;
    /**
     * Name
     * @description Indicator name (e.g., 'hrv', 'self_reported_stress')
     */
    readonly name: string;
    /** @description 'continuous', 'binary', 'count', 'ordinal', 'categorical' */
    readonly measurement_dtype: MeasurementDtype;
    /** @description Aggregation function applied when bucketing raw extractions within the indicator support window. Supported operators: first, last, sum, count, mean, std. A computed_rule must produce this same summary. */
    readonly aggregation: SummaryOperator;
    /**
     * Observation Window
     * @description Optional duration string describing the support window summarized by this indicator, in positive fixed units s, m, h, d or w (for example '2w'). Resolved by the preparation window or the generative model clock.
     */
    readonly observation_window: WindowT;
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
export type OrderedLogisticLawSpec<A> = {
    /**
     * Distribution
     * @default OrderedLogistic
     * @constant
     */
    readonly distribution: "OrderedLogistic";
    /** Predictor */
    readonly predictor: A;
    /** Cutpoints */
    readonly cutpoints: A;
};
export type PoissonLawSpec<A> = {
    /**
     * Distribution
     * @default Poisson
     * @constant
     */
    readonly distribution: "Poisson";
    /** Rate */
    readonly rate: A;
};
export type Removed<PayloadT> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "removed";
    /** Before */
    readonly before: PayloadT;
};
export type Revised<PayloadT> = {
    /**
     * @description discriminator enum property added by openapi-typescript
     * @enum {string}
     */
    readonly kind: "revised";
    /** Before */
    readonly before: PayloadT;
    /** After */
    readonly after: PayloadT;
};
export type Sourced<T> = {
    /** Value */
    readonly value: T;
    readonly source: FactSource;
};
export type StudentTLawSpec<A> = {
    /**
     * Distribution
     * @default StudentT
     * @constant
     */
    readonly distribution: "StudentT";
    /** Df */
    readonly df: A;
    /** Loc */
    readonly loc: A;
    /** Scale */
    readonly scale: A;
};
export type Unchanged<PayloadT> = {
    /**
     * Kind
     * @default unchanged
     * @constant
     */
    readonly kind: "unchanged";
    /** Before */
    readonly before: PayloadT;
    /** After */
    readonly after: PayloadT;
};
