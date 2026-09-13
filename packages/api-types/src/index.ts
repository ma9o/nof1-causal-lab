// ---------------------------------------------------------------------------
// Hand-written (frontend-only) — not generated from Python
// ---------------------------------------------------------------------------

import type { ArtifactViewId } from "./transitions";

export type { ArtifactStatus, ArtifactViewState, PipelineRun, RunStatus } from "./run";
export type {
  ArtifactId,
  ArtifactViewId,
  TransitionId,
  TransitionLogScopePolicy,
  TransitionMeta,
} from "./transitions";
export { ARTIFACT_IDS, ARTIFACT_VIEW_IDS, TRANSITION_META, TRANSITIONS } from "./transitions";

// ---------------------------------------------------------------------------
// Generated from Python, using the same type names in both languages
// ---------------------------------------------------------------------------

// Persisted artifacts and their domain values
export type {
  AggregationFunction,
  ArtifactEnvelope,
  ArtifactFileSpec,
  ArtifactFreshness,
  ArtifactRef,
  ArtifactVersionInfo,
  AutoRunResponse,
  BaselineReportArtifact,
  BaselineReportVisualization,
  CapabilitiesResponse,
  CausalDesign,
  CausalEdge,
  CompiledParameterBinding,
  CompiledSSMArtifact,
  ConstantDriftMechanism,
  Construct,
  ConstructId,
  ConstructRef,
  DistributionFamily,
  DynamicsMechanism,
  EdgeId,
  EdgeRef,
  EffectSummary,
  EffectTrajectoryPoint,
  EnergyDiagnostics,
  EnergyHistogram,
  EpisodeState,
  EpisodeStatus,
  EstimatedCoefficient,
  EventsResponse,
  FactSource,
  FitSummary,
  FixedCoefficient,
  HillEdgeMechanism,
  IdentifiabilityStatus,
  IdentificationReport,
  IdentifiedTreatmentStatus,
  Indicator,
  IndicatorAudit,
  IndicatorEmpiricalProfile,
  IndicatorId,
  IndicatorRef,
  IndicatorValidation,
  InferenceMetadata,
  JournalStatus,
  JsonObject,
  JsonScalar,
  JsonValue,
  KnownInput,
  LatentClampInput,
  LatentStructure,
  LatentStructureArtifact,
  LikelihoodSpec,
  LinearEdgeMechanism,
  LinkFunction,
  LiteratureSource,
  LLMTrace,
  LOODiagnostics,
  MachineDescription,
  MCMCDiagnostics,
  MCMCParamDiagnostic,
  MeasurementDtype,
  MeasurementStructure,
  MeasurementStructureArtifact,
  MeasurementsArtifact,
  MechanismCoefficient,
  ModelRef,
  ModelSnapshot,
  Move,
  MoveOutcome,
  NodePotentialMechanism,
  NonIdentifiableTreatmentStatus,
  ParameterConstraint,
  ParameterRole,
  ParameterSpec,
  PosteriorArtifact,
  PosteriorAssessment,
  PosteriorDrawsInfo,
  PosteriorEstimate,
  PosteriorMarginal,
  PosteriorPair,
  PosteriorPredictiveChecks,
  PosteriorProvenance,
  PPCOverlay,
  PPCTestStat,
  PPCWarning,
  PriorDistributionFamily,
  PriorPredictiveDiagnostic,
  NumPyroDistribution,
  PriorSource,
  Provenance,
  QuestionArtifact,
  RankHistogram,
  RankHistogramChain,
  RawDataArtifact,
  ResumeRef,
  RetractedArtifact,
  Role,
  RuntimeEvent,
  SavedScenario,
  SavedScenariosArtifact,
  ScenarioEvaluation,
  ScenarioEvaluationId,
  ScenarioEvaluationResult,
  ScenarioQueryId,
  ScenarioResult,
  ScenarioStartResult,
  ScientificOnlyConstruct,
  SimulateScenarioInput,
  SimulateScenarioResult,
  SimulateScenarioToolResult,
  SMCDiagnostics,
  SourceValidity,
  StartEpisodeResponse,
  StatisticalModelSpec,
  StatisticalModelSpecArtifact,
  StructuralDisposition,
  StructuralEdge,
  StructuralInducedDependency,
  StructuralItemDisposition,
  StructuralKnownInput,
  StructuralPlan,
  StructuralSemanticCatalog,
  TemporalStatus,
  TimelineResponse,
  ToolError,
  TraceChain,
  TraceData,
  TraceMessage,
  TraceUsage,
  TransitionRecord,
  TransitionTraceIndex,
  TreatmentEffect,
  UploadResponse,
  ValidationIssue,
  ValidationReportArtifact,
  WindowExpression,
  WorkerStatus,
  WorkspaceEntry,
  WorkspaceList,
} from "./generated/models";

export type StatisticalModelSpecPersistedViewData =
  import("./generated/models").StatisticalModelSpecArtifact;
export type {
  ArtifactViews,
  EntityRef,
  HistogramBin,
  MeasurementStructureViewData,
  MeasurementsData,
  ModelSpecLikelihoodDiagnostics,
  ObservationRecord,
  ParameterCoordinate,
  RawDataColumnDescription,
  RawDataData,
  RawDataDateRange,
  ScenarioClamp,
  ScenarioQuery,
  StateEquation,
  StatisticalModelSpecData,
} from "./generated/models";

import type {
  MeasurementStructureViewData,
  MeasurementsData,
  RawDataData,
  StatisticalModelSpecData,
} from "./generated/models";

export interface ArtifactViewDataMap {
  raw_data: RawDataData;
  latent_structure: import("./generated/models").LatentStructureArtifact;
  measurement_structure: MeasurementStructureViewData;
  measurements: MeasurementsData;
  validation_report: import("./generated/models").ValidationReportArtifact;
  statistical_model_spec: StatisticalModelSpecData;
  posterior: import("./generated/models").PosteriorArtifact;
  baseline_report: import("./generated/models").BaselineReportArtifact;
}

export type ArtifactViewData<K extends ArtifactViewId = ArtifactViewId> = ArtifactViewDataMap[K];

// Distribution catalog metadata (codegen'd from Python)
export type { ObservationHyperparameter } from "./generated/metadata";
export {
  ARTIFACT_FILE_SPECS,
  MACHINE_DESCRIPTION,
  OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION,
} from "./generated/metadata";
// Tool definitions (codegen'd from Python ToolDefinition)
export type { ToolDefinition } from "./generated/tools";
export { CONTEXT_TOOLS, INTERACTIVE_CONTEXTS } from "./generated/tools";

export interface ArtifactData<T = unknown> {
  artifactId: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export { createModelClient } from "./client";
