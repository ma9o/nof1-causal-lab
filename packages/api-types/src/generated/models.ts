/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python Pydantic models via:
 *   cd apps/data-pipeline && uv run python -m scripts.export_schemas
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py
 * plus facade API models exported from apps/data-pipeline/src/nof1_causal_lab/episode_api.py
 */

/**
 * A persistent edge identity identifies one authored causal relationship.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeId".
 */
export type EdgeId = `edge:${string}`;
/**
 * A persistent mechanism identity distinguishes additive terms through reordering and revision.
 *
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
export type Expression =
  | LiteralExpression
  | StateExpression
  | CoefficientExpression
  | BinaryExpression
  | CallExpression;
/**
 * A persistent construct identity survives changes to its display name.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructId".
 */
export type ConstructId = `construct:${string}`;
/**
 * A coefficient role identifies an expression operand’s scientific quantity and support.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CoefficientRole".
 */
export type CoefficientRole =
  | "center"
  | "decay"
  | "quartic"
  | "intercept"
  | "weight"
  | "emax"
  | "ec50"
  | "exponent"
  | "loading"
  | "observation_intercept"
  | "observation_scale"
  | "degrees_of_freedom"
  | "shape"
  | "dispersion"
  | "concentration"
  | "cutpoint_base"
  | "cutpoint_gaps"
  | "category_intercepts"
  | "category_slopes";
/**
 * A component coefficient is a fixed value or a reference to a scientific parameter.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Coefficient".
 */
export type Coefficient = FixedCoefficient | ParameterCoefficient;
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
 * A persistent indicator identity survives changes to its measurement label.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorId".
 */
export type IndicatorId = `indicator:${string}`;
/**
 * Indicator polarity states whether a measurement increases or decreases with its
 * construct.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorPolarity".
 */
export type IndicatorPolarity = "positive" | "negative";
/**
 * A measurement dtype defines the observed value domain of an indicator.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementDtype".
 */
export type MeasurementDtype = "continuous" | "binary" | "count" | "ordinal" | "categorical";
/**
 * An aggregation function summarizes observations within a measurement window.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AggregationFunction".
 */
export type AggregationFunction =
  | "mean"
  | "sum"
  | "min"
  | "max"
  | "std"
  | "var"
  | "last"
  | "first"
  | "count"
  | "median"
  | "p10"
  | "p25"
  | "p75"
  | "p90"
  | "p99"
  | "skew"
  | "kurtosis"
  | "iqr"
  | "range"
  | "cv"
  | "entropy"
  | "instability"
  | "trend"
  | "n_unique";
/**
 * Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WindowExpression".
 */
export type WindowExpression = string;
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
export type JsonArray = JsonValue[];
/**
 * A shared native law whose membership is defined by the model's scientific quantities.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DistributionId".
 */
export type DistributionId = `distribution:${string}`;
/**
 * A construct usage declares a measured driver or a scientific-only variable excluded from executable states.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructUsage".
 */
export type ConstructUsage = KnownInput | ScientificOnlyConstruct;
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
 * An entity reference identifies a construct, edge, indicator, or mechanism by its persistent identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EntityRef".
 */
export type EntityRef = ConstructRef | EdgeRef | IndicatorRef | MechanismRef;
/**
 * An artifact identity selects one node in the machine's artifact graph.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactId".
 */
export type ArtifactId =
  | "question"
  | "raw_data"
  | "model"
  | "identification_report"
  | "panel"
  | "validation_report"
  | "admission_report"
  | "baseline_report";
/**
 * A machine move requests computation or an authored artifact write.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Move".
 */
export type Move = RunOperation | WriteArtifact;
/**
 * An operation identity selects an action independently of its output artifacts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "OperationId".
 */
export type OperationId =
  | "raw_data"
  | "latent_structure"
  | "measurement_structure"
  | "measurements"
  | "statistical_model_spec"
  | "posterior"
  | "baseline_report";
/**
 * Artifact provenance records whether its content was computed, authored by a human, or proposed by an LLM.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Provenance".
 */
export type Provenance = "computed" | "human" | "llm";
/**
 * One available artifact projection returned by the model view endpoint.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactViewResponse".
 */
export type ArtifactViewResponse =
  | RawDataData
  | ModelSpec
  | MeasurementsData
  | ValidationReportArtifact
  | AdmissionReport
  | ModelDiagnostics
  | InferenceReport
  | BaselineReportArtifact;
/**
 * A parameter element identity identifies a logical scalar component across model revisions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterElementId".
 */
export type ParameterElementId = `element:${string}`;
/**
 * A runtime event records transition progress, agent activity, or extraction telemetry.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RuntimeEvent".
 */
export type RuntimeEvent =
  | TransitionRuntimeEvent
  | ExtractionPlanEvent
  | ExtractionWorkerEvent
  | ExtractionSnapshotEvent
  | ModelSpecAdmissionEvent;
/**
 * Source validity records whether a fact still matches its pinned inputs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SourceValidity".
 */
export type SourceValidity = "fresh" | "stale";
/**
 * A journal status distinguishes applied revisions from rejected or failed attempts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JournalStatus".
 */
export type JournalStatus = "applied" | "rejected" | "raised";
/**
 * A structural disposition classifies how compilation uses or excludes an authored model
 * entity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralDisposition".
 */
export type StructuralDisposition =
  | "retained_state"
  | "known_input"
  | "marginalized"
  | "identification_only"
  | "retained_edge"
  | "projected_edge"
  | "manifest"
  | "known_input_source"
  | "excluded_indicator";
/**
 * A simulation tool response carries either a resolved scenario result or a reported tool error.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulateScenarioToolResult".
 */
export type SimulateScenarioToolResult = SimulationResult | ToolError;

/**
 * Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.
 */
export interface CausalSSMContracts {
  question: QuestionArtifact;
  model: ModelSpec;
  identification_report: IdentificationReport;
  validation_report: ValidationReportArtifact;
  admission_report: AdmissionReport;
  baseline_report: BaselineReportArtifact;
}
/**
 * A research question states the observational causal question under investigation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "QuestionArtifact".
 */
export interface QuestionArtifact {
  text: string;
}
/**
 * One connected causal graph whose endpoints and relationships gain scientific detail.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSpec".
 */
export interface ModelSpec {
  /**
   * @minItems 1
   */
  edges: [CausalEdge, ...CausalEdge[]];
  parameters: ParameterSpec[];
  /**
   * Shared joint laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. Inline scalar parameter laws apply independently to their elements.
   */
  distributions: {
    [k: string]: NumPyroDistribution;
  };
  time_points: number[];
  measurement_clock?: string | null;
  default_outcome?: ConstructRef | null;
}
/**
 * A causal edge declares a directed causal relationship between two constructs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalEdge".
 */
export interface CausalEdge {
  id: EdgeId;
  mechanisms: DynamicsMechanism[];
  /**
   * Cause construct; shared endpoints have one identity.
   */
  cause: Construct | ConstructRef;
  /**
   * Effect construct; shared endpoints have one identity.
   */
  effect: Construct | ConstructRef;
  /**
   * Theoretical justification for this causal link
   */
  description: string;
  /**
   * If True, effect at t is caused by cause at t-1 (one model_clock tick delay). If False (contemporaneous), effect at t is caused by cause at t.
   */
  lagged: boolean;
  /**
   * Literature sources supporting this causal link
   */
  sources: LiteratureSource[];
}
/**
 * A dynamics mechanism declares one contribution to continuous-time drift.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DynamicsMechanism".
 */
export interface DynamicsMechanism {
  id: MechanismId;
  kind: "drift" | "potential";
  expression: Expression;
}
/**
 * A finite scalar constant in a model equation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LiteralExpression".
 */
export interface LiteralExpression {
  kind: "literal";
  value: number;
}
/**
 * A construct's state or declared known input, referenced by identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StateExpression".
 */
export interface StateExpression {
  kind: "state";
  construct_id: ConstructId;
}
/**
 * A scientifically typed coefficient operand, literal or parameter reference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CoefficientExpression".
 */
export interface CoefficientExpression {
  kind: "coefficient";
  role: CoefficientRole;
  coefficient?: Coefficient | null;
}
/**
 * A coefficient held at a specified value on the continuous-time model scale.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FixedCoefficient".
 */
export interface FixedCoefficient {
  kind: "fixed";
  value: number;
}
/**
 * A slot referencing a scientific parameter, whether fixed or estimated.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterCoefficient".
 */
export interface ParameterCoefficient {
  kind: "parameter";
  parameter_id: ParameterId;
}
/**
 * A supported scalar operation composing two expressions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BinaryExpression".
 */
export interface BinaryExpression {
  kind: "binary";
  operator: BinaryOperator;
  left: Expression;
  right: Expression;
}
/**
 * A supported mathematical function, including explicit discrete contrasts.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CallExpression".
 */
export interface CallExpression {
  kind: "call";
  function: ExpressionFunction;
  arguments: Expression[];
}
/**
 * A construct represents a theoretical entity in the scientific causal model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Construct".
 */
export interface Construct {
  id: ConstructId;
  /**
   * Construct name (e.g., 'stress', 'sleep_quality')
   */
  name: string;
  /**
   * What this theoretical construct represents
   */
  description: string;
  indicators: Indicator[];
  dynamics: DynamicsMechanism[];
  innovation?: InnovationSpec | null;
  initial_state?: InitialStateSpec | null;
  /**
   * Law of this construct's trajectory on ModelSpec.time_points; may be joint.
   */
  distribution?: NumPyroDistribution | DistributionId | null;
  usage?: ConstructUsage | null;
  role: Role;
  temporal_status: TemporalStatus;
}
/**
 * An indicator defines an observed measurement of a construct and how to extract it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Indicator".
 */
export interface Indicator {
  id: IndicatorId;
  likelihood?: LikelihoodSpec | null;
  /**
   * Indicator name (e.g., 'hrv', 'self_reported_stress')
   */
  name: string;
  /**
   * Instructions for workers on how to extract this from data
   */
  how_to_measure: string;
  construct_polarity: IndicatorPolarity;
  measurement_dtype: MeasurementDtype;
  aggregation: AggregationFunction;
  /**
   * Optional duration string describing the support window summarized by this indicator (for example '1mo' for a monthly average on a daily model clock). If omitted, the support window defaults to the global model_clock.
   */
  observation_window?: string | null;
  /**
   * Ordered list of level labels from lowest to highest for ordinal indicators (e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' to ensure correct numeric encoding.
   */
  ordinal_levels?: string[] | null;
  /**
   * Exhaustive list of level labels for categorical indicators (e.g., ['home', 'work', 'other']). Required when measurement_dtype='categorical' to ensure correct numeric encoding.
   */
  categorical_levels?: string[] | null;
  /**
   * Raw data column names referenced by how_to_measure. Used to project chunks to only relevant columns before extraction.
   */
  source_columns: string[];
  /**
   * Optional deterministic support-window expression for extraction_mode='computed'. Use this when a computed indicator needs formulas, thresholds, or multiple source columns instead of a direct single-column aggregation. The expression must return one scalar per support window.
   */
  computed_rule?: WindowExpression | null;
  /**
   * 'computed' (deterministic pipeline extraction) or 'semantic' (LLM extraction). Use 'computed' when the indicator can be derived deterministically either from a direct source-column aggregation or from a computed_rule support-window expression over the declared source_columns.
   */
  extraction_mode: "computed" | "semantic";
}
/**
 * An indicator's conditional probability law and its scientific justification.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodSpec".
 */
export interface LikelihoodSpec {
  law: ObservationLaw;
  /**
   * Whether observations are mean-centered and scaled before fitting.
   */
  standardized: boolean;
  /**
   * Why this conditional law was chosen for the indicator
   */
  reasoning: string;
  sources: LiteratureSource[];
}
/**
 * A native probability constructor applied to model-dependent expressions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationLaw".
 */
export interface ObservationLaw {
  distribution:
    | "Delta"
    | "Normal"
    | "StudentT"
    | "Poisson"
    | "Gamma"
    | "Bernoulli"
    | "NegativeBinomial2"
    | "Beta"
    | "OrderedLogistic"
    | "Categorical";
  arguments: {
    [k: string]: Expression;
  };
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
  title: string;
  /**
   * URL of the source if available
   */
  url?: string | null;
  /**
   * Relevant excerpt or paraphrase from the source
   */
  snippet: string;
}
/**
 * Continuous-time driving noise, including conditional loadings from common causes.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InnovationSpec".
 */
export interface InnovationSpec {
  distribution: "gaussian" | "student_t";
  scale: Coefficient;
  loadings: StateCoupling[];
  degrees_of_freedom?: Coefficient | null;
}
/**
 * A coefficient connecting an owned component to another construct.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StateCoupling".
 */
export interface StateCoupling {
  other_id: ConstructId;
  coefficient: Coefficient;
}
/**
 * Initial location, marginal scale and correlations, including shared baseline factors.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InitialStateSpec".
 */
export interface InitialStateSpec {
  mean: Coefficient;
  scale: Coefficient;
  correlations: StateCoupling[];
}
/**
 * A native NumPyro probability distribution serialized by its constructor tree.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NumPyroDistribution".
 */
export interface NumPyroDistribution {
  distribution: string;
  params: {
    [k: string]: JsonValue;
  };
}
/**
 * A JSON object transports string-keyed recursively typed values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "JsonObject".
 */
export interface JsonObject {
  [k: string]: JsonValue;
}
/**
 * An observed-input declaration binds a construct to its measured driver trajectory.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "KnownInput".
 */
export interface KnownInput {
  kind: "known_input";
  source_indicator_id: IndicatorId;
  /**
   * Positive divisor applied before inference
   */
  scale: number;
  missing_policy: "zero" | "forward_fill";
}
/**
 * A scientific-only declaration excludes an identified construct from executable states.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScientificOnlyConstruct".
 */
export interface ScientificOnlyConstruct {
  kind: "scientific_only";
  reason: string;
}
/**
 * A construct reference identifies a construct independently of its current name or
 * revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructRef".
 */
export interface ConstructRef {
  kind: "construct";
  id: ConstructId;
}
/**
 * A named quantity's current uncertainty; component slots define its meaning.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterSpec".
 */
export interface ParameterSpec {
  id: ParameterId;
  /**
   * Authored parameter label; relationships use its persistent ID
   */
  name: string;
  /**
   * Human-readable description of what this parameter represents
   */
  description: string;
  distribution_transform:
    | "identity"
    | "dt_persistence_to_ct_decay"
    | "dt_effect_to_ct_rate"
    | "initial_state_correlation";
  /**
   * Known constant on the model quantity scale, exclusive with a distribution.
   */
  value?: number | null;
  /**
   * Native law or a reference to a shared joint law in ModelSpec.distributions.
   */
  distribution?: NumPyroDistribution | DistributionId | null;
  reference_interval_days?: number | null;
}
/**
 * Positive and negative causal identification findings for the model's default query.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentificationReport".
 */
export interface IdentificationReport {
  outcome: ConstructId | null;
  status: IdentifiabilityStatus;
}
/**
 * Status of causal effect identifiability.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentifiabilityStatus".
 */
export interface IdentifiabilityStatus {
  /**
   * Treatment IDs mapped to their identification strategy and supporting construct IDs
   */
  identifiable_treatments: {
    [k: string]: IdentifiedTreatmentStatus;
  };
  /**
   * Treatment IDs mapped to the construct IDs blocking identification
   */
  non_identifiable_treatments: {
    [k: string]: NonIdentifiableTreatmentStatus;
  };
}
/**
 * Details on how a treatment effect is identified.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentifiedTreatmentStatus".
 */
export interface IdentifiedTreatmentStatus {
  status: "identified";
  /**
   * Nonparametric identification; linear-IV arguments do not certify ModelSpec.
   */
  method: "do_calculus";
  /**
   * Nonparametric estimand returned by do-calculus
   */
  estimand: string;
  /**
   * Unobserved confounders the estimand integrates out
   */
  marginalized_confounders: ConstructId[];
  /**
   * Instrument constructs appearing in the nonparametric identification argument
   */
  instruments: ConstructId[];
}
/**
 * Context on why a treatment effect is not identifiable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NonIdentifiableTreatmentStatus".
 */
export interface NonIdentifiableTreatmentStatus {
  status: "not_identified";
  /**
   * Unobserved constructs blocking identification
   */
  confounders: ConstructId[];
  /**
   * Optional explanation if confounders cannot be enumerated
   */
  notes?: string | null;
}
/**
 * A validation report summarizes whether extracted measurements satisfy the required data
 * checks.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationReportArtifact".
 */
export interface ValidationReportArtifact {
  is_valid: boolean;
  indicators: {
    [k: string]: IndicatorAudit;
  };
  dataset_issues: ValidationIssue[];
}
/**
 * An indicator audit combines its empirical data profile with the results of validation
 * checks.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorAudit".
 */
export interface IndicatorAudit {
  profile?: IndicatorEmpiricalProfile | null;
  validation: IndicatorValidation;
}
/**
 * An empirical profile summarizes an indicator's observed values, coverage, and data-
 * quality signals.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorEmpiricalProfile".
 */
export interface IndicatorEmpiricalProfile {
  measurement_dtype?: string | null;
  n_obs: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  q25?: number | null;
  q50?: number | null;
  q75?: number | null;
  variance: number | null;
  time_coverage_ratio: number | null;
  max_gap_ratio: number | null;
  dtype_violations?: number | null;
  duplicate_pct?: number | null;
  arithmetic_sequence_detected: boolean;
  n_unparseable_timestamps?: number | null;
  zero_fraction?: number | null;
  is_nonnegative?: boolean | null;
  is_unit_interval?: boolean | null;
  looks_integer_valued?: boolean | null;
  variance_to_mean_ratio?: number | null;
}
/**
 * Indicator validation records the outcomes and issues from checks on one extracted
 * indicator.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorValidation".
 */
export interface IndicatorValidation {
  issues: ValidationIssue[];
  checks: {
    [k: string]: "ok" | "warning" | "error";
  };
}
/**
 * A validation issue explains a data problem and its severity for an indicator or the
 * dataset.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ValidationIssue".
 */
export interface ValidationIssue {
  subject?: EntityRef | null;
  issue_type: string;
  severity: "error" | "warning" | "info";
  message: string;
}
/**
 * An edge reference identifies a causal relationship independently of edits to its
 * definition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeRef".
 */
export interface EdgeRef {
  kind: "edge";
  id: EdgeId;
}
/**
 * An indicator reference identifies a measurement definition independently of its name or
 * revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IndicatorRef".
 */
export interface IndicatorRef {
  kind: "indicator";
  id: IndicatorId;
}
/**
 * A particular additive term, independently of its position or coefficient values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismRef".
 */
export interface MechanismRef {
  kind: "mechanism";
  id: MechanismId;
}
/**
 * Prior research and admission findings pinned to the model that was checked.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AdmissionReport".
 */
export interface AdmissionReport {
  search_queries?: {
    [k: string]: string;
  } | null;
  validation_warnings?: string[] | null;
  prior_predictive_samples?: {
    [k: string]: number[];
  } | null;
  prior_predictive_diagnostics: PriorPredictiveDiagnostic[];
}
/**
 * A prior predictive diagnostic records the result of one exact model-admission check.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorPredictiveDiagnostic".
 */
export interface PriorPredictiveDiagnostic {
  check: string;
  construct_id: ConstructId;
  value: string;
  band: string;
  passed: boolean;
  note: string;
  diagnosis: string[];
  mode: string;
}
/**
 * A baseline report collects treatment effects and explicitly retained simulations from the fitted model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BaselineReportArtifact".
 */
export interface BaselineReportArtifact {
  intervention_results: TreatmentEffect[];
  simulation_results: SimulationResult[];
  final_summary?: string | null;
}
/**
 * A treatment effect stores posterior effect draws and optional temporal or observed-scale
 * summaries.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TreatmentEffect".
 */
export interface TreatmentEffect {
  treatment: string;
  treatment_id: ConstructId;
  summary: EffectSummary | null;
  histogram: HistogramBin[];
  posterior_draws?: number[] | null;
  temporal?: TemporalEffect | null;
  manifest_effects?: {
    [k: string]: number;
  } | null;
}
/**
 * An effect summary reports posterior location, uncertainty, and sign probability.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EffectSummary".
 */
export interface EffectSummary {
  mean: number;
  median: number;
  lower_95: number;
  upper_95: number;
  prob_positive: number;
}
/**
 * A histogram bin gives its interval, center, and number of posterior draws.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "HistogramBin".
 */
export interface HistogramBin {
  bin_center: number;
  bin_start: number;
  bin_end: number;
  count: number;
}
/**
 * A temporal effect summarizes a trajectory at requested horizons and its absolute peak.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TemporalEffect".
 */
export interface TemporalEffect {
  /**
   * @minItems 1
   */
  horizons: [EffectTrajectoryPoint, ...EffectTrajectoryPoint[]];
  peak_effect: number;
  time_to_peak_days: number;
}
/**
 * An effect trajectory point records a causal delta at one rollout time.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EffectTrajectoryPoint".
 */
export interface EffectTrajectoryPoint {
  day: number;
  effect: number;
}
/**
 * Ephemeral response, retained only when explicitly included in a report.
 *
 * This engine integrates the true nonlinear drift for each posterior draw.
 * It does not include future process noise or claim the mean of the SDE.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationResult".
 */
export interface SimulationResult {
  request: ScenarioRequest;
  provenance: SimulationProvenance;
  labels: {
    [k: string]: string;
  };
  summary: EffectSummary;
  effect_trajectory?: EffectTrajectoryPoint[] | null;
  trajectory_peak?: EffectTrajectoryPoint | null;
  visualization?: BaselineReportVisualization | null;
  manifest_effects?: {
    [k: string]: number;
  } | null;
  reference_mean: number;
  warnings: string[];
}
/**
 * A simulation request declares the initial state, timed clamps, and requested outcome readout.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioRequest".
 */
export interface ScenarioRequest {
  start: ScenarioStartInput;
  /**
   * One or more timed latent clamps composing the scenario.
   *
   * @minItems 1
   */
  clamps: [ScenarioClamp, ...ScenarioClamp[]];
  outcome: ConstructRef;
  readout: ScenarioQueryInput;
}
/**
 * Where the forward rollout begins (replaces the rung-2/rung-3 split).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioStartInput".
 */
export interface ScenarioStartInput {
  /**
   * 'baseline' starts from the deterministic drift equilibrium (an interventional, rung-2 query). 'abducted' conditions on the individual's observed evidence and starts from the recovered fitted latent state (a counterfactual, rung-3 query).
   */
  kind: "baseline" | "abducted";
  /**
   * Abducted start only: observed fitted-state index to begin from. Defaults to the final retained fitted latent state.
   */
  time_index?: number | null;
  /**
   * Abducted start only: ISO-8601 observed timestamp matching a retained fitted latent state. Use either time_index or time, not both.
   */
  time?: string | null;
}
/**
 * A do-operator on one latent variable over a time window.
 *
 * The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
 * the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
 * value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
 * interpolates across the window, and ``trajectory`` tracks a list of values across it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioClamp".
 */
export interface ScenarioClamp {
  target: ConstructRef;
  /**
   * How the clamped value is specified over the window.
   */
  mode: "set" | "shift" | "ramp" | "trajectory";
  /**
   * Required when mode='set'. Absolute latent-space value.
   */
  value?: number | null;
  /**
   * Required when mode='shift'. Additive delta from the start-state value.
   */
  amount?: number | null;
  /**
   * Required when mode='ramp'. Value at from_day.
   */
  value_start?: number | null;
  /**
   * Required when mode='ramp'. Value at to_day.
   */
  value_end?: number | null;
  /**
   * Required when mode='trajectory'. Values sampled evenly across the window.
   */
  values?: number[] | null;
  /**
   * Window onset in days from the rollout start.
   */
  from_day: number;
  /**
   * Window end in days from the rollout start. Null runs through the horizon.
   */
  to_day?: number | null;
}
/**
 * A scenario readout requests an estimand, forward horizon, and output scale.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioQueryInput".
 */
export interface ScenarioQueryInput {
  /**
   * Report the final-horizon outcome effect or the full effect trajectory.
   */
  estimand: "end_state" | "trajectory";
  /**
   * Forward horizon in days from the rollout start.
   */
  horizon_days: number;
  /**
   * Report latent outcome effects, manifest projections, or both.
   */
  projection: "latent" | "manifest" | "both";
}
/**
 * The retained fit and actual numerical settings used by this response.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulationProvenance".
 */
export interface SimulationProvenance {
  model: ModelRevision;
  engine: "nonlinear_drift_v1" | "illustrative_fixture";
  solver: "Tsit5";
  rtol: number;
  atol: number;
  max_steps: number;
  draw_count: number;
  /**
   * @minItems 2
   */
  time_grid_days: [number, number, ...number[]];
  start_time_index?: number | null;
  start_time?: string | null;
}
/**
 * The workspace and version of the scientific design supporting an inference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelRevision".
 */
export interface ModelRevision {
  workspace_id: string;
  version: number;
}
/**
 * Scenario visualization data contains exact-engine reference and action trajectories.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BaselineReportVisualization".
 */
export interface BaselineReportVisualization {
  /**
   * Per-construct latent trajectories for the reference (no-clamp) path aligned to effect_trajectory days.
   */
  reference_node_trajectories?: {
    [k: string]: number[];
  } | null;
  /**
   * Per-construct latent trajectories under the composed clamps aligned to effect_trajectory days.
   */
  action_node_trajectories?: {
    [k: string]: number[];
  } | null;
  /**
   * Per-construct latent effect trajectories aligned to effect_trajectory days. Values are causal deltas relative to the reference path.
   */
  node_effect_trajectories?: {
    [k: string]: number[];
  } | null;
  /**
   * Posterior mean latent state the rollout started from.
   */
  start_state?: {
    [k: string]: number;
  } | null;
}
/**
 * An action declares a machine operation and its interaction context.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ActionSpec".
 */
export interface ActionSpec {
  action_id: string;
  namespace: string;
  name: string;
  kind: "read" | "produce" | "check" | "query" | "driver" | "external";
  mode: "direct" | "delegated" | "async" | "read";
  context_id: string;
  consumes: ArtifactId[];
  produces: ArtifactId[];
  produces_optional: ArtifactId[];
  derives: ArtifactId[];
  move?: Move | null;
  query?: ToolQuerySpec | null;
  lower_context_id?: string | null;
}
/**
 * A run move invokes an authoring or computation operation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RunOperation".
 */
export interface RunOperation {
  kind: "run";
  operation_id: OperationId;
}
/**
 * A write move requests a validated authored artifact revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WriteArtifact".
 */
export interface WriteArtifact {
  kind: "write";
  artifact_id: ArtifactId;
  provenance: Provenance;
  expected_model_version?: number | null;
}
/**
 * A tool query specification declares a context's callable query.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ToolQuerySpec".
 */
export interface ToolQuerySpec {
  context_id: string;
  tool_name: string;
  freshness_checked: boolean;
}
/**
 * Compiler proof that one retained latent has location and scale anchors.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AnchorCertificate".
 */
export interface AnchorCertificate {
  construct_id: string;
  construct_name: string;
  location_anchor: "standardized_manifest" | "exact_state_observation" | "fixed_dynamics_center" | "fixed_initial_mean";
  location_source_id?: string | null;
  scale_anchor: "fixed_manifest_loading" | "categorical_slope_pin";
  scale_source_id: string;
}
/**
 * An artifact envelope delivers a stored payload with its version, provenance, and file
 * list.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactEnvelope".
 */
export interface ArtifactEnvelope {
  workspace_id: string;
  artifact_id: ArtifactId;
  version: number;
  meta: ArtifactVersionInfo;
  payload: UncheckedJsonObject;
  binary_files: string[];
}
/**
 * Artifact version metadata records how a stored artifact was produced and which inputs it
 * used.
 *
 * ``derived_from`` pins the exact input versions the payload was computed
 * from. For root artifacts (user writes) it is empty. ``created_at`` is
 * stamped by the activity that produced the version — never inside workflow
 * code, where wall-clock time is non-deterministic.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactVersionInfo".
 */
export interface ArtifactVersionInfo {
  artifact_id: ArtifactId;
  version: number;
  provenance: Provenance;
  derived_from: {
    [k: string]: number;
  };
  model_inputs: {
    [k: string]: string;
  };
  consumed_model_inputs: {
    [k: string]: string;
  };
  produced_by?: string | null;
  created_at: string;
}
/**
 * An unchecked JSON object carries data across a boundary before domain validation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "UncheckedJsonObject".
 */
export interface UncheckedJsonObject {
  [k: string]: any;
}
/**
 * An artifact file specification declares its JSON payloads, tables, and executable binaries.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactFileSpec".
 */
export interface ArtifactFileSpec {
  json: {
    [k: string]: string;
  };
  parquet: {
    [k: string]: string;
  };
}
/**
 * An artifact's presence and freshness are derived from the selected journal revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactFreshness".
 */
export interface ArtifactFreshness {
  artifact_id: ArtifactId;
  exists: boolean;
  stale: boolean;
  version?: number | null;
  provenance?: Provenance | null;
  produced_by?: string | null;
}
/**
 * An artifact reference identifies the exact stored version that supports a model fact.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactRef".
 */
export interface ArtifactRef {
  artifact_id: ArtifactId;
  version: number;
}
/**
 * Profile and representative rows from one uploaded table version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataData".
 */
export interface RawDataData {
  n_records: number;
  n_columns: number;
  date_range: RawDataDateRange;
  sample: {
    [k: string]: string | null;
  }[];
  column_descriptions: RawDataColumnDescription[];
}
/**
 * Observed date bounds of the uploaded table, when it contains a date column.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataDateRange".
 */
export interface RawDataDateRange {
  start: string;
  end: string;
}
/**
 * A stored column's physical type and authored interpretation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataColumnDescription".
 */
export interface RawDataColumnDescription {
  name: string;
  dtype: string;
  description: string;
}
/**
 * Counts and representative observations read directly from one panel version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementsData".
 */
export interface MeasurementsData {
  n_observations: number;
  per_indicator_counts: {
    [k: string]: number;
  };
  combined_extractions_sample: ObservationRecord[];
}
/**
 * Canonical serialized extraction observation row.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationRecord".
 */
export interface ObservationRecord {
  indicator_id: IndicatorId;
  value: string | number | boolean | null;
  anchor_time: string | null;
  support_kind: string | null;
  summary_operator: string | null;
  anchor_policy: string | null;
  observation_window: string | null;
  support_start: string | null;
  support_end: string | null;
}
/**
 * Server-derived equations and comparisons with pinned observations.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelDiagnostics".
 */
export interface ModelDiagnostics {
  confounder_equations: StateEquation[];
  state_equations: StateEquation[];
  observation_equations: {
    [k: string]: string;
  };
  likelihood_diagnostics: {
    [k: string]: LikelihoodDiagnostics;
  };
  prior_densities: {
    [k: string]: DensityPoint[];
  };
}
/**
 * A continuous-time state equation rendered from declared scientific mechanisms.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StateEquation".
 */
export interface StateEquation {
  construct_id: ConstructId;
  label: string;
  latex: string;
}
/**
 * Observed values and validation profile for one likelihood's pinned panel.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodDiagnostics".
 */
export interface LikelihoodDiagnostics {
  indicator_id: IndicatorId;
  profile: IndicatorEmpiricalProfile | null;
  histogram: HistogramBin[];
  prior_counts?: number[] | null;
  prior_outside_fraction?: number | null;
}
/**
 * A plotting coordinate evaluated from the native prior's log density.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DensityPoint".
 */
export interface DensityPoint {
  x: number;
  y: number;
}
/**
 * Display findings recorded by an inference transition, separate from ModelSpec.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceReport".
 */
export interface InferenceReport {
  inference_metadata: InferenceMetadata;
  inference_diagnostics: JsonObject;
  assessment: PosteriorAssessment;
  posterior_marginals?: PosteriorMarginal[] | null;
  posterior_pairs?: PosteriorPair[] | null;
}
/**
 * Inference metadata records the sampling method, sample count, and run duration.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InferenceMetadata".
 */
export interface InferenceMetadata {
  method: string;
  n_samples: number;
  duration_seconds: number;
}
/**
 * Predictive assessments of a fitted posterior.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorAssessment".
 */
export interface PosteriorAssessment {
  ppc: PosteriorPredictiveChecks;
  loo_diagnostics?: LOODiagnostics | null;
}
/**
 * Posterior predictive checks report exact-model checks and their supporting plot data.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorPredictiveChecks".
 */
export interface PosteriorPredictiveChecks {
  per_variable_warnings: PPCWarning[];
  checked: boolean;
  n_subsample: number;
  overlays: PPCOverlay[];
  test_stats: PPCTestStat[];
}
/**
 * A predictive-check finding records whether one indicator passes a calibration,
 * dependence, or variance check.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCWarning".
 */
export interface PPCWarning {
  indicator_id: IndicatorId;
  check_type: "calibration" | "autocorrelation" | "variance";
  message: string;
  value: number;
  passed: boolean;
}
/**
 * A predictive overlay compares observed values with posterior predictive bands for one
 * indicator.
 *
 * Provides the data for Gabry's ppc_dens_overlay / ppc_ribbon plots:
 * observed time series vs posterior predictive quantile bands.
 * Optionally includes individual y_rep draw lines for spaghetti plots.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PPCOverlay".
 */
export interface PPCOverlay {
  indicator_id: IndicatorId;
  observed: (number | null)[];
  q025: number[];
  q25: number[];
  median: number[];
  q75: number[];
  q975: number[];
  spaghetti_draws: number[][];
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
  indicator_id: IndicatorId;
  stat_name: "mean" | "sd" | "min" | "max";
  observed_value: number;
  rep_values: number[];
  p_value: number | null;
  histogram: HistogramBin[];
}
/**
 * Leave-one-out diagnostics assess predictive fit and the reliability of its cross-
 * validation estimate.
 *
 * Exact emission factors on joint parameter/state draws support holding out
 * one measurement row. All other rows, including future rows, are available
 * for interpolation. PSIS reliability is assessed with Pareto-k diagnostics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LOODiagnostics".
 */
export interface LOODiagnostics {
  elpd_loo: number;
  p_loo: number;
  se: number;
  n_data_points: number;
  observation_unit: "measurement_row";
  prediction_task: "interpolation_given_other_measurements";
  likelihood_source: "exact_emission_on_joint_particle_draws";
  pareto_k?: number[] | null;
  n_bad_k?: number | null;
  loo_pit?: number[] | null;
}
/**
 * A posterior marginal summarizes uncertainty in one scalar parameter and supplies its
 * density plot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorMarginal".
 */
export interface PosteriorMarginal {
  mean: number;
  lower: number;
  upper: number;
  interval_kind: "hdi" | "equal_tail";
  /**
   * Posterior probability mass of the interval.
   */
  interval_mass: number;
  parameter: string;
  subject: ParameterRef;
  x_values: number[];
  density: number[];
  sd: number;
}
/**
 * A scalar finding identifies its scientific parameter and declared logical component.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterRef".
 */
export interface ParameterRef {
  parameter_id: ParameterId;
  element_id: ParameterElementId;
}
/**
 * A posterior pair supplies joint samples of two parameters to visualize their dependence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorPair".
 */
export interface PosteriorPair {
  param_x: string;
  subject_x: ParameterRef;
  param_y: string;
  subject_y: ParameterRef;
  x_values: number[];
  y_values: number[];
  divergent?: boolean[] | null;
}
/**
 * Available artifact projections read from one committed model state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactViews".
 */
export interface ArtifactViews {
  raw_data?: RawDataData | null;
  model?: ModelSpec | null;
  measurements?: MeasurementsData | null;
  validation_report?: ValidationReportArtifact | null;
  admission_report?: AdmissionReport | null;
  model_diagnostics?: ModelDiagnostics | null;
  inference_report?: InferenceReport | null;
  baseline_report?: BaselineReportArtifact | null;
}
/**
 * An auto-run acknowledgement identifies the active background episode driver.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "AutoRunResponse".
 */
export interface AutoRunResponse {
  ok: true;
  auto_running: true;
  workspace_id: string;
}
/**
 * This response tells clients whether the episode facade supports model-changing moves.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CapabilitiesResponse".
 */
export interface CapabilitiesResponse {
  moves_enabled: boolean;
}
/**
 * An interaction context declares the tools and machine actions available to an agent.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ContextSpec".
 */
export interface ContextSpec {
  context_id: string;
  layer: "navigator" | "registry" | "machine" | "delegated" | "tool";
  label: string;
  parent_id?: string | null;
  owns: ArtifactId[];
  allowed_tools: string[];
  runtime_state: string[];
}
/**
 * A derivation declares an artifact maintained atomically with its input versions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Derivation".
 */
export interface Derivation {
  produces: ArtifactId;
  from: ArtifactId[];
  optional: boolean;
}
/**
 * Episode state identifies the artifact versions currently selected by the transition
 * journal.
 *
 * ``current`` maps artifact id → the version info that is *current* for the
 * episode. Absent key = the artifact does not exist (either never produced,
 * or produced-when-nonempty semantics withheld it).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EpisodeState".
 */
export interface EpisodeState {
  current: {
    [k: string]: ArtifactVersionInfo;
  };
}
/**
 * Episode status reports committed artifacts, their freshness, and available moves.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EpisodeStatus".
 */
export interface EpisodeStatus {
  workspace_id: string;
  seq: number;
  state: EpisodeState;
  artifacts: ArtifactFreshness[];
  next_operation?: OperationId | null;
  legal: Move[];
  auto_running: boolean;
}
/**
 * An events response pages runtime telemetry without reconstructing model state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EventsResponse".
 */
export interface EventsResponse {
  workspace_id: string;
  events: RuntimeEvent[];
}
/**
 * One transition lifecycle event.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionRuntimeEvent".
 */
export interface TransitionRuntimeEvent {
  cursor: string;
  event:
    | "nof1-causal-lab.transition.running"
    | "nof1-causal-lab.transition.completed"
    | "nof1-causal-lab.transition.failed";
  payload: TransitionRuntimeEventPayload;
}
/**
 * Payload for transition lifecycle telemetry.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionRuntimeEventPayload".
 */
export interface TransitionRuntimeEventPayload {
  transition_id: string;
  status: "running" | "completed" | "failed";
  error?: RuntimeEventError | null;
}
/**
 * Serialized transition failure.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RuntimeEventError".
 */
export interface RuntimeEventError {
  type: string;
  message: string;
}
/**
 * Extraction plan telemetry event.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionPlanEvent".
 */
export interface ExtractionPlanEvent {
  cursor: string;
  event: "nof1-causal-lab.extraction.plan";
  payload: ExtractionPlanEventPayload;
}
/**
 * Static extraction fan-out plan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionPlanEventPayload".
 */
export interface ExtractionPlanEventPayload {
  context_id: "measurement";
  type: "plan";
  total_workers: number;
  max_concurrent_workers?: number | null;
  max_rpm?: number | null;
}
/**
 * Extraction worker telemetry event.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionWorkerEvent".
 */
export interface ExtractionWorkerEvent {
  cursor: string;
  event: "nof1-causal-lab.extraction.worker";
  payload: ExtractionWorkerEventPayload;
}
/**
 * One extraction worker state transition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionWorkerEventPayload".
 */
export interface ExtractionWorkerEventPayload {
  context_id: "measurement";
  type: "worker";
  worker_id: number;
  state: "pending" | "running" | "completed" | "failed";
  n_windows: number;
  n_extractions?: number | null;
  n_llm_calls?: number | null;
  error?: string | null;
}
/**
 * Extraction snapshot telemetry event.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionSnapshotEvent".
 */
export interface ExtractionSnapshotEvent {
  cursor: string;
  event: "nof1-causal-lab.extraction.snapshot";
  payload: ExtractionSnapshotEventPayload;
}
/**
 * Aggregate extraction progress snapshot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExtractionSnapshotEventPayload".
 */
export interface ExtractionSnapshotEventPayload {
  context_id: "measurement";
  type: "snapshot";
  total_workers: number;
  pending_workers: number;
  running_workers: number;
  completed_workers: number;
  failed_workers: number;
  llm_requests_last_60s: number;
}
/**
 * Construct-admission event with JSON-safe event-specific fields.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSpecAdmissionEvent".
 */
export interface ModelSpecAdmissionEvent {
  cursor: string;
  event:
    | "nof1-causal-lab.model-spec.admission.plan"
    | "nof1-causal-lab.model-spec.admission.resumed"
    | "nof1-causal-lab.model-spec.admission.construct_started"
    | "nof1-causal-lab.model-spec.admission.construct_checking"
    | "nof1-causal-lab.model-spec.admission.construct_report"
    | "nof1-causal-lab.model-spec.admission.barrier_report"
    | "nof1-causal-lab.model-spec.admission.done"
    | "nof1-causal-lab.model-spec.admission.failed";
  payload: JsonObject;
}
/**
 * Current model requirements and latent anchors, computed without a stored receipt.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ExecutionReadiness".
 */
export interface ExecutionReadiness {
  unmet_requirements: string[];
  anchor_certificates: AnchorCertificate[];
}
/**
 * A fact source locates supporting content within an artifact version and records its freshness.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FactSource".
 */
export interface FactSource {
  ref: ArtifactRef | TransitionRef;
  pointer: string;
  validity: SourceValidity;
}
/**
 * An immutable entry in the workspace transition journal.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionRef".
 */
export interface TransitionRef {
  seq: number;
}
/**
 * A fit read contains the inference log report and server-composed display findings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitSummary".
 */
export interface FitSummary {
  report: InferenceReport;
  predictive_checks_passed: number;
  predictive_checks_total: number;
  edge_estimates: {
    [k: string]: PosteriorEstimate;
  };
  decay_estimates: {
    [k: string]: PosteriorEstimate;
  };
}
/**
 * A posterior estimate reports a mean and a credible interval with explicit semantics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorEstimate".
 */
export interface PosteriorEstimate {
  mean: number;
  lower: number;
  upper: number;
  interval_kind: "hdi" | "equal_tail";
  /**
   * Posterior probability mass of the interval.
   */
  interval_mass: number;
}
/**
 * An LLM trace records a conversation, its model, elapsed time, and token usage.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LLMTrace".
 */
export interface LLMTrace {
  messages: TraceMessage[];
  model: string;
  total_time_seconds: number;
  usage: TraceUsage;
}
/**
 * A trace message records one conversational step, including any reasoning or tool
 * interaction.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceMessage".
 */
export interface TraceMessage {
  role: string;
  content: string;
  reasoning?: string | null;
  tool_calls?: UncheckedJsonObject[] | null;
  tool_call_id?: string | null;
  tool_name?: string | null;
  tool_result?: string | null;
  tool_is_error: boolean;
}
/**
 * Trace usage records the input, output, and reasoning tokens consumed by a conversation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceUsage".
 */
export interface TraceUsage {
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens?: number | null;
}
/**
 * The machine description exposes the artifact graph, storage contracts, and action hierarchy.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MachineDescription".
 */
export interface MachineDescription {
  artifact_ids: ArtifactId[];
  topological_artifact_order: ArtifactId[];
  topological_transition_order: OperationId[];
  contexts: ContextSpec[];
  actions: ActionSpec[];
  roots: Root[];
  transitions: MachineTransition[];
  derivations: Derivation[];
  files: {
    [k: string]: ArtifactFileSpec;
  };
}
/**
 * A root declares an independently writable artifact and any contextual input pins.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Root".
 */
export interface Root {
  artifact_id: ArtifactId;
  write_pins: ArtifactId[];
}
/**
 * A transition declares the artifacts it consumes and produces and how it can run.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MachineTransition".
 */
export interface MachineTransition {
  transition_id: OperationId;
  consumes: ArtifactId[];
  produces: ArtifactId[];
  produces_optional: ArtifactId[];
  creation_class: "deterministic" | "batch_llm" | "judgment";
  writable: boolean;
}
/**
 * ModelSpec data pairs the causal question and observed evidence with their source versions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelData".
 */
export interface ModelData {
  question?: SourcedQuestionArtifact | null;
  raw_data?: SourcedRawDataData | null;
  measurements?: SourcedMeasurementsData | null;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_QuestionArtifact_".
 */
export interface SourcedQuestionArtifact {
  value: QuestionArtifact;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_RawDataData_".
 */
export interface SourcedRawDataData {
  value: RawDataData;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_MeasurementsData_".
 */
export interface SourcedMeasurementsData {
  value: MeasurementsData;
  source: FactSource;
}
/**
 * ModelSpec findings collect identification, validation, and fitted results with their provenance.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelFindings".
 */
export interface ModelFindings {
  identification?: SourcedIdentificationReport | null;
  execution?: SourcedExecutionReadiness | null;
  dispositions?: SourcedTupleStructuralItemDisposition | null;
  graph_status: {
    [k: string]: "observed" | "marginalized" | "blocking";
  };
  validation_report?: SourcedValidationReportArtifact | null;
  admission_report?: SourcedAdmissionReport | null;
  diagnostics?: ModelDiagnostics | null;
  fit?: SourcedFitSummary | null;
  baseline_report?: SourcedBaselineReportArtifact | null;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_IdentificationReport_".
 */
export interface SourcedIdentificationReport {
  value: IdentificationReport;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_ExecutionReadiness_".
 */
export interface SourcedExecutionReadiness {
  value: ExecutionReadiness;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_tuple_StructuralItemDisposition__________".
 */
export interface SourcedTupleStructuralItemDisposition {
  value: StructuralItemDisposition[];
  source: FactSource;
}
/**
 * An item disposition explains the compilation decision for one identified authored
 * entity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralItemDisposition".
 */
export interface StructuralItemDisposition {
  source_id: string;
  source_kind: "construct" | "edge" | "indicator";
  disposition: StructuralDisposition;
  reason: string;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_ValidationReportArtifact_".
 */
export interface SourcedValidationReportArtifact {
  value: ValidationReportArtifact;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_AdmissionReport_".
 */
export interface SourcedAdmissionReport {
  value: AdmissionReport;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_FitSummary_".
 */
export interface SourcedFitSummary {
  value: FitSummary;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_BaselineReportArtifact_".
 */
export interface SourcedBaselineReportArtifact {
  value: BaselineReportArtifact;
  source: FactSource;
}
/**
 * A model reference identifies the workspace that owns the scientific model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelRef".
 */
export interface ModelRef {
  kind: "model";
  id: string;
}
/**
 * The canonical scientific definition with independently sourced inputs and findings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSnapshot".
 */
export interface ModelSnapshot {
  model?: SourcedModelSpec | null;
  context: SnapshotContext;
  data: ModelData;
  findings: ModelFindings;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_ModelSpec_".
 */
export interface SourcedModelSpec {
  value: ModelSpec;
  source: FactSource;
}
/**
 * A snapshot context identifies the selected journal revision and its artifact versions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SnapshotContext".
 */
export interface SnapshotContext {
  workspace: ModelRef;
  seq: number;
  can_simulate: boolean;
  state: EpisodeState;
  artifacts: ArtifactFreshness[];
  installed_at: {
    [k: string]: number;
  };
  retracted: ArtifactId[];
}
/**
 * A move outcome reports the attempted transition and resulting committed state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MoveOutcome".
 */
export interface MoveOutcome {
  seq: number;
  status: JournalStatus;
  reason?: string | null;
  error_type?: string | null;
  error_message?: string | null;
  diagnostics: JsonObject;
  produced: ArtifactVersionInfo[];
  retracted: RetractedArtifact[];
  state: EpisodeState;
}
/**
 * A current artifact removed by a move, with the finding that caused it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RetractedArtifact".
 */
export interface RetractedArtifact {
  artifact_id: ArtifactId;
  reason_ref: string;
}
/**
 * Stage-owned checkpoint selection retained by a raised transition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ResumeRef".
 */
export interface ResumeRef {
  kind: "model_spec";
  run_id: string;
  checkpoint_id: string;
}
/**
 * A tool error reports why a requested operation could not produce a result.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ToolError".
 */
export interface ToolError {
  error: string;
  identifiable_treatments?: string[] | null;
}
/**
 * Starting an episode returns its current status and any question-write outcome.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StartEpisodeResponse".
 */
export interface StartEpisodeResponse {
  workspace_id: string;
  seq: number;
  state: EpisodeState;
  artifacts: ArtifactFreshness[];
  next_operation?: OperationId | null;
  legal: Move[];
  auto_running: boolean;
  ok: true;
  outcome: MoveOutcome | null;
}
/**
 * Typed transition journal returned by the episode read plane.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TimelineResponse".
 */
export interface TimelineResponse {
  workspace_id: string;
  transitions: TransitionRecord[];
}
/**
 * One journaled transition attempt — applied, rejected, or raised.
 *
 * Rejections are recorded deliberately (a Temporal validator rejection
 * leaves no trace in event history). Current state is reconstructed by
 * replaying applied effects, not serialized into transition records.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionRecord".
 */
export interface TransitionRecord {
  seq: number;
  ts: string;
  move: Move;
  status: JournalStatus;
  reason?: string | null;
  error_type?: string | null;
  error_message?: string | null;
  diagnostics: UncheckedJsonObject;
  produced: ArtifactVersionInfo[];
  retracted: RetractedArtifact[];
  trace_ids: string[];
  resume: ResumeRef | null;
}
/**
 * Promoted traces identified by their committed execution sequence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionTraceIndex".
 */
export interface TransitionTraceIndex {
  workspace_id: string;
  seq: number;
  trace_ids: string[];
}
/**
 * An upload response identifies the stored location of an accepted data upload.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "UploadResponse".
 */
export interface UploadResponse {
  path: string;
}
/**
 * A workspace entry identifies an available model workspace and its research question.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkspaceEntry".
 */
export interface WorkspaceEntry {
  href: string;
  question?: string | null;
  workspaceId: string;
}
/**
 * A workspace list provides the available model workspaces for client navigation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkspaceList".
 */
export interface WorkspaceList {
  workspaces: WorkspaceEntry[];
}
