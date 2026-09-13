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
 * A persistent construct identity survives changes to its display name.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstructId".
 */
export type ConstructId = `construct:${string}`;
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
 * A persistent edge identity identifies one authored causal relationship.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EdgeId".
 */
export type EdgeId = `edge:${string}`;
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
 * An entity reference identifies a construct, edge, or indicator by its persistent identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EntityRef".
 */
export type EntityRef = ConstructRef | EdgeRef | IndicatorRef;
/**
 * This enumeration identifies the probability family used to model an observed variable.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DistributionFamily".
 */
export type DistributionFamily =
  | "gaussian"
  | "student_t"
  | "poisson"
  | "gamma"
  | "bernoulli"
  | "negative_binomial"
  | "beta"
  | "ordered_logistic"
  | "categorical";
/**
 * A link function connects an observation distribution to the model's predictor.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LinkFunction".
 */
export type LinkFunction = "identity" | "log" | "inverse" | "logit" | "probit" | "cumulative_logit" | "softmax";
/**
 * A scientific parameter identity binds a quantity to its explicit owners.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterId".
 */
export type ParameterId = `parameter:${string}`;
/**
 * Semantic role for each sample site.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SiteKind".
 */
export type SiteKind =
  | "dynamics_decay"
  | "dynamics_cint"
  | "dynamics_weight"
  | "dynamics_potential_center"
  | "dynamics_potential_quartic"
  | "hill_emax"
  | "hill_ec50"
  | "hill_n"
  | "diffusion_diag"
  | "diffusion_lower"
  | "input_effect"
  | "static_state_sd"
  | "loading"
  | "manifest_means"
  | "manifest_var_diag"
  | "t0_means"
  | "t0_var_diag"
  | "t0_var_lower"
  | "obs_df"
  | "obs_shape"
  | "obs_r"
  | "obs_concentration"
  | "obs_ordered_base"
  | "obs_ordered_gaps"
  | "obs_cat_intercepts"
  | "obs_cat_slopes"
  | "proc_df";
/**
 * A parameter role identifies which part of the statistical model a parameter controls.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterRole".
 */
export type ParameterRole =
  | "fixed_effect"
  | "ar_coefficient"
  | "dynamics_parameter"
  | "dynamics_parameter_positive"
  | "residual_sd"
  | "state_intercept"
  | "observation_intercept"
  | "initial_state_mean"
  | "initial_state_sd"
  | "static_state_sd"
  | "correlation"
  | "initial_state_correlation"
  | "loading"
  | "measurement_error_sd"
  | "observation_hyperparameter"
  | "observation_hyperparameter_positive";
/**
 * A parameter constraint specifies the permitted range of a model parameter.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterConstraint".
 */
export type ParameterConstraint = "none" | "positive" | "negative" | "unit_interval" | "correlation";
/**
 * How an authored semantic prior is transformed before site attachment.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorAuthoringTransform".
 */
export type PriorAuthoringTransform =
  | "identity"
  | "positive_identity"
  | "dt_persistence_to_ct_decay"
  | "dt_effect_to_ct_rate"
  | "initial_state_correlation"
  | "site_wide"
  | "site_row";
/**
 * A dynamics mechanism declares one contribution to continuous-time drift.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DynamicsMechanism".
 */
export type DynamicsMechanism =
  | NodePotentialMechanism
  | ConstantDriftMechanism
  | LinearEdgeMechanism
  | HillEdgeMechanism;
/**
 * A mechanism coefficient is either fixed or bound to an estimated scientific parameter.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MechanismCoefficient".
 */
export type MechanismCoefficient = FixedCoefficient | EstimatedCoefficient;
/**
 * This policy selects stationary-derived or freely estimated initial conditions for
 * dynamic states.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "InitializationPolicy".
 */
export type InitializationPolicy = "stationary" | "free";
/**
 * This policy determines whether eligible observation intercepts are fixed or freely
 * estimated.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ObservationInterceptPolicy".
 */
export type ObservationInterceptPolicy = "fixed" | "free";
/**
 * A prior proposal specifies a parameter's prior distribution with its rationale and
 * supporting evidence.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorProposal".
 */
export type PriorProposal = {
  distribution: PriorDistributionFamily;
  /**
   * Arguments to the declared NumPyro family
   */
  params: {
    [k: string]: number;
  };
  parameter_id: ParameterId;
  /**
   * Literature sources supporting this prior
   */
  sources: PriorSource[];
  /**
   * Justification for the chosen prior distribution and parameters
   */
  reasoning: string;
  /**
   * Observation interval (in days) that the DT prior is expressed in. Sourced from the study's measurement schedule (e.g., 7 for a weekly study). Used for DT→CT conversion of dynamic priors (e.g. beta/dt for cross-lags, -log(rho)/dt for baseline persistence).
   */
  reference_interval_days?: number | null;
  /**
   * Pre-computed density curve points [{x, y}, ...] for frontend visualization. Computed by the pipeline before persistence so the frontend doesn't need to approximate the PDF client-side.
   */
  density_points?: DensityPoint[] | null;
} & (
  | {
      distribution: "Normal";
      params: {
        mu: number;
        sigma: number;
      };
    }
  | {
      distribution: "HalfNormal";
      params: {
        sigma: number;
      };
    }
  | {
      distribution: "Beta";
      params: {
        alpha: number;
        beta: number;
      };
    }
  | {
      distribution: "Uniform";
      params: {
        lower: number;
        upper: number;
      };
    }
  | {
      distribution: "TruncatedNormal";
      params: {
        mu: number;
        sigma: number;
        lower: number;
        upper: number;
      };
    }
  | {
      distribution: "Gamma";
      params: {
        concentration: number;
        rate: number;
      };
    }
  | {
      distribution: "LogNormal";
      params: {
        mu: number;
        sigma: number;
      };
    }
  | {
      distribution: "Exponential";
      params: {
        rate: number;
      };
    }
  | {
      distribution: "Delta";
      params: {
        value: number;
      };
    }
);
/**
 * This enumeration identifies the probability families permitted in authored prior
 * proposals.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorDistributionFamily".
 */
export type PriorDistributionFamily =
  | "Normal"
  | "HalfNormal"
  | "Beta"
  | "Uniform"
  | "TruncatedNormal"
  | "Gamma"
  | "LogNormal"
  | "Exponential"
  | "Delta";
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
 * Runtime support class for a sample site.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SupportClass".
 */
export type SupportClass = "real" | "positive" | "correlation";
/**
 * A scalar distribution recipe; NumPyro owns its density and transforms.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledDistribution".
 */
export type CompiledDistribution = {
  distribution: PriorDistributionFamily;
  /**
   * Arguments to the declared NumPyro family
   */
  params: {
    [k: string]: number;
  };
  transforms: DistributionTransform[];
} & (
  | {
      distribution: "Normal";
      params: {
        mu: number;
        sigma: number;
      };
    }
  | {
      distribution: "HalfNormal";
      params: {
        sigma: number;
      };
    }
  | {
      distribution: "Beta";
      params: {
        alpha: number;
        beta: number;
      };
    }
  | {
      distribution: "Uniform";
      params: {
        lower: number;
        upper: number;
      };
    }
  | {
      distribution: "TruncatedNormal";
      params: {
        mu: number;
        sigma: number;
        lower: number;
        upper: number;
      };
    }
  | {
      distribution: "Gamma";
      params: {
        concentration: number;
        rate: number;
      };
    }
  | {
      distribution: "LogNormal";
      params: {
        mu: number;
        sigma: number;
      };
    }
  | {
      distribution: "Exponential";
      params: {
        rate: number;
      };
    }
  | {
      distribution: "Delta";
      params: {
        value: number;
      };
    }
);
/**
 * A parameter element identity identifies a logical scalar component across model revisions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterElementId".
 */
export type ParameterElementId = `element:${string}`;
/**
 * A scientific query identity survives model revisions and posterior refits.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioQueryId".
 */
export type ScenarioQueryId = `query:${string}`;
/**
 * An evaluation identity binds one scientific query to its exact model and posterior.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioEvaluationId".
 */
export type ScenarioEvaluationId = `evaluation:${string}`;
/**
 * An artifact identity selects one node in the machine's artifact graph.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactId".
 */
export type ArtifactId =
  | "question"
  | "raw_data"
  | "latent_structure"
  | "measurement_structure"
  | "causal_design"
  | "structural_plan"
  | "identification_report"
  | "measurements"
  | "panel"
  | "validation_report"
  | "statistical_model_spec"
  | "compiled_ssm"
  | "posterior"
  | "baseline_report"
  | "saved_scenarios";
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
  | LatentStructureArtifact
  | MeasurementStructureViewData
  | MeasurementsData
  | ValidationReportArtifact
  | StatisticalModelSpecData
  | PosteriorArtifact
  | BaselineReportArtifact;
/**
 * A machine move requests computation or an authored artifact write.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Move".
 */
export type Move = RunArtifact | WriteArtifact;
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
 * A simulation tool response carries either a resolved scenario result or a reported tool error.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulateScenarioToolResult".
 */
export type SimulateScenarioToolResult = SimulateScenarioResult | ToolError;

/**
 * Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.
 */
export interface CausalSSMContracts {
  question: QuestionArtifact;
  raw_data: RawDataArtifact;
  latent_structure: LatentStructureArtifact;
  measurement_structure: MeasurementStructureArtifact;
  causal_design: CausalDesignArtifact;
  structural_plan: StructuralPlanArtifact;
  identification_report: IdentificationReport;
  measurements: MeasurementsArtifact;
  validation_report: ValidationReportArtifact;
  statistical_model_spec: StatisticalModelSpecArtifact;
  compiled_ssm: CompiledSSMArtifact;
  posterior: PosteriorArtifact;
  baseline_report: BaselineReportArtifact;
  saved_scenarios: SavedScenariosArtifact;
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
 * This artifact describes the uploaded dataset's columns for downstream scientific
 * interpretation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RawDataArtifact".
 */
export interface RawDataArtifact {
  column_descriptions: ColumnDescription[];
}
/**
 * A column description explains the meaning of one column in the uploaded dataset.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ColumnDescription".
 */
export interface ColumnDescription {
  name: string;
  description: string;
}
/**
 * This artifact stores the authored causal structure proposed for the research question.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LatentStructureArtifact".
 */
export interface LatentStructureArtifact {
  latent_structure: LatentStructure;
}
/**
 * A latent structure defines the scientific causal graph of constructs and directed
 * relationships.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LatentStructure".
 */
export interface LatentStructure {
  /**
   * Default query target for this workspace; outcome selection is not a construct property.
   */
  default_outcome?: ConstructRef | null;
  /**
   * Theoretical constructs in the model
   *
   * @minItems 1
   */
  constructs: [Construct, ...Construct[]];
  /**
   * Causal edges between constructs
   */
  edges: CausalEdge[];
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
  role: Role;
  temporal_status: TemporalStatus;
}
/**
 * A causal edge declares a directed causal relationship between two constructs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalEdge".
 */
export interface CausalEdge {
  id: EdgeId;
  cause_id: ConstructId;
  effect_id: ConstructId;
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
 * This artifact stores measurement definitions and declarations that shape the executable
 * model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementStructureArtifact".
 */
export interface MeasurementStructureArtifact {
  measurement_structure: MeasurementStructure;
  /**
   * Authored declarations of observed construct trajectories compiled as transition inputs rather than latent states
   */
  known_inputs: KnownInput[];
  /**
   * Measured scientific-context constructs explicitly excluded from the executable N-of-1 state vector
   */
  scientific_only_constructs: ScientificOnlyConstruct[];
}
/**
 * A measurement structure defines the indicators and common clock used to observe
 * constructs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementStructure".
 */
export interface MeasurementStructure {
  /**
   * Observed indicators, each measuring a construct
   */
  indicators: Indicator[];
  /**
   * Observation window width for extraction and SSM discretization. Any Polars-compatible duration string (e.g. '1h', '4h', '1d', '1w'). Choose based on data density: need enough events per support window.
   */
  model_clock: string;
}
/**
 * An indicator defines an observed measurement of a construct and how to extract it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Indicator".
 */
export interface Indicator {
  id: IndicatorId;
  construct_id: ConstructId;
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
 * An observed-input declaration binds a construct to its measured driver trajectory.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "KnownInput".
 */
export interface KnownInput {
  construct_id: ConstructId;
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
  construct_id: ConstructId;
  reason: string;
}
/**
 * The causal-design file stores the derived scientific design.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalDesignArtifact".
 */
export interface CausalDesignArtifact {
  causal_design: CausalDesign;
}
/**
 * Scientific causal design before executable structural compilation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalDesign".
 */
export interface CausalDesign {
  latent: LatentStructure;
  measurement: MeasurementStructure;
  /**
   * Identifiability status of target causal effects
   */
  identifiability?: IdentifiabilityStatus | null;
  /**
   * Authored observed-input declarations compiled by StructuralPlan
   */
  known_inputs: KnownInput[];
  /**
   * Measured constructs explicitly excluded from the executable SSM
   */
  scientific_only_constructs: ScientificOnlyConstruct[];
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
   * Identification strategy (e.g., do_calculus, instrumental_variable)
   */
  method: string;
  /**
   * Closed-form estimand or IV placeholder
   */
  estimand: string;
  /**
   * Unobserved confounders the estimand integrates out
   */
  marginalized_confounders: ConstructId[];
  /**
   * Instrumental variables used (if method=instrumental_variable)
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
 * The structural-plan file stores the self-contained executable plan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralPlanArtifact".
 */
export interface StructuralPlanArtifact {
  structural_plan: StructuralPlan;
}
/**
 * A structural plan translates the scientific causal design into the topology used by
 * model compilation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralPlan".
 */
export interface StructuralPlan {
  schema_version: 1;
  semantics: StructuralSemanticCatalog;
  state_order: ConstructId[];
  edges: StructuralEdge[];
  manifest_indicator_order: IndicatorId[];
  reference_indicator_ids: {
    [k: string]: IndicatorId;
  };
  known_inputs: StructuralKnownInput[];
  induced_dependencies: StructuralInducedDependency[];
  dispositions: StructuralItemDisposition[];
}
/**
 * The semantic catalog preserves authored definitions under the IDs used by the executable
 * plan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralSemanticCatalog".
 */
export interface StructuralSemanticCatalog {
  constructs: {
    [k: string]: Construct;
  };
  edges: {
    [k: string]: CausalEdge;
  };
  indicators: {
    [k: string]: Indicator;
  };
  model_clock: string;
}
/**
 * A structural edge connects retained states or known inputs in the executable model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralEdge".
 */
export interface StructuralEdge {
  source_id: EdgeId;
  cause_id: ConstructId;
  effect_id: ConstructId;
  lagged: boolean;
}
/**
 * A structural known input binds an observed indicator to a driver of the executable
 * dynamics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralKnownInput".
 */
export interface StructuralKnownInput {
  construct_id: ConstructId;
  source_indicator_id: IndicatorId;
  /**
   * Positive divisor applied before inference
   */
  scale: number;
  missing_policy: "zero" | "forward_fill";
  source_id: string;
}
/**
 * An induced dependency records dependence created by projecting explicit latent root
 * confounders.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StructuralInducedDependency".
 */
export interface StructuralInducedDependency {
  source_id: string;
  /**
   * @minItems 2
   * @maxItems 2
   */
  between: [any, any];
  kind: "innovation_correlation" | "initial_state_correlation";
  source_confounder_ids: ConstructId[];
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
 * The positive identification finding.
 *
 * Only produced when at least one treatment effect is explicitly
 * identifiable. Negative findings remain in ``causal_design.identifiability``.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "IdentificationReport".
 */
export interface IdentificationReport {
  outcome_id: ConstructId;
  /**
   * @minItems 1
   */
  estimable_treatments: [ConstructId, ...ConstructId[]];
  non_identifiable_treatments: {
    [k: string]: NonIdentifiableTreatmentStatus;
  };
}
/**
 * This artifact records extraction-worker progress and output counts for a measurement
 * run.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementsArtifact".
 */
export interface MeasurementsArtifact {
  workers: WorkerStatus[];
}
/**
 * A worker status reports extraction progress, produced measurements, and any failure for
 * one worker.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "WorkerStatus".
 */
export interface WorkerStatus {
  worker_id: number;
  status: "pending" | "running" | "completed" | "failed";
  n_extractions: number;
  n_windows: number;
  error?: string | null;
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
 * This artifact combines the statistical specification with prior proposals and admission
 * diagnostics.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StatisticalModelSpecArtifact".
 */
export interface StatisticalModelSpecArtifact {
  statistical_model_spec: StatisticalModelSpec;
  authored_priors: {
    [k: string]: PriorProposal;
  };
  resolved_priors: PriorProposal[];
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
 * A statistical model specification defines likelihoods, parameter roles, and estimation
 * policies.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StatisticalModelSpec".
 */
export interface StatisticalModelSpec {
  /**
   * Likelihood specifications for each observed indicator
   */
  likelihoods: LikelihoodSpec[];
  /**
   * All parameters requiring priors
   */
  parameters: ParameterSpec[];
  /**
   * Explicit state and edge dynamics; coefficients reference parameter IDs.
   */
  mechanisms: DynamicsMechanism[];
  initialization_policy: InitializationPolicy;
  observation_intercept_policy: ObservationInterceptPolicy;
}
/**
 * A likelihood specification defines how an indicator's observed values follow from the
 * model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LikelihoodSpec".
 */
export interface LikelihoodSpec {
  indicator_id: IndicatorId;
  distribution: DistributionFamily;
  link: LinkFunction;
  /**
   * Whether deterministic standardization (mean-centering and unit-scaling) is applied to the observed values before fitting
   */
  standardized: boolean;
  /**
   * Why this distribution/link was chosen for this variable
   */
  reasoning: string;
  /**
   * Literature sources supporting this likelihood choice
   */
  sources: LiteratureSource[];
}
/**
 * A parameter specification declares a named model quantity, its role, and its allowed
 * values.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterSpec".
 */
export interface ParameterSpec {
  id: ParameterId;
  owners: EntityRef[];
  quantity: SiteKind;
  /**
   * Authored parameter label; relationships use its persistent ID
   */
  name: string;
  role: ParameterRole;
  constraint: ParameterConstraint;
  /**
   * Human-readable description of what this parameter represents
   */
  description: string;
  prior_transform: PriorAuthoringTransform;
  /**
   * Logical scalar components and their labels, declared during compilation.
   */
  elements: {
    [k: string]: string;
  };
}
/**
 * Restoring drift -stiffness * (x - center) - quartic * (x - center)^3.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "NodePotentialMechanism".
 */
export interface NodePotentialMechanism {
  kind: "node_potential";
  target_id: ConstructId;
  center: MechanismCoefficient;
  stiffness: MechanismCoefficient;
  quartic: MechanismCoefficient;
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
 * A free coefficient referencing its scientific parameter definition.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EstimatedCoefficient".
 */
export interface EstimatedCoefficient {
  kind: "estimated";
  parameter_id: ParameterId;
}
/**
 * An additive continuous-time forcing of a state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ConstantDriftMechanism".
 */
export interface ConstantDriftMechanism {
  kind: "constant_drift";
  target_id: ConstructId;
  intercept: EstimatedCoefficient;
}
/**
 * A directed effect proportional to the source state or known input.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LinearEdgeMechanism".
 */
export interface LinearEdgeMechanism {
  kind: "linear";
  edge_id: EdgeId;
  weight: EstimatedCoefficient;
}
/**
 * A directed saturating effect of a state through the native Hill response.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "HillEdgeMechanism".
 */
export interface HillEdgeMechanism {
  kind: "hill";
  edge_id: EdgeId;
  emax: MechanismCoefficient;
  ec50: MechanismCoefficient;
  n: MechanismCoefficient;
}
/**
 * A prior source records literature evidence used to justify a parameter's prior
 * distribution.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorSource".
 */
export interface PriorSource {
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
  /**
   * Reported effect size if available (e.g., 'r=0.3', 'β=0.2')
   */
  effect_size?: string | null;
  /**
   * Observation/measurement interval of this study in days (daily=1, weekly=7, monthly=30)
   */
  study_interval_days?: number | null;
}
/**
 * A density point stores one coordinate of a prior density curve for plotting.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DensityPoint".
 */
export interface DensityPoint {
  x: number;
  y: number;
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
 * Complete versioned artifact required to restore an executable SSM.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledSSMArtifact".
 */
export interface CompiledSSMArtifact {
  schema_version: 2;
  structure: CompiledStructure;
  compiled_prior_semantics: CompiledPriorSemantics;
  observation_bindings: {
    [k: string]: string;
  };
  parameters: ParameterSpec[];
  parameter_bindings: CompiledParameterBinding[];
  auxiliary_coordinates: ParameterCoordinate[];
  compile_diagnostics: PriorValidationResult[];
}
/**
 * Executable structure plus total provenance back to StructuralPlan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledStructure".
 */
export interface CompiledStructure {
  spec: SerializedSSMSpec;
  edge_lag_days: SerializedEdgeLag[];
  bindings: CompiledStructuralBinding[];
  anchor_certificates: AnchorCertificate[];
}
/**
 * JSON representation of the structural ``SSMSpec`` runtime contract.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SerializedSSMSpec".
 */
export interface SerializedSSMSpec {
  n_latent: number;
  n_manifest: number;
  dynamics_spec: JsonObject;
  diffusion_block: JsonObject;
  lambda_block: JsonObject;
  manifest_means_block: JsonObject;
  manifest_chol_block: JsonObject;
  t0_means_block: JsonObject;
  t0_chol_block: JsonObject;
  input_effect_block: JsonObject;
  static_state_sd_block: JsonObject;
  static_factor_loadings: number[][];
  diffusion_dists: DistributionFamily[];
  manifest_dists: DistributionFamily[];
  manifest_level_counts?: number[] | null;
  manifest_links?: LinkFunction[] | null;
  manifest_standardized?: boolean[] | null;
  manifest_cat_anchor?: boolean[] | null;
  latent_ids?: ConstructId[] | null;
  manifest_ids?: IndicatorId[] | null;
  input_ids?: ConstructId[] | null;
  static_factor_ids?: ParameterId[] | null;
  latent_names?: string[] | null;
  manifest_names?: string[] | null;
  input_names?: string[] | null;
  input_source_indicators?: string[] | null;
  input_scales?: number[] | null;
  input_missing_policies?: ("zero" | "forward_fill")[] | null;
  input_lagged: boolean[];
  static_factor_names?: string[] | null;
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
 * One directed continuous-time lag attached to a compiled edge.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SerializedEdgeLag".
 */
export interface SerializedEdgeLag {
  source_id: string;
  effect_idx: number;
  cause_idx: number;
  lag_days: number;
}
/**
 * Stable structural-plan source identity bound to one runtime target.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledStructuralBinding".
 */
export interface CompiledStructuralBinding {
  source_id: string;
  source_kind: "state" | "manifest" | "known_input" | "edge" | "induced_dependency";
  target_kind:
    | "latent_state"
    | "manifest_channel"
    | "transition_input"
    | "dynamics_edge"
    | "input_effect"
    | "diffusion_correlation"
    | "static_factor";
  target_indices: number[];
  target_name: string;
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
  location_anchor: "standardized_manifest" | "fixed_dynamics_center" | "fixed_initial_mean";
  location_source_id?: string | null;
  scale_anchor: "fixed_manifest_loading" | "categorical_slope_pin";
  scale_source_id: string;
}
/**
 * Versioned runtime site registry and lossless native distribution recipes.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledPriorSemantics".
 */
export interface CompiledPriorSemantics {
  schema_version: 7;
  site_registry: SerializedSiteDescriptor[];
  priors: {
    [k: string]: CompiledDistribution[];
  };
}
/**
 * Persisted topology for one runtime sample site.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SerializedSiteDescriptor".
 */
export interface SerializedSiteDescriptor {
  name: string;
  shape: number[];
  support: SupportClass;
  assembly_group: string;
  site_kind: SiteKind;
  deterministic_name?: string | null;
  fixed_spec_field?: string | null;
  priors_field?: string | null;
  runtime_prior_key?: string | null;
  is_runtime_prior_controlled: boolean;
}
/**
 * One supported NumPyro transform, applied after the base distribution.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "DistributionTransform".
 */
export interface DistributionTransform {
  kind: "affine" | "exp" | "persistence_to_decay";
  loc: number;
  scale: number;
}
/**
 * Semantic parameter-to-runtime-site binding.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CompiledParameterBinding".
 */
export interface CompiledParameterBinding {
  parameter_id: ParameterId;
  coordinates: {
    [k: string]: ParameterCoordinate;
  };
  site_name: string;
  prior_field: string | null;
  flat_index: number;
  site_kind: SiteKind;
  transform: PriorAuthoringTransform;
  construct_names: string[];
  indicator_names: string[];
  component_index: number | null;
  effect_idx: number | null;
  cause_idx: number | null;
}
/**
 * A parameter coordinate identifies a scalar element of a named runtime sample site.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ParameterCoordinate".
 */
export interface ParameterCoordinate {
  site_name: string;
  indices: number[];
}
/**
 * Typed model-spec validation diagnostic.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorValidationResult".
 */
export interface PriorValidationResult {
  /**
   * Name of the parameter that was validated
   */
  parameter: string;
  /**
   * Whether the prior passed validation
   */
  is_valid: boolean;
  code: string;
  origin: "compile" | "prior_predictive";
  severity: "error" | "warning";
  issue?: string | null;
  suggested_adjustment?: string | null;
  related_parameters: string[];
  compiled_site_name?: string | null;
  compiled_flat_index?: number | null;
  supporting_codes: string[];
  repair_scope?: PriorRepairScope | null;
  failure_stage?:
    | (
        | "compiled_parameters"
        | "latent_dynamics"
        | "observation_mean"
        | "observation_sample"
        | "support_violation"
        | "model_build"
        | "prior_sampling"
        | "unknown"
      )
    | null;
  bad_sample_sites: string[];
  bad_manifest_names: string[];
  failing_draw_indices: number[];
  first_bad_time_index?: number | null;
  pathology_certificate?: PriorPathologyCertificate | null;
}
/**
 * Deterministic repair scope for nonlocal prior-validation failures.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorRepairScope".
 */
export interface PriorRepairScope {
  /**
   * Repair-scope family for a nonlocal validation failure
   */
  kind: "dynamics_scc";
  /**
   * Ordered latent constructs included in the minimal repair scope
   */
  construct_names: string[];
}
/**
 * Comparable summary of one validation pathology.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PriorPathologyCertificate".
 */
export interface PriorPathologyCertificate {
  /**
   * Stable certificate family for same-scope retry gating
   */
  kind: "nonfinite_samples" | "dynamics_stability" | "dt_ct_approximation";
  /**
   * Primary severity score. Lower means the pathology improved.
   */
  primary_score: number;
  /**
   * Optional tie-break severity score. Lower means the pathology improved.
   */
  secondary_score?: number | null;
}
/**
 * A joint posterior's draw axes, exact provenance, summaries, and separate assessment.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorArtifact".
 */
export interface PosteriorArtifact {
  draws: PosteriorDrawsInfo;
  provenance: PosteriorProvenance;
  inference_metadata: InferenceMetadata;
  assessment: PosteriorAssessment;
  posterior_marginals?: PosteriorMarginal[] | null;
  posterior_pairs?: PosteriorPair[] | null;
}
/**
 * Axes of aligned joint draws stored in the posterior's fitted payload.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorDrawsInfo".
 */
export interface PosteriorDrawsInfo {
  n_draws: number;
  parameter_shapes: {
    [k: string]: number[];
  };
  /**
   * Time and state axis lengths per retained latent draw.
   */
  latent_shape?: [any, any] | null;
}
/**
 * Exact model and observation versions defining the posterior distribution.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorProvenance".
 */
export interface PosteriorProvenance {
  causal_design: CausalDesignRef;
  compiled_ssm_version: number;
  panel_version: number;
}
/**
 * The workspace and version of the scientific design supporting an inference.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "CausalDesignRef".
 */
export interface CausalDesignRef {
  workspace_id: string;
  version: number;
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
 * Sampling and predictive assessments of a fitted posterior.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "PosteriorAssessment".
 */
export interface PosteriorAssessment {
  ppc: PosteriorPredictiveChecks;
  mcmc_diagnostics?: MCMCDiagnostics | null;
  smc_diagnostics?: SMCDiagnostics | null;
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
 * MCMC diagnostics add parameter-owned convergence checks and plots to the sampler summary.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MCMCDiagnostics".
 */
export interface MCMCDiagnostics {
  num_divergences: number;
  divergence_rate: number;
  tree_depth_mean: number;
  tree_depth_max: number;
  accept_prob_mean: number;
  latent_accept_prob_mean?: number | null;
  parameter_accept_prob_mean?: number | null;
  num_chains?: number | null;
  num_samples?: number | null;
  per_parameter: MCMCParamDiagnostic[];
  trace_data?: TraceData[] | null;
  rank_histograms?: RankHistogram[] | null;
  energy?: EnergyDiagnostics | null;
}
/**
 * These diagnostics assess convergence and sampling precision for one model parameter.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MCMCParamDiagnostic".
 */
export interface MCMCParamDiagnostic {
  parameter: string;
  subject: ParameterRef;
  r_hat: number | null;
  ess_bulk: number | null;
  ess_tail?: number | null;
  mcse_mean?: number | null;
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
 * Trace data groups a parameter's sampled paths across chains for visual inspection.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceData".
 */
export interface TraceData {
  parameter: string;
  subject: ParameterRef;
  chains: TraceChain[];
}
/**
 * A trace chain stores a thinned sequence of parameter draws from one sampling chain.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TraceChain".
 */
export interface TraceChain {
  chain: number;
  values: number[];
}
/**
 * A rank histogram compares parameter ranks across chains to assess mixing.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RankHistogram".
 */
export interface RankHistogram {
  parameter: string;
  subject: ParameterRef;
  n_bins: number;
  expected_per_bin: number;
  chains: RankHistogramChain[];
}
/**
 * This histogram stores one chain's rank counts for a parameter mixing plot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RankHistogramChain".
 */
export interface RankHistogramChain {
  chain: number;
  counts: number[];
}
/**
 * Energy diagnostics assess Hamiltonian sampling through energy distributions and mixing
 * measures.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EnergyDiagnostics".
 */
export interface EnergyDiagnostics {
  energy_hist: EnergyHistogram;
  energy_transition_hist: EnergyHistogram;
  bfmi: number[];
}
/**
 * An energy histogram supplies bin centers and densities for a sampler energy plot.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "EnergyHistogram".
 */
export interface EnergyHistogram {
  bin_centers: number[];
  density: number[];
}
/**
 * SMC diagnostics track particle sampling through its tempering schedule, effective sample
 * sizes, and acceptance rates.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SMCDiagnostics".
 */
export interface SMCDiagnostics {
  beta_schedule: number[];
  ess_history: number[];
  accept_rates: number[];
  n_levels: number;
  n_particles: number;
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
 * A baseline report collects treatment effects and saved scenarios from the fitted model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "BaselineReportArtifact".
 */
export interface BaselineReportArtifact {
  intervention_results: TreatmentEffect[];
  saved_scenarios?: SavedScenario[] | null;
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
 * A saved scenario preserves a labeled causal query and its optional narrative summary.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SavedScenario".
 */
export interface SavedScenario {
  label: string;
  query: ScenarioQuery;
  evaluations: ScenarioEvaluationResult[];
  summary?: string | null;
}
/**
 * A scientific query keeps its identity across model and posterior revisions.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioQuery".
 */
export interface ScenarioQuery {
  start: ScenarioStartInput;
  /**
   * @minItems 1
   */
  clamps: [ScenarioClamp, ...ScenarioClamp[]];
  outcome: ConstructRef;
  readout: ScenarioQueryInput;
  id: ScenarioQueryId;
}
/**
 * Where the forward rollout begins (replaces the rung-2/rung-3 split).
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioStartInput".
 */
export interface ScenarioStartInput {
  /**
   * 'baseline' starts from the population baseline steady state (an interventional, rung-2 query). 'abducted' conditions on the individual's observed evidence and starts from the recovered fitted latent state (a counterfactual, rung-3 query).
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
 * A resolved clamp binds its transport label to a persistent construct identity.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioClamp".
 */
export interface ScenarioClamp {
  /**
   * Latent construct to clamp.
   */
  variable: string;
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
  target: ConstructRef;
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
 * One posterior-specific evaluation and its matching computed outputs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioEvaluationResult".
 */
export interface ScenarioEvaluationResult {
  evaluation: ScenarioEvaluation;
  result: ScenarioResult;
}
/**
 * An evaluation binds a scientific query to one model and exact posterior version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioEvaluation".
 */
export interface ScenarioEvaluation {
  id: ScenarioEvaluationId;
  query_id: ScenarioQueryId;
  model: ModelRef;
  posterior: ArtifactRef;
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
 * Computed outputs reference the evaluation that fixes their query and posterior.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioResult".
 */
export interface ScenarioResult {
  evaluation_id: ScenarioEvaluationId;
  start: ScenarioStartResult;
  /**
   * Display name of the query's outcome at execution time.
   */
  outcome_label: string;
  summary: EffectSummary;
  effect_trajectory?: EffectTrajectoryPoint[] | null;
  trajectory_peak?: EffectTrajectoryPoint | null;
  visualization?: BaselineReportVisualization | null;
  manifest_effects?: {
    [k: string]: number;
  } | null;
  /**
   * Mean reference outcome (baseline steady state or factual forecast).
   */
  reference_mean: number;
  warnings: string[];
}
/**
 * A resolved scenario start records the initial-state source and evidence time.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ScenarioStartResult".
 */
export interface ScenarioStartResult {
  kind: "baseline" | "abducted";
  time_index?: number | null;
  time?: string | null;
  state_source: "baseline_steady_state" | "fitted_latent_paths";
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
 * Saved scenarios preserve the selected queries for a fitted model.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SavedScenariosArtifact".
 */
export interface SavedScenariosArtifact {
  scenarios: SavedScenario[];
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
  move?: MachineMoveSpec | null;
  query?: ToolQuerySpec | null;
  lower_context_id?: string | null;
}
/**
 * A move specification declares an operation on an artifact and its provenance.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MachineMoveSpec".
 */
export interface MachineMoveSpec {
  kind: "run" | "write";
  artifact_id: ArtifactId;
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
  pickle: {
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
 * Measurement definitions with their corresponding causal design and structural plan.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementStructureViewData".
 */
export interface MeasurementStructureViewData {
  causal_design: CausalDesign;
  structural_plan: StructuralPlan;
}
/**
 * Worker outcomes and panel counts derived from the same extraction revision.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MeasurementsData".
 */
export interface MeasurementsData {
  workers: WorkerStatus[];
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
 * A specification with observed likelihood diagnostics from its pinned inputs.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "StatisticalModelSpecData".
 */
export interface StatisticalModelSpecData {
  statistical_model_spec: StatisticalModelSpec;
  authored_priors: {
    [k: string]: PriorProposal;
  };
  resolved_priors: PriorProposal[];
  search_queries?: {
    [k: string]: string;
  } | null;
  validation_warnings?: string[] | null;
  prior_predictive_samples?: {
    [k: string]: number[];
  } | null;
  prior_predictive_diagnostics: PriorPredictiveDiagnostic[];
  structural_plan?: StructuralPlan | null;
  state_equations: StateEquation[];
  parameters: ParameterSpec[];
  likelihood_diagnostics: {
    [k: string]: ModelSpecLikelihoodDiagnostics;
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
 * via the `definition` "ModelSpecLikelihoodDiagnostics".
 */
export interface ModelSpecLikelihoodDiagnostics {
  indicator_id: IndicatorId;
  profile: IndicatorEmpiricalProfile | null;
  histogram: HistogramBin[];
  prior_counts?: number[] | null;
  prior_outside_fraction?: number | null;
}
/**
 * Available artifact projections read from one committed model state.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ArtifactViews".
 */
export interface ArtifactViews {
  raw_data?: RawDataData | null;
  latent_structure?: LatentStructureArtifact | null;
  measurement_structure?: MeasurementStructureViewData | null;
  measurements?: MeasurementsData | null;
  validation_report?: ValidationReportArtifact | null;
  statistical_model_spec?: StatisticalModelSpecData | null;
  posterior?: PosteriorArtifact | null;
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
  legal: Move[];
  auto_running: boolean;
}
/**
 * A run move requests the transition that produces an artifact.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "RunArtifact".
 */
export interface RunArtifact {
  kind: "run";
  artifact_id: ArtifactId;
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
 * A fact source locates supporting content within an artifact version and records its freshness.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FactSource".
 */
export interface FactSource {
  artifact: ArtifactRef;
  pointer: string;
  validity: SourceValidity;
}
/**
 * A fit read contains the canonical posterior and server-composed display findings.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "FitSummary".
 */
export interface FitSummary {
  posterior: PosteriorArtifact;
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
 * A do-operator on one latent variable over a time window.
 *
 * The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
 * the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
 * value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
 * interpolates across the window, and ``trajectory`` tracks a list of values across it.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "LatentClampInput".
 */
export interface LatentClampInput {
  /**
   * Latent construct to clamp.
   */
  variable: string;
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
 * The machine description exposes the artifact graph, storage contracts, and action hierarchy.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "MachineDescription".
 */
export interface MachineDescription {
  artifact_ids: ArtifactId[];
  topological_artifact_order: ArtifactId[];
  topological_transition_order: ArtifactId[];
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
  transition_id: ArtifactId;
  consumes: ArtifactId[];
  produces: ArtifactId[];
  produces_optional: ArtifactId[];
  creation_class: "deterministic" | "batch_llm" | "judgment";
  writable: boolean;
}
/**
 * A model snapshot batches independently sourced aggregates at one committed revision.
 *
 * Authored structure, measurement declarations, specification, and posterior retain their
 * canonical hierarchy. Optional reads represent partial models; each source preserves its
 * own version and freshness. Only cross-artifact ownership and provenance belong here.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "ModelSnapshot".
 */
export interface ModelSnapshot {
  model: ModelRef;
  seq: number;
  state: EpisodeState;
  question?: SourcedQuestionArtifact | null;
  latent_structure?: SourcedLatentStructure | null;
  measurement_structure?: SourcedMeasurementStructureArtifact | null;
  identification?: SourcedIdentifiabilityStatus | null;
  dispositions?: SourcedTupleStructuralItemDisposition | null;
  graph_status: {
    [k: string]: "observed" | "marginalized" | "blocking";
  };
  raw_data?: SourcedRawDataData | null;
  measurements?: SourcedMeasurementsData | null;
  validation_report?: SourcedValidationReportArtifact | null;
  specification?: SourcedStatisticalModelSpecArtifact | null;
  compiled_parameters?: SourcedTupleParameterSpec | null;
  fit?: SourcedFitSummary | null;
  baseline_report?: SourcedBaselineReportArtifact | null;
  saved_scenarios?: SourcedSavedScenariosArtifact | null;
  artifacts: ArtifactFreshness[];
  installed_at: {
    [k: string]: number;
  };
  retracted: ArtifactId[];
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
 * via the `definition` "Sourced_LatentStructure_".
 */
export interface SourcedLatentStructure {
  value: LatentStructure;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_MeasurementStructureArtifact_".
 */
export interface SourcedMeasurementStructureArtifact {
  value: MeasurementStructureArtifact;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_IdentifiabilityStatus_".
 */
export interface SourcedIdentifiabilityStatus {
  value: IdentifiabilityStatus;
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
 * via the `definition` "Sourced_StatisticalModelSpecArtifact_".
 */
export interface SourcedStatisticalModelSpecArtifact {
  value: StatisticalModelSpecArtifact;
  source: FactSource;
}
/**
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_tuple_ParameterSpec__________".
 */
export interface SourcedTupleParameterSpec {
  value: ParameterSpec[];
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
 * A sourced value pairs one model finding with its supporting artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "Sourced_SavedScenariosArtifact_".
 */
export interface SourcedSavedScenariosArtifact {
  value: SavedScenariosArtifact;
  source: FactSource;
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
 * A simulation request declares the initial state, timed clamps, and requested outcome readout.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulateScenarioInput".
 */
export interface SimulateScenarioInput {
  start: ScenarioStartInput;
  /**
   * One or more timed latent clamps composing the scenario.
   *
   * @minItems 1
   */
  clamps: [LatentClampInput, ...LatentClampInput[]];
  /**
   * Outcome construct. Defaults to the workspace’s selected query outcome.
   */
  outcome?: string | null;
  query: ScenarioQueryInput;
}
/**
 * A simulation response includes its reusable scientific query and pinned evaluation.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "SimulateScenarioResult".
 */
export interface SimulateScenarioResult {
  evaluation: ScenarioEvaluation;
  result: ScenarioResult;
  query: ScenarioQuery;
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
 * Promoted traces of the applied transition that produced an artifact version.
 *
 * This interface was referenced by `CausalSSMContracts`'s JSON-Schema
 * via the `definition` "TransitionTraceIndex".
 */
export interface TransitionTraceIndex {
  workspace_id: string;
  artifact_id: ArtifactId;
  version: number;
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
