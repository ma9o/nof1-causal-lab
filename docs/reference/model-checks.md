# Scientific actions and checks

This catalog maps model, data, authoring, inference, and causal-reporting checks onto four scientific actions: `edit_model`, `prepare_data`, `fit`, and `simulate`. The implementation was inspected on 2026-09-15. The [action contracts](scientific-actions.md) describe the implemented first migration and remaining boundaries; the inventory records calculations and their owners. Closely related validators are grouped by scientific purpose. Costs are qualitative implementation estimates, not benchmark timings. Repository tests, infrastructure health checks, and LLM authoring work are outside this inventory.

| Action | Inputs | Outputs and automatic findings |
| --- | --- | --- |
| `edit_model` | A model revision and changes to constructs, mechanisms, measurements, fixed/free parameters, or probability laws. | A revised model; model-only validity and capability findings; refreshed compatibility findings for available datasets. |
| `prepare_data` | Sources or an existing dataset, selection/transformation instructions, and measurement definitions when extracting indicators. | A versioned dataset with its derivation, profiles, and data checks; compatibility findings for an available model. |
| `fit` | A model revision, dataset, and inference settings. | A model revision with conditioned joint uncertainty; initialization and sampler diagnostics; posterior summaries and optional PSIS-LOO. |
| `simulate` | A model revision, simulation design and settings, optional interventions or paired contrasts, and optional comparison data. | Requested parameter draws, trajectories, observations, and scientific summaries; applicable simulation measurements and causal-result prerequisites. |

Measurement meaning, extraction rules, recording assumptions, and support windows belong to the model specification. Data preparation evaluates those definitions against sources. The current [ingestion](../pipeline/ingestion.md) and [extraction](../pipeline/extraction.md) workflows provide the underlying operations. Changing an extraction definition can require preparing affected observations again; changing a parameter law alone can reuse them.

Model-only findings depend on a model revision. Data-only findings depend on a dataset revision. Compatibility findings depend on both and refresh when either relevant input changes. Missing inputs leave a check unevaluated. Check findings and action prerequisites accompany results; these actions do not impose a required sequence of authoring admissions.

The cost of producing a check's inputs is often much greater than the cost of calculating the check. A sampler diagnostic needs a fit, but reading another diagnostic from an existing fit does not require fitting again. Many simulation measurements similarly share one batch. The current implementation separates the [array-based prior measurements](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py) from the [compilation, simulation, and admission decisions](../../apps/data-pipeline/src/nof1_causal_lab/recipes/construct_authoring.py).

The cost classes used below distinguish producing evidence from measuring it:

| Class | Work required | Main cost drivers |
| --- | --- | --- |
| A: metadata | Validate definitions, references, support metadata, and prerequisites. | Model size, distribution metadata, and imports; no trajectory simulation or optimization. |
| B: data or numerical analysis | Scan observations, join indicator series, run graph identification, or perform small matrix calculations. | Panel size, indicator pairs, graph complexity, and matrix dimension. Graph identification is not guaranteed to remain cheap as graphs grow. |
| C: summarize existing draws | Compute quantiles, correlations, convergence summaries, or importance-sampling diagnostics. | Number of draws, parameters, times, and indicators; no new fit or forward simulation. |
| D: forward simulation | Generate a batch of nonlinear trajectories and observations. | First JAX compilation, draw count, integration steps, nonlinear dynamics, and observation family. |
| E: iterative initialization | Run Pathfinder/MAP and latent-mode optimization for sampler initialization. | Repeated likelihood/gradient evaluations, inner solves, curvature calculations, and compilation. |
| F: posterior inference | Run the production particle sampler. | Chains, iterations, particles, time points, state dimension, and initialization. |

These are work categories, not universal timing rankings. A large data join can outlast a small simulation. First compilation and loading large stored arrays can dominate an otherwise inexpensive check. Classes D–F describe input-producing computations; the individual diagnostic reductions below usually cost A–C once those computations have finished.

## Edit model

The model-only checks below need no observed values, trajectory simulation, or new posterior fit. Some require complete model definitions or executable probability laws, so an incomplete edit cannot produce every finding. An edit triggers affected checks and refreshes the model–data checks cataloged under [Prepare data](#prepare-data) when compatible observations exist. Fast findings can return on submission; graph identification or compiler work that takes longer can finish asynchronously.

### Model and compiler checks

| Check family | Current checks | Cost | Current owner |
| --- | --- | --- | --- |
| Definition and reference integrity | Allowed fields and enum values; finite literals; unique IDs/names; valid entity and distribution references; referenced parameters; valid outcome; increasing trajectory times. | A | [ModelSpec](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_spec.py) and its component schemas. |
| Graph consistency | Connected graph; one edge per endpoint pair; exogenous targets and temporal edge restrictions; no contemporaneous cycles. | A–B: graph traversal | [Construct/edge validators](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/construct.py). |
| Mechanism consistency | Intrinsic dynamics reference their owning construct; edge expressions reference their source and declared parents; node ownership of potentials; coefficient role/support consistency. | A | [Model references](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_spec.py), [mechanisms](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/mechanism.py), and [expressions](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/expressions.py). |
| Measurement definition | Dtype/aggregation compatibility; recording/extraction compatibility; window durations; ordinal/category metadata; computed-expression syntax and declared source columns; semantic collisions and duplicate measurement definitions. | A | [Indicator](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/indicator.py), [observation semantics](../../apps/data-pipeline/src/nof1_causal_lab/utils/observation_semantics.py), and [model checks](../../apps/data-pipeline/src/nof1_causal_lab/models/model_checks.py). |
| Observation-law validity | Constructor signatures, supported argument expressions and links, compatible indicator dtype, and required measurement coefficients. | A | [Likelihood](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/likelihood.py) and [numerical execution validation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/numerics.py). |
| Distribution membership | Scalar versus joint ownership; event-coordinate identity and dimension; trajectory-law time coordinates. | A–B: metadata and parameter binding | [Distribution memberships](../../apps/data-pipeline/src/nof1_causal_lab/models/model_distributions.py). |
| Execution completeness and capability | Measurement clock and indicators; required priors, initial-state and innovation coefficients; supported emissions; retained-state measurement coverage; unsupported static-target edges and marginalization; manifest/state count restriction. | A–B: model traversal and compilation | [Model structure](../../apps/data-pipeline/src/nof1_causal_lab/models/model_structure.py), [model checks](../../apps/data-pipeline/src/nof1_causal_lab/models/model_checks.py), and [numerics](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/numerics.py). |
| Location/scale anchors | Location anchor per retained state; fixed loading or categorical slope scale anchor; redundant free categorical loadings. | A–B: coefficient/array inspection | [Anchor certificates](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/structural/closure.py). This is an algebraic certificate under the supported model conventions, not a Hessian calculation. |
| Prior binding and support | Missing/inactive/duplicate bindings; scalar coordinate laws; positive and correlation support; fixed values; transform eligibility, reference intervals, and degenerate transformed priors. | A–B: distribution inspection and transforms | [Prior validation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/priors.py) and [prior compilation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/prior_compilation.py). |
| DT-to-CT conversion diagnostic | `dt_ct_approximation_warning`: constructs a reference matrix from decay/linear-effect priors, compares elementwise conversion with a matrix logarithm, and reports mismatch or a non-real logarithm. | B: dense matrix logarithm, potentially repeated across edges; roughly cubic matrix cost per evaluation | [Conversion diagnostic](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/prior_compilation.py). This is reachable through [input compilation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/inputs.py); it is not a nonlinear trajectory simulation or a nonlinear identification certificate. |
| Graph-based causal identification | Time-expanded graph projection and do-calculus for treatments against the default outcome; positive identification or blocking context. Linear-IV identification is disabled for ModelSpec. | B, variable with graph complexity | [Model identification](../../apps/data-pipeline/src/nof1_causal_lab/models/identification.py) and [ID implementation](../../apps/data-pipeline/src/nof1_causal_lab/utils/identifiability.py). Automatically derived after model changes by the [derivation cascade](../../apps/data-pipeline/src/nof1_causal_lab/machine/derivations.py). |
| Engine-specific admissibility | Particle inference requires nonlinear Euler–Maruyama transitions, Gaussian process diffusion, and point observations. Prior prediction also checks Gaussian process diffusion. | A once model support is compiled | [Particle problem](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/problem.py) and [predictive runtime](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py). |

The DT-to-CT diagnostic is an important qualification to any statement that all non-initialization diagnostics avoid linear reasoning: it explicitly analyzes a reference linear component matrix. Its output should not be interpreted as a statement about the full nonlinear system. The [reference prior value](../../apps/data-pipeline/src/nof1_causal_lab/prior_distributions.py) is also explicitly a transformed base-law mean where applicable, rather than necessarily the transformed distribution's expectation.

## Prepare data

This action imports, selects, transforms, or extracts data and returns an explicit dataset revision. Sources can be imported and profiled before a model exists; indicator extraction requires its measurement definitions. Extraction, particularly semantic extraction with an LLM, is a separate input-producing cost and should remain explicit when a model edit requires new observations. The [current extraction workflow](../pipeline/extraction.md) produces observations with indicator identity, values, and time/support metadata.

### Data summaries and compatibility checks

Counts, timestamp ranges, numeric summaries, and missing-value profiles can be computed from data alone. Many existing audit verdicts additionally require model declarations, as listed below. These joint findings belong to the model–dataset pair and can accompany either `edit_model` or `prepare_data`. The ten named audit rules are registered in [validation rules](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/rules.py); their numerical implementations are in [validation checks](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/checks.py).

| Check | Calculation | Inputs beyond the data being checked | Added cost |
| --- | --- | --- | --- |
| Extraction-output validity | Output shape, known indicators/windows, duplicate window–indicator pairs, dtype compatibility, ordinal codes. [Worker validator](../../apps/data-pipeline/src/nof1_causal_lab/workers/schemas.py). | Measurement definitions and expected extraction windows. | A–B: scan returned rows. |
| Empty panel / unknown indicator | No usable extraction data, or observations outside the pinned measurement design. [Validation entry point](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/flow.py). | Declared indicators for the unknown-indicator verdict. | A–B. |
| `missing` | Indicator has no extracted rows. | Expected indicators. | B: row counts. |
| C5d data availability | Indicator has no observed values. Reported by model/data compatibility after submission. | Expected indicators; no parameter draws. | B: counts; A once counts exist. |
| `no_numeric` | Indicator rows contain no usable numeric values. | None for an existing indicator series. | B: conversion and scan. |
| `timestamps` | Counts and severity for timestamps that cannot be parsed. | None beyond the time-column designation. | B: timestamp parsing. |
| `sample_size` | Fewer than 10 observations. | None for an existing indicator series; fixed threshold. | B: count. |
| `variance` | Zero variance for time-varying constructs; static constructs are exempt. | Construct temporal status. | B: scan. |
| `dtype_range` | Binary support; negative/fractional counts; continuous outliers beyond three interquartile ranges. | Declared measurement dtype. | B: scans and quantiles. |
| `time_coverage` | Study duration below ten model-clock periods. | Model clock and construct temporal status. | B: timestamp range. |
| `timestamp_gaps` | Largest consecutive gap above five model-clock periods. | Model clock and construct temporal status. | B: timestamp sorting. |
| `hallucination_signals` | Dominant duplicate value and arithmetic-sequence patterns. These are pattern warnings, not proof of fabrication. | Measurement dtype for pattern exemptions. | B: counts and sorting. |
| `construct_correlations` | Daily alignment of indicator pairs within each construct; negative correlation warning when at least ten days align. | Indicator-to-construct assignments. | B, potentially larger: repeated grouping/joins. |
| Observation-support validity | Actual values in emission support; ordinal/categorical level metadata. [Observation support](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/observation_support.py). | Observation laws and level definitions. | B: data scan. |
| Fit preflight: standardization | Array shape; marked standardized columns have appropriate mean and SD. [Preflight](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/preflight.py). | Expected array layout and standardization declarations. | B: column summaries. |
| Fit preflight: location/prior reach | For eligible unstandardized identity-link location channels with supported Normal-like mean priors, observed mean lies within six prior SDs. | Observation links, location laws, and standardization declarations. | B: summaries and prior inspection; no simulation. |
| Exact-observation consistency | Direct-state Delta observation support; finite/missing values; conflicting exact readings for one state/time. [Conditioning](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/conditioning.py). | Exact-observation mappings to states. | B: observation-array scan. |

The audit also computes profiles such as quantiles, zero fraction, whether values are integers, and variance-to-mean ratio. These are descriptive outputs from the same observations, with B-level cost; they are not additional simulation experiments. See [profile construction](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/rules.py).

The fit-preflight and exact-observation rows are classified here because they validate the model/data contract without fitting. They run after model/data submissions and again at fit preparation. C5d now runs with model/data compatibility when observations are available. Fit readiness still requires these findings to match the actual prepared inputs used for execution.

## Fit

This action conditions the selected model on the selected dataset. [Conditioning](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/persistence.py) writes the full joint empirical law for parameters and retained latent trajectories into another `ModelSpec` revision; the input revision retains its original laws. The action also returns diagnostics of initialization, sampler execution, and the stored fit result.

Posterior summaries and PSIS-LOO belong here because they consume posterior draws or likelihood factors without generating new predictive trajectories. The current [inference orchestration](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/inference/fit.py) computes sampler summaries and optionally LOO after fitting, then runs posterior-predictive checks. In the proposed interface, forward prediction is requested through `simulate`.

### Sampler diagnostics and posterior summaries

| Diagnostic | Evidence needed | Input-producing cost | Added cost and current behavior |
| --- | --- | --- | --- |
| R-hat, effective sample size, tail ESS, mean Monte Carlo error | Parameter draws grouped by chain | F: existing particle fit | C: chain summaries; no refit. The field named `ess_bulk` currently comes from NumPyro's `n_eff`; tail ESS and MCSE come from ArviZ. [Diagnostic extraction](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/types.py). |
| Trace and rank plots | Existing chain draws | Same F result | C: thinning/ranking/histograms. [Diagnostic visualizations](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/diagnostics_viz.py). |
| Marginal/pair plots and credible intervals | Existing posterior draws | Same F result | C: density/histogram/interval summaries and selected pair samples. Visual evidence, not a separate convergence verdict. [Posterior result](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/types.py). |
| Particle-kernel movement and acceptance | Telemetry collected during sampling | F; tracking adds per-step work/storage | A–C to summarize parameter acceptance, latent updates/frozen fraction, sign-flip acceptance when enabled, step-size adaptation, and complete-log-posterior histories. [Fit telemetry](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/methods/marginal_particle_gibbs/fit.py). |
| Optional particle-identity / parameter-movement groups | Optional traces enabled before sampling | F plus optional trace allocation/computation | A–C to summarize reference-path hits, selected-particle uniqueness, latent movement, and parameter jumps. These cannot all be reconstructed if their traces were not collected. [Metric flags](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/methods/marginal_particle_gibbs/diagnostics.py). |
| PSIS-LOO, Pareto-k, bad-k count, ELPD and uncertainty | Saved exact emission log factors on joint parameter/state draws | F produces the factors | C, potentially substantial over many draws/rows: importance-weight calculations, no leave-one-out refits. It assesses measurement-row interpolation using other rows, including future rows. Unavailable for exact-observation constraints. [LOO calculation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/types.py). |

### Initialization diagnostics and conditional helpers

| Existing computation or helper | Cost | Scope |
| --- | --- | --- |
| Pathfinder finite starts, ELBO comparisons, initialization selection, setup/runtime timings | E to run the initializer; A–C to summarize its output | Sampler initialization. Multiple starts and inner likelihood solves can make this expensive. [Parameter warmup](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/warmup/parameter_warmup.py). |
| MAP/IEKS optimizer status, gradient/step convergence, inner solver diagnostics, curvature/Cholesky information | E; optional Hessian/covariance work can add substantial cost | Sampler initialization and proposal preparation. These use local linearization/Gaussian approximations and are not parameter-identification certificates. [MAP implementation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/warmup/map.py) and [initialization transitions](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/targets/transitions.py). |
| HMC divergence, step-count and energy/BFMI summaries | C if the relevant traces exist | Generic extraction helpers remain in [inference types](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/types.py), but the current production particle runner does not supply those HMC fields. Do not count them as active particle diagnostics. |
| Tempered-SMC beta/ESS/acceptance summaries | A–C if that telemetry exists | Conditional helper in [inference types](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/types.py); not the active production particle-Gibbs diagnostic set. |
| LOO-PIT | No active computation found | The [LOO schema](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/posterior_diagnostics.py) has a field; current LOO extraction does not populate it. |

The [linearization guard](../../apps/data-pipeline/tests/models/ssm/test_linearization_init_only.py) restricts references to the linearized Laplace backend to initialization. This is a static import/reference guard, not a test proving that every other calculation is free of linear assumptions; the compiler's matrix-log diagnostic above illustrates that distinction.

There is no production posterior-contraction, prior/likelihood power-scaling, or simulation-based-calibration check wired into these runtime paths. [Reachability's references to contraction and power-scaling](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py) describe where practical-identification assessment belongs, rather than invoking an implementation. The removed C4a edge-detectability screen should likewise not be counted among current checks.

## Simulate

This action samples from the selected model revision's current probability laws and generates the requested quantities. `ModelSpec` uses one [distribution representation](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_spec.py); [conditioning](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/persistence.py) updates that representation while retaining joint dependence. Prior and posterior describe a distribution's relationship to conditioning data, so both use the same proposed action.

The request specifies which quantities to regenerate or condition on, the time and observation design, noise treatment, optional interventions or paired contrasts, optional comparison data, and draw settings. This distinguishes replicating a study, forecasting from inferred states, and evaluating interventions. The result records the model revision and its conditioning history, simulation settings, and comparison dataset. Joint parameter/state dependence must be preserved wherever the requested simulation uses those states.

### Simulation measurements

The explicit simulation action and optional authoring recipe share the nonlinear [predictive runtime](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py) and [Diffrax forward solver](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/simulator.py). The table retains current check labels and rules so the inventory remains traceable. The action samples current model laws and records a separate simulation report. The optional authoring recipe retains its admission decisions.

| Check or output | Required evidence | Evidence-producing cost | Measurement and current rule |
| --- | --- | --- | --- |
| C1a finiteness | Latent trajectories | D: shared simulation batch. | C: finite-value scan. Current prior battery treats this as a hard failure. |
| C1b confinement | Early/late latent amplitudes | Same D batch. | C: path summaries and explosive-draw fraction. |
| C2 latent scale | Latent draws over the latter part of the window | Same D batch. | C: robust scale and quantiles against the latent-scale convention. |
| C3 resolvability | Decay-parameter draws and observation times | Parameter sampling only; currently obtained inside the prior simulation batch. | B–C: reciprocal decay and gap/span comparisons. |
| C4b edge overwhelm | Paired edge-on and edge-off trajectories with the same parameter draws and Brownian noise | Shared D batch plus another D trajectory batch per checked incoming edge. | C: paired path-variation summaries. |
| C4c saturation | Simulated parent values and Hill EC50/exponent draws | Shared D batch. | C: evaluate Hill response across the simulated range. |
| C5a location reach | Replicated observations and comparison data | Shared D batch. | C: replicated-versus-observed location summaries. |
| C5b width | Replicated observations and comparison data | Shared D batch. | C: replicated-versus-observed spread comparison. |
| C5c transmission | Simulated emission means/probabilities and conditional variances | Shared D batch plus analytical family moments. | C: temporal signal-variation share; time-varying constructs only. |
| Predictive interval coverage | Replicated observations and comparison data on a compatible observation design | Shared D batch. | C: 95% interval coverage; current PPC flags coverage below 70% or above 98%. |
| Residual autocorrelation | Same replicated observations and comparison data | Same D batch. | C: lag-one residual correlation; current threshold is absolute correlation above 0.3. The lag is adjacent retained observations, not fixed elapsed time. |
| Predictive spread ratio | Same replicated observations and comparison data | Same D batch. | C: average per-draw temporal SD divided by observed SD; current PPC flags ratios outside one-third to three. |
| Replicated test statistics and overlays | Same replicated observations and comparison data | Same D batch. | C: observed versus replicated mean, SD, minimum, maximum, tail proportions, quantile bands, and selected trajectories. |

C1–C5 calculations and current authoring decisions are in [reachability](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py); the four predictive comparison output groups are in [posterior predictive checks](../../apps/data-pipeline/src/nof1_causal_lab/models/posterior_predictive.py). Their measurements are broadly reusable with suitable draws and designs. Simulation reports record their context and explicit confinement criteria; acceptance rules belong to the optional authoring recipe: a posterior finding does not automatically warrant the same judgment as a prior-design finding. Comparisons also need to distinguish data used in fitting from held-out data.

C3 remains a design screen based on `tau = 1 / decay`, not a full nonlinear relaxation analysis or an empirical parameter-identification test. The simulation battery has nine checks; C5d is owned by [Prepare data](#prepare-data). C1a is hard and the other eight findings are soft in the optional authoring recipe under the current [CHECK_MODES and stage_outcome](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py).

### Existing execution costs and result prerequisites

The current [prior battery runner](../../apps/data-pipeline/src/nof1_causal_lab/recipes/construct_authoring.py) compiles and simulates a candidate; standard authoring requests 200 draws. Its [closed-loop recheck and full-model barrier](../../apps/data-pipeline/src/nof1_causal_lab/recipes/construct_authoring.py) repeat the battery on a different assembled model. The full-model barrier shares one base simulation across constructs, while edge-off contrasts add simulations. Per-construct admission, loop closure, and repair can multiply the D-level work. [Admission timing records](../../apps/data-pipeline/src/nof1_causal_lab/recipes/construct_authoring.py) separate compilation, prediction, check phases, and the admission decision.

The [simulation action](../../apps/data-pipeline/src/nof1_causal_lab/actions/simulate.py) uses 100 draws by default and passes one generated batch to the [predictive measurements](../../apps/data-pipeline/src/nof1_causal_lab/models/posterior_predictive.py). Producing these paths adds D-level work even when the fit already exists. The nonlinear forward simulations use the declared emission families and retain numerical integration error.

| Result prerequisite | Required evidence | Added cost |
| --- | --- | --- |
| Causal-result certification | Identified treatment/outcome, matching model revision, committed production inference evidence, nonlinear transition target, and retained joint uncertainty. [Causal proofs](../../apps/data-pipeline/src/nof1_causal_lab/models/causal_proofs.py). | A–B: metadata, fingerprints, and log reads. This governs causal reporting; it does not generate or measure simulated paths. |

The [steady-state helper](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/steady_state.py) performs a nonlinear root solve for scenario initialization, but returns only the solution value with solver throwing disabled. It does not currently expose a convergence or stability report, so this catalog does not count it as an implemented equilibrium diagnostic.

Across all four actions, each check declares its required inputs. Applicable checks return with the action's results; expensive additional simulations or fits remain explicit work. Reuse depends on unchanged relevant model, data, design, and numerical inputs. An inexpensive reduction such as R-hat still needs sampling records for the model being assessed.
