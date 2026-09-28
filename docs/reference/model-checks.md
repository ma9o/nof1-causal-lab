# Scientific actions and checks

This catalog maps model, data, authoring, inference, and causal-reporting checks onto four scientific actions: `edit_model`, `prepare_data`, `fit`, and `simulate`. The execution policy reflects the incremental action workflow. The [action contracts](scientific-actions.md) describe dispatch, conditional checks and publication; the inventory records calculations and their owners. Closely related validators are grouped by scientific purpose. Costs are qualitative implementation estimates, not benchmark timings. Repository tests, infrastructure health checks, and LLM authoring work are outside this inventory.

| Action | Inputs | Outputs and automatic findings |
| --- | --- | --- |
| `edit_model` | A model revision and changes to constructs, mechanisms, measurements, fixed/free parameters, or probability laws. | A revised model; model-only validity and capability findings; refreshed compatibility findings and automatic predictive checks on a compatible panel. |
| `prepare_data` | Uploaded files plus preparation instructions, or one recorded simulation replicate. | Model-independent observations, schema/provenance and numerical data checks. |
| `fit` | A model revision, dataset, and inference settings. | A model revision with conditioned joint uncertainty; initialization and sampler diagnostics; posterior summaries and optional PSIS-LOO. |
| `simulate` | Model revision, design and optional comparison panel. | Generated trajectories, dynamics/emission checks, optional prior/posterior predictive comparisons and certified causal summaries. |

Measurement meaning, extraction rules, recording assumptions, and support windows belong to the model specification. Data preparation evaluates those definitions against sources. The current [ingestion](../pipeline/ingestion.md) and [extraction](../pipeline/extraction.md) workflows provide the underlying operations. Changing an extraction definition can require preparing affected observations again; changing a parameter law alone can reuse them.

Model-only findings depend on a model revision. Data-only findings depend on a dataset revision. Compatibility findings depend on both and refresh when either relevant input changes. Missing inputs leave a check unevaluated. Check findings and action prerequisites accompany results; these actions do not impose a required sequence of authoring admissions.

The cost of producing a check's inputs is often much greater than the cost of calculating the check. A sampler diagnostic needs a fit, but reading another diagnostic from an existing fit does not require fitting again. Many simulation measurements similarly share one batch. The implementation separates the [array-based measurements](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py) from [shared generation and reduction](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/simulation.py) and [action-owned check selection](../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py).

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

The model-only checks below need no observed values, trajectory simulation, or new posterior fit. Some require complete model definitions or executable probability laws, so an incomplete edit cannot produce every finding. An edit triggers affected model checks and refreshes the model–data compatibility findings below when prepared observations exist. Numerical data profiles are produced only by `prepare_data`. All findings return through action polling after atomic publication. The same action also runs [automatic predictive checks](scientific-actions.md#automatic-predictive-checks) when their prerequisites hold.

### Model and compiler checks

| Check family | Current checks | Cost | Current owner |
| --- | --- | --- | --- |
| Definition and reference integrity | Allowed fields and enum values; finite literals; unique IDs/names; valid entity and distribution references; referenced parameters; valid outcome; increasing trajectory times. | A | [ModelSpec](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_spec.py) and its component schemas. |
| Graph consistency | Connected graph; one edge per endpoint pair; exogenous targets and temporal edge restrictions; no contemporaneous cycles. | A–B: graph traversal | [Construct/edge validators](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/construct.py). |
| Mechanism consistency | Intrinsic dynamics reference their owning construct; edge expressions reference their source and declared parents; node ownership of potentials; coefficient role/support consistency. | A | [Model references](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_spec.py), [mechanisms](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/mechanism.py), and [expressions](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/expressions.py). |
| Measurement definition | Dtype/aggregation compatibility, window durations, ordinal/category metadata and generative measurement definitions. Extraction instructions and computed scoring rules are validated by data preparation. | A | [Indicator](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/indicator.py), [observation semantics](../../apps/data-pipeline/src/nof1_causal_lab/utils/observation_semantics.py), and [model checks](../../apps/data-pipeline/src/nof1_causal_lab/models/model_checks.py). |
| Observation-law validity | Constructor signatures, supported argument expressions and links, compatible indicator dtype, and required measurement coefficients. | A | [Likelihood](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/likelihood.py) and [numerical execution validation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/numerics.py). |
| Distribution membership | Scalar versus joint ownership; event-coordinate identity and dimension; trajectory-law time coordinates. | A–B: metadata and parameter binding | [Distribution memberships](../../apps/data-pipeline/src/nof1_causal_lab/models/model_distributions.py). |
| Execution completeness and capability | Measurement clock and indicators; required priors, initial-state and innovation coefficients; supported emissions; retained-state measurement coverage; unsupported static-target edges and marginalization; manifest/state count restriction. | A–B: model traversal and compilation | [Model structure](../../apps/data-pipeline/src/nof1_causal_lab/models/model_structure.py), [model checks](../../apps/data-pipeline/src/nof1_causal_lab/models/model_checks.py), and [numerics](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/numerics.py). |
| Location/scale anchors | Location anchor per retained state; fixed loading or categorical slope scale anchor; redundant free categorical loadings. | A–B: coefficient/array inspection | [Anchor certificates](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/structural/closure.py). This is an algebraic certificate under the supported model conventions, not a Hessian calculation. |
| Prior binding and support | Missing/inactive/duplicate bindings; scalar coordinate laws; positive and correlation support; fixed values; transform eligibility, reference intervals, and degenerate transformed priors. | A–B: distribution inspection and transforms | [Prior validation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/priors.py) and [prior compilation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/prior_compilation.py). |
| DT-to-CT conversion diagnostic | `dt_ct_approximation_warning`: constructs a reference matrix from decay/linear-effect priors, compares elementwise conversion with a matrix logarithm, and reports mismatch or a non-real logarithm. | B: dense matrix logarithm, potentially repeated across edges; roughly cubic matrix cost per evaluation | [Conversion diagnostic](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/prior_compilation.py). This is reachable through [input compilation](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/inputs.py); it is not a nonlinear trajectory simulation or a nonlinear identification certificate. |
| Graph-based causal identification | Time-expanded graph projection and do-calculus for treatments against the default outcome; positive identification or blocking context. Linear-IV identification is disabled for ModelSpec. | B, variable with graph complexity | [Model identification](../../apps/data-pipeline/src/nof1_causal_lab/models/identification.py) and [ID implementation](../../apps/data-pipeline/src/nof1_causal_lab/utils/identifiability.py). Refreshed when its consumed inputs change by the [action evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py). |
| Engine-specific admissibility | Particle inference requires nonlinear Euler–Maruyama transitions, Gaussian process diffusion, and point observations. Prior prediction also checks Gaussian process diffusion. | A once model support is compiled | [Particle problem](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/problem.py) and [predictive runtime](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py). |

The DT-to-CT diagnostic is an important qualification to any statement that all non-initialization diagnostics avoid linear reasoning: it explicitly analyzes a reference linear component matrix. Its output should not be interpreted as a statement about the full nonlinear system. The [reference prior value](../../apps/data-pipeline/src/nof1_causal_lab/prior_distributions.py) is also explicitly a transformed base-law mean where applicable, rather than necessarily the transformed distribution's expectation.

### Model–data compatibility

These findings describe a selected model/dataset pair. `edit_model` checks them when a panel exists, using its saved numerical profile. `fit` verifies the exact selected pair before inference, even when it is historical. They never trigger extraction or alter the dataset.

| Check | Calculation and prerequisites | Added cost |
| --- | --- | --- |
| Observation binding | Model indicator IDs must exist in the data schema; dtype, summary, codebooks and resolved windows must agree. Extra data variables may remain unused. [Binding checks](../../apps/data-pipeline/src/nof1_causal_lab/actions/data_checks.py). | A: schema comparison. |
| `construct_correlations` | Daily alignment of indicators assigned to the same construct; negative correlation warning with at least ten aligned days. [Compatibility rules](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/rules.py). | B: grouping and joins. |
| Observation support | Values inside the emission support and valid ordinal/categorical levels. [Observation support](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/observation_support.py). | B: data scan. |
| Fit preflight: standardization | Array shape and appropriate mean/SD for columns declared standardized. [Preflight](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/preflight.py). | B: column summaries. |
| Fit preflight: location/prior reach | For supported unstandardized identity-link channels with Normal-like location priors, observed mean within six prior SDs. | B: summaries and prior inspection; no simulation. |
| Exact-observation consistency | Delta-observation support, missing/finite values and conflicting exact readings for one state/time. [Conditioning](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/conditioning.py). | B: observation-array scan. |

## Prepare data

This action produces observations from uploaded files or one recorded simulation replicate. It requires no model. Files use a versioned [preparation spec](../pipeline/extraction.md#datapreparationspec), ingestion, deterministic extraction and semantic worker fan-out. A simulation source already has observations and its own schema, so preparation materializes it directly. Both branches run the same numerical checks, using the dataset's dtypes, windows and codebooks.

### Data definitions and numerical checks

| Check | Calculation and data-owned inputs | Added cost |
| --- | --- | --- |
| Preparation definition | Unique variable IDs, valid windows/codebooks, dtype/summary and recording/extraction compatibility, computed-expression syntax and declared columns. [Preparation schema](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/data_preparation.py). | A: schema validation. |
| Scoring semantics | Warnings when the scoring rubric implies a different aggregation. | A: instruction/aggregation comparison. |
| Extraction-output validity | Known variables/windows, dtype, discrete levels and duplicate window/variable pairs. [Worker validator](../../apps/data-pipeline/src/nof1_causal_lab/workers/schemas.py). | A–B: returned-row scan. |
| Empty or undeclared observations | No usable data, or observed variable IDs absent from the dataset schema. [Profile construction](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/flow.py). | A–B. |
| `missing` and C5d availability | Declared variable has no rows or no observed values; uses only expected data variables. | B: counts. |
| `no_numeric` / non-finite values | No usable numeric values, or non-finite emitted values. | B: conversion and scan. |
| `timestamps` | Unparseable timestamps and severity. | B: parsing. |
| `sample_size` | Fewer than ten observations. | B: count. |
| `variance` | Zero variance warning; no assumption about construct dynamics. | B: scan. |
| `dtype_range` and codebook bounds | Binary support, negative/fractional counts, ordinal/categorical code bounds, continuous outliers beyond three interquartile ranges. | B: scans and quantiles. |
| `time_coverage` | Duration below ten declared observation-window periods. | B: timestamp range. |
| `timestamp_gaps` | Largest consecutive gap above five observation-window periods. | B: sorting. |
| `hallucination_signals` | Dominant duplicate values and arithmetic-sequence patterns, with dtype exemptions. Pattern warnings are not proof of fabrication. | B: counts and sorting. |

The [data rules](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/validation/rules.py) also compute descriptive quantiles, zero fraction, integer status and variance-to-mean ratio. Semantic extraction is a separate input-producing cost before these checks. Preparation runs no specification checks, causal identification, fitting, prior reach or predictive simulation. The resulting data profile remains meaningful when the current model changes or is absent.

## Fit

This action conditions the selected model on the selected dataset. [Conditioning](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/inference/persistence.py) writes the full joint empirical law for parameters and retained latent trajectories into another `ModelSpec` revision; the input revision retains its original laws. The action also returns diagnostics of initialization, sampler execution, and the stored fit result.

Posterior summaries and PSIS-LOO belong here because they consume posterior draws or likelihood factors without generating new predictive trajectories. The current [inference orchestration](../../apps/data-pipeline/src/nof1_causal_lab/flows/transitions/inference/fit.py) computes sampler summaries and optionally LOO after fitting, without producing predictive trajectories. Forward prediction is requested through `simulate`; later edits can run automatic predictive checks of the fitted laws.

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

The [request](scientific-actions.md#simulation) takes a model revision, an absolute end time, an optional start and optional timestamped interventions. The framework supplies the grid and sampling settings and always generates stochastic states and observations. The starting state comes from the current joint law, preserving parameter/state dependence. The result records the model reference, requested design and resolved execution coordinates; observed-data comparisons live in the automatic check report.

### Simulation measurements

Automatic edit/data checks and explicit simulation share the [current-law generator](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/simulation.py), nonlinear [predictive runtime](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py) and [Diffrax solver](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/simulator.py). All measure the full authored model. Findings describe evidence; they do not accept or reject constructs.

| Check or output | Required evidence | Evidence-producing cost | Measurement and current rule |
| --- | --- | --- | --- |
| C1a finiteness | Latent trajectories | D: shared simulation batch. | C: finite-value scan. Non-finite paths produce a failed finding; dependent reductions are unevaluated and the edit still saves. |
| C1b confinement | Early/late latent amplitudes | Same D batch. | C: path summaries and explosive-draw fraction. |
| C2 latent scale | Latent draws over the latter part of the window | Same D batch. | C: robust scale and quantiles against the latent-scale convention. |
| C3 resolvability | Decay-parameter draws and observation times | Parameter sampling only; obtained inside the shared simulation batch. | B–C: reciprocal decay and gap/span comparisons. |
| C4b edge overwhelm | Paired edge-on and edge-off trajectories with the same parameter draws and Brownian noise | Shared D batch plus another D trajectory batch per checked incoming edge. | C: paired path-variation summaries. |
| C4c saturation | Simulated parent values and Hill EC50/exponent draws | Shared D batch. | C: evaluate Hill response across the simulated range. |
| C5a location reach | Replicated observations and comparison data | Shared D batch. | C: replicated-versus-observed location summaries. |
| C5b width | Replicated observations and comparison data | Shared D batch. | C: replicated-versus-observed spread comparison. |
| C5c transmission | Simulated emission means/probabilities and conditional variances | Shared D batch plus analytical family moments. | C: temporal signal-variation share; time-varying constructs only. |
| Predictive interval coverage | Replicated observations and comparison data on a compatible observation design | Shared D batch. | C: 95% interval coverage; current PPC flags coverage below 70% or above 98%. |
| Residual autocorrelation | Same replicated observations and comparison data | Same D batch. | C: lag-one residual correlation; current threshold is absolute correlation above 0.3. The lag is adjacent retained observations, not fixed elapsed time. |
| Predictive spread ratio | Same replicated observations and comparison data | Same D batch. | C: average per-draw temporal SD divided by observed SD; current PPC flags ratios outside one-third to three. |
| Replicated test statistics and overlays | Same replicated observations and comparison data | Same D batch. | C: observed versus replicated mean, SD, minimum, maximum, tail proportions, quantile bands, and selected trajectories. |

C1–C5 calculations are in [reachability](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/reachability.py); predictive comparison outputs are in [posterior predictive checks](../../apps/data-pipeline/src/nof1_causal_lab/models/posterior_predictive.py). Reports record the design and distinguish authored, fitted, mixed or unknown laws. In-sample posterior predictive findings must not be interpreted as held-out calibration.

C3 remains a design screen based on `tau = 1 / decay`, not a full nonlinear relaxation analysis or an empirical parameter-identification test. The simulation battery has nine checks; C5d is owned by [Prepare data](#prepare-data). Checks without their prerequisites carry `passed: null` and an explicit reason. There are no hard/soft admission modes or acceptance rationales.

### Execution costs and result prerequisites

The [automatic evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/predictive_checks.py) runs one shared exact batch with 200 draws and seed 0 after relevant model edits. It reuses a matching snapshot report when model, panel and check-policy fingerprints match. C4b edge-knockout experiments remain a separate internal analysis because they require an additional batch per edge. There are no per-construct simulations, loop-closure rechecks or full-model repair barriers.

The [simulation action](../../apps/data-pipeline/src/nof1_causal_lab/actions/simulate.py) uses 100 draws and passes one generated batch to the [dynamics and measurement checks](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/simulation_checks.py). When a comparison panel is selected, explicit simulation and automatic model edits also run the [predictive comparisons](../../apps/data-pipeline/src/nof1_causal_lab/models/posterior_predictive.py) on their shared batch. Producing these paths adds D-level work even when the fit already exists. The nonlinear forward simulations use the declared emission families and retain numerical integration error.

| Result prerequisite | Required evidence | Added cost |
| --- | --- | --- |
| Causal-result certification | Identified treatment/outcome, matching model revision, committed production inference evidence, nonlinear transition target, and retained joint uncertainty. [Causal proofs](../../apps/data-pipeline/src/nof1_causal_lab/models/causal_proofs.py). | A–B: metadata, fingerprints, and log reads. This governs causal reporting; it does not generate or measure simulated paths. |

Across all four actions, each check declares its required inputs. Applicable checks return with the action's results. One affected predictive batch runs automatically inside `edit_model`; `prepare_data` runs only data checks. Post-fit predictive comparisons belong to explicit `simulate` requests with a comparison panel. Fitting and additional simulation designs remain explicit actions. Reuse depends on unchanged relevant model, data, design, and numerical inputs. An inexpensive reduction such as R-hat still needs sampling records for the model being assessed.
