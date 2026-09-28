# Statistical Model Specification and Prior Elicitation

Agents add likelihoods, dynamics, parameters and probability laws through [`edit_model`](../reference/scientific-actions.md). These choices may be interleaved with structural and measurement edits. Valid partial definitions save with explicit readiness findings; the backend runs applicable checks on the complete submitted model.

## Inputs

| Input | Description |
| --- | --- |
| `model` | The question, constructs, measurements, mechanisms, parameter definitions and current laws. |
| `expected_revision` | The current model tree OID, or `null` for the first model. |
| Prepared panel | When compatible, supplies the observation schedule and values for automatic predictive checks. |

## Process

The external agent selects supported [likelihoods](../reference/statistical-model-spec/likelihoods.md), [parameters and priors](../reference/statistical-model-spec/parameters.md), and coefficients satisfying the [anchor invariant](../reference/statistical-model-spec/identification.md). Literature can inform these decisions when population, estimand and timescale match[^gelman2020] [^gelman2013]. There is no internal construct proposal or repair workflow.

The [action evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py) checks specification, identification, data profile and model/data compatibility in a fixed sequence, reusing groups whose scientific inputs and policy versions are unchanged. The [automatic predictive evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/predictive_checks.py) then samples the current laws and runs one whole-model nonlinear batch on the compatible panel's grid. The [check catalog](../reference/model-checks.md) defines the individual measurements and costs.

Scientific failures are saved findings. Missing prerequisites leave checks unevaluated. Unexpected execution errors leave the prior model selected. The candidate, reports and action messages publish atomically; the external agent reads the results and decides its next edit. [Incremental check execution](../reference/statistical-model-spec/state-machine.md) describes selection and reuse, while [agent authoring](../reference/statistical-model-spec/llm-driven-specification.md) describes the interaction contract.

## Outputs

| Field | Description |
| --- | --- |
| `model` | Revised [ModelSpec](latent-structure.md#modelspec), including statistical definitions and current probability laws. |
| `specification` | Execution and fitting-law readiness findings. |
| `identification` | Causal-identification report for the selected graph and measurements. |
| `validation` | Available [model/data compatibility findings](extraction-validation.md). |
| `predictive` | Automatic [ModelPredictiveReport](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/model_checks.py): status, prerequisites, source revisions, design, law provenance and shared simulation findings. |

### LikelihoodSpec

| Field | Type | Description |
|---|---|---|
| `law` | `ObservationLawSpec` | Native distribution constructor whose arguments are expressions over scientific states and coefficients |
| `standardized` | `bool` | Declared standardization choice for eligible additive-location indicators |
| `reasoning` | `str` | Scientific justification |
| `sources` | `tuple[LiteratureSource, ...]` | Evidence for the choice |

The [conditional law grammar](../reference/statistical-model-spec/likelihoods.md#conditional-expressions) composes native constructor arguments from the same expressions used by dynamics. For example, `Normal(loc=a + b * state(x), scale=s)` and `Bernoulli(logits=a + b * state(x))` declare the complete observation formula. State and parameter references use persistent scientific identities. Coefficient operands carry their scientific roles and may remain explicitly unassigned in a partial model; authoring fills them before execution.

Parameter ownership, support checks, numerical lowering, and displayed observation equations derive from this declaration. The compiler recognizes the supported affine predictors and response functions, preserving the existing exact emission kernels and the indicator's interval and category semantics. Unsupported formulas fail validation.

### ParameterSpec

| Field | Description |
|---|---|
| `id` | Stable identity referenced by component coefficients; preserved through model revisions |
| `name` | Display label; references use the identity |
| `description` | Scientific interpretation |
| `distribution` | ID of a law in `ModelSpec.distributions`; required for a free parameter in a complete model |
| `value` | Fixed model-scale value, mutually exclusive with a distribution |
| `distribution_transform` | Identity, interval persistence to decay, interval effect to rate, or initial correlation |
| `reference_interval_days` | Positive interval defining an authored interval-scale distribution |

Scientific rationales and supporting sources belong with their model definitions and the external agent conversation. Fitting updates the same parameter's law membership in a new ModelSpec revision; the original law remains available in the input revision.

Prior density curves are computed on backend reads from the native NumPyro law, on the declared authoring scale. A small deterministic prior draw sets the plotting range; curve heights use the native `log_prob`. Curves are cached for display and never stored in `ParameterSpec` or used by inference. Joint, batched, discrete, and point-mass laws do not have a scalar density plot.

### Model Statistical Choices

| Field | Owner | Description |
|---|---|---|
| `dynamics` | `ConstructSpec` | Intrinsic drift contributions and node potentials |
| `likelihood` | `IndicatorSpec` | Conditional probability law, standardization, and evidence |
| `mechanisms` | `CausalEdgeSpec` | Additive causal effect functions |
| `coefficients` | `ConstructSpec` | Shared coefficient expressions for diffusion scale and loadings, initial mean and scale, correlations, and the optional process tail parameter |
| `innovation_family` | `ConstructSpec` | Gaussian or Student-t driving noise |
| `parameters` | `ModelSpec` | Referenced definitions with law memberships or fixed values |
| `distributions` | `ModelSpec` | Native NumPyro laws keyed by the IDs referenced by parameters and constructs |

### Construct Coefficients

The same [coefficient expressions](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/expressions.py) used in dynamics and likelihoods declare construct-level scalar uses. Existing numerical assembly maps these uses to diffusion and initial-distribution coordinates.

| Field | Description |
|---|---|
| `role` | `diffusion_scale`, `diffusion_loading`, `process_degrees_of_freedom`, `initial_mean`, `initial_scale`, or `initial_correlation` |
| `value` | Finite number or persistent parameter ID; `null` marks an incomplete declaration |
| `construct_ids` | One additional construct for a joint loading or correlation; empty for a scalar use |

| Role | Numerical use | Description |
|---|---|---|
| `diffusion_scale`, `diffusion_loading` | Diffusion factor | Marginal driving scale and conditional noise coordinates |
| `initial_mean`, `initial_scale`, `initial_correlation` | Initial distribution | Initial location, marginal scale, and joint correlations; unmeasured static roots share their baseline scale |
| `process_degrees_of_freedom` | Process tail parameter | Shared parameter for Student-t innovations |

### DynamicsMechanismSpec

| Field | Description |
|---|---|
| `id` | Persistent identity of one dynamics contribution, preserved through revisions and reordering |
| `kind` | `"drift"` adds the expression to the owning state's derivative; `"potential"` declares a node energy whose negative gradient contributes to the drift |
| `expression` | A scalar [expression](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/expressions.py) with explicit construct and coefficient references |

| Expression form | Meaning |
|---|---|
| `literal` | A finite numerical constant |
| `state` | A construct's state or declared known input, referenced by its ID |
| `coefficient` | A fixed value or parameter reference, with its scientific role and support |
| `binary` | Addition, subtraction, multiplication, division, power, or maximum of two expressions |

Scientific constructors such as `restoring_potential`, `linear_effect`, and `hill` expand into this language. Multiplying a Hill expression by another state expresses moderation without a separate edge-specification class. The [numerical compiler](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/expression.py) binds parameter references directly. [Dynestyx state evolution](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/vector_field.py) differentiates declared node potentials and adds the remaining drift expressions; [displayed equations](../../apps/data-pipeline/src/nof1_causal_lab/machine/equations.py) interpret the same tree.

Intrinsic dynamics reference only their owning construct. Potentials belong to nodes; directed edges require drift terms. An edge expression references its primary cause, and any additional state dependencies require explicit causal edges into its effect. Fixed coefficient values must satisfy their declared support. Location anchoring requires a complete restoring expression of the declared kind; a coefficient's role alone does not certify an anchor. Known-input and marginalized-confounder matrix boundaries continue to require a scalar linear expression.

### CoefficientExpression

| Field | Description |
|---|---|
| `kind` | `"coefficient"`, identifying an operand in the expression grammar |
| `role` | Scientific meaning and required support of this use |
| `value` | Finite model-scale number, persistent [parameter ID](#parameterspec), or `null` while unassigned |
| `construct_ids` | Additional constructs participating in a joint coefficient use |

For example, `{"kind": "coefficient", "role": "loading", "value": 1}` declares a fixed unit loading. A parameter ID in `value` references the same scientific quantity wherever it appears; its role describes the local use. Zero is an assigned value. No nested fixed-value or parameter-reference record is stored.

### PriorPredictiveResult

Earlier study histories retain this report for historical inspection. Current edits produce the `predictive` report described above.

| Field | Type | Description |
|---|---|---|
| `samples` | `dict[IndicatorId, list[float]]` | Exact prior-predictive observations for Data-vs-Prior inspection |
| `diagnostics` | `list[PriorPredictiveDiagnostic]` | Recorded construct-level checks from the original operation |

The [model snapshot](../design/model-snapshot.md) exposes retained historical results as `findings.prior_predictive`, sourced to the original journal record. New actions write their typed check reports in `checks.json` and the action result; they create no admission checkpoints.

[^gelman2020]: Gelman, A., Vehtari, A., Simpson, D., et al. (2020). Bayesian Workflow. arXiv:2011.01808. [Bibliography entry](../reference/bibliography.md)
[^gelman2013]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Bibliography entry](../reference/bibliography.md)
