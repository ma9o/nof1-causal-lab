# Statistical Model Specification and Prior Elicitation

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | Yes | `ModelSpec` with a native prior on each parameter |

Enriches the [measurement-stage ModelSpec](measurement-structure.md) into a fully specified statistical model by choosing observation-model distributions for ambiguous indicators and eliciting Bayesian priors for every parameter, validated against prior predictive checks.

For the high-level reducer flow, see the [`statistical_model_spec` construct-admission state machine](../reference/statistical-model-spec/state-machine.md). For its exact prompts, validation, checkpoint, and recovery semantics, see the [LLM-driven specification](../reference/statistical-model-spec/llm-driven-specification.md).

## Inputs

| Input | Source | Description |
|---|---|---|
| `question` | User | Original research question, used to justify prior reasoning |
| `model` | [Measurement authoring](measurement-structure.md) | The scientific entities to enrich with likelihoods, dynamics, parameters, and priors |
| `data_for_model` | [`measurements` transition](extraction.md) | Encoded long-format [`ObservationRecord`](extraction.md#observationrecord) table |
| `indicator_audits` | [`validation_report` derivation](extraction-validation.md) | Per-indicator [`EmpiricalProfile`](extraction-validation.md#empiricalprofile)s and validation summaries |
| `enable_literature` | Pipeline config | Whether the `search_literature` tool is offered to the LLM |

`statistical_model_spec` transition is the first point where the pipeline reasons about statistical model form. Earlier transitions defined what to measure and how.

## Process

`statistical_model_spec` transition admits constructs incrementally along the causal topology. Independent ready constructs use concurrent LLM subroutines, while members of a feedback component remain sequential. Deterministic code compiles each cumulative partial model and runs the exact prior-predictive reachability battery before accepting a branch.

```mermaid
flowchart LR
    S[Concrete component proposal] --> O[SCC condensation DAG]
    O --> P[Ready-frontier fanout]
    P --> A{Compile + exact\nbranch battery}
    A -- revise --> P
    A -- admitted --> C[Immutable branch checkpoints]
    C --> M[Deterministic frontier merge]
    M --> N{More constructs?}
    N -- yes --> P
    N -- no --> B{Shared full-model barrier}
    B -- reopen failed unit + descendants --> P
    B -- pass --> F([Completed Model])
```

**Component proposal:** Before LLM judgment, deterministic authoring proposes concrete mechanisms, [likelihoods](../reference/statistical-model-spec/likelihoods.md), innovation and initial-state components, and their coefficient references. Prompt rows describe the parameters used by that proposal. The LLM may revise supported components and declare the parameters they reference while preserving the causal structure and measurement definitions.

**Admission Topology:** Strongly connected components of the model’s retained execution edges form a deterministic condensation DAG. All ready singleton units may run concurrently. Members of a lagged feedback component remain adjacent and sequential, and the edge that closes a feedback loop is authored when its final endpoint is admitted.

**Construct Submission:** The active construct submission contains:

- conditional probability expressions for its indicators;
- innovation and initial-state coefficients, and priors for its referenced parameters;
- priors for incoming or cycle-closing causal effects; and
- optional written acceptance rationales for soft reachability findings.

Each submission includes the components and their parameter definitions together. Parameter coefficient slots reference stable IDs; fixed slots carry model-scale values. The complete candidate rejects dangling references and unused parameter definitions. A cycle-closing construct must author the closing edge in the same submission so the restricted cumulative model never contains an unbound edge site.

**Validation:** Each submission compiles its immutable causal-ancestor closure plus the proposed construct and simulates it through the exact nonlinear prior-predictive engine. Hard failures require revision. Soft failures require either revision or an explicit rationale accepting the consequence. Each successful branch merges as it completes, allowing newly ready descendants to start while unrelated work remains in flight.

**Full-Model Barrier:** Once every construct is accepted, deterministic code compiles the complete model once and draws one shared exact prior-predictive sample set. Every construct is rechecked against that same model. A failure reopens the failing feedback unit from that member onward and all descendant units while retaining independent admitted branches.

When enabled, the LLM can query [Exa](https://exa.ai/) for empirical studies to inform prior calibration, justifying narrower priors only when the estimand, population, and timescale align[^gelman2020] [^gelman2013].

The reachability battery includes:

- *Numerical health and confinement*: exact nonlinear SDE trajectories must remain finite; sustained growth is surfaced separately.
- *Marginal latent scale*: across-draw late-time scale must remain compatible with the standardized-latent convention.
- *Design resolvability*: sufficient prior timescale mass must be visible through the active construct's actual irregular observation gaps and span.
- *Edge influence and Hill activation*: same-noise per-edge contrasts detect parent-dominated dynamics, while draw-paired Hill occupancy checks the actual nonnegative response region.
- *Replicated-data checks*: family-specific location and dispersion statistics compare the observed panel with complete prior-replicate datasets rather than flattened samples.
- *Transmission*: the support-aware expected response must move meaningfully relative to the sampled predictive response.

Only deterministic numerical failures are hard gates. Monte Carlo discrepancies require revision or an exact target-scoped acceptance rationale. When a submission closes a feedback component, every affected member is rechecked before the tentative state is committed.

### Checkpointing and Recovery

Checkpoints are immutable execution sidecars. Original LLM submissions and their revision history live in the state-machine records; each model parameter carries its current distribution. They store the accepted dependency-closed set, exact input-version pins, validation outcomes, search state, repair feedback, and full-model barrier status. Concurrent submissions write immutable child checkpoints from their launch snapshots; one merge activity serializes each completion batch into the next master checkpoint. The completed `model` and `admission_report` are written only after every construct is admitted and the barrier passes.

Temporal resumes an interrupted in-flight workflow from its recorded activity and child-workflow history. When a model-spec run terminates, its episode-journal record carries a typed run/checkpoint selection. The checkpoint layer resolves that selection when the outer orchestrator modifies an upstream artifact through normal machine moves and runs `statistical_model_spec` again.

On the next run:

- unchanged input pins restore the accepted dependency-closed set without rerunning it;
- changed input pins rebuild the component proposal and replay saved contributions through the same exact admission checks; and
- each invalid unit and its descendants reopen while independent valid branches remain accepted.

Each accepted tool submission is keyed by its tool-request identifier. Retrying the activity returns the same immutable checkpoint rather than applying the submission twice.

### Example

For a study of classroom engagement and academic performance, the transition could admit independent `Teacher Feedback Frequency` and `Home Study Support` roots concurrently. Once their branch checkpoints merge, `Student Engagement` authors its dynamics and incoming effects. A feedback pair involving engagement stays sequential, and the complete model must pass the shared barrier before the transition writes its public artifact.

## Outputs

| Output | Type | Description |
|---|---|---|
| `model` | [`ModelSpec`](latent-structure.md#modelspec) | Completed scientific entities, owned components, parameter definitions, and priors |
| `admission_report` | [`AdmissionReport`](#admissionreport) | Prior research and accepted C1–C5 findings, pinned to the completed model |

### LikelihoodSpec

| Field | Type | Description |
|---|---|---|
| `law` | `ObservationLaw` | Native distribution constructor whose arguments are expressions over scientific states and coefficients |
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
| `distribution` | Native NumPyro law or reference to a shared ModelSpec distribution; required for a free parameter in a complete model |
| `value` | Fixed model-scale value, mutually exclusive with a distribution |
| `distribution_transform` | Identity, interval persistence to decay, interval effect to rate, or initial correlation |
| `reference_interval_days` | Positive interval defining an authored interval-scale distribution |

Authoring rationales and supporting sources belong in the state-machine log. Fitting updates the same parameter's distribution in a new ModelSpec revision; the original law remains available in the input revision.

Prior density curves are computed on backend reads from the native NumPyro law, on the declared authoring scale. A small deterministic prior draw sets the plotting range; curve heights use the native `log_prob`. Curves are cached for display and never stored in `ParameterSpec` or used by inference. Joint, batched, discrete, and point-mass laws do not have a scalar density plot.

### Model Statistical Choices

| Field | Owner | Description |
|---|---|---|
| `dynamics` | `Construct` | Intrinsic drift contributions and node potentials |
| `likelihood` | `Indicator` | Conditional probability law, standardization, and evidence |
| `mechanisms` | `CausalEdge` | Additive causal effect functions |
| `innovation` | `Construct` | Noise distribution, scale, conditional loadings, and optional tail parameter |
| `initial_state` | `Construct` | Initial mean, scale, and correlations |
| `parameters` | `ModelSpec` | Referenced definitions with native laws or fixed values and supporting evidence |

### State Distributions

| Component | Fields | Description |
|---|---|---|
| `InnovationSpec` | `distribution`, `scale`, `loadings`, `degrees_of_freedom` | Construct-owned driving noise; loadings retain conditional noise coordinates |
| `InitialStateSpec` | `mean`, `scale`, `correlations` | Construct-owned initial location, marginal scale, and joint correlations |
| `StateCoupling` | `other_id`, `coefficient` | Explicit reference to another construct and the associated coefficient |

### DynamicsMechanism

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

### Coefficient

| Variant | Fields | Description |
|---|---|---|
| `FixedCoefficient` | `kind="fixed"`, `value` | Finite coefficient on the continuous-time model scale |
| `ParameterCoefficient` | `kind="parameter"`, `parameter_id` | Explicit reference to its [scientific parameter](#parameterspec) |

### AdmissionReport

| Field | Type | Description |
|---|---|---|
| `search_queries` | `dict[str, str]` ∣ `null` | Research queries used for prior elicitation |
| `validation_warnings` | `list[str]` ∣ `null` | Retained validation findings |
| `prior_predictive_samples` | `dict[IndicatorId, list[float]]` ∣ `null` | Exact prior-predictive observations for Data-vs-Prior inspection |
| `prior_predictive_diagnostics` | `list[PriorPredictiveDiagnostic]` | Accepted construct-level checks, including feedback-component rechecks |

[^gelman2020]: Gelman, A., Vehtari, A., Simpson, D., et al. (2020). Bayesian Workflow. arXiv:2011.01808. [Bibliography entry](../reference/bibliography.md)
[^gelman2013]: Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Bibliography entry](../reference/bibliography.md)
