# Intervention Analysis

| Modality | Interactive | Produces |
|---|---|---|
| Hybrid | Yes | [`TreatmentEffect`](#treatmenteffect) list, interactive simulation tools |

Applies steady-state interventional-effect and trajectory-simulation semantics to the [`posterior` transition fitted model](inference.md#fittedartifact), ranks treatments by causal effect size, generates LLM commentary, and exposes three interactive tools for follow-up interventional (rung 2) and counterfactual (rung 3) queries in Pearl's ladder of causation[^pearl2019] [^pearl2009]. This is the terminal transition; interactive edits persist in place with no downstream replay.

## Inputs

| Input | Source | Description |
|---|---|---|
| `fitted_artifact` | [`posterior` transition](inference.md) | [`FittedArtifact`](inference.md#fittedartifact) with a production particle posterior, non-optional model spec, observation times, and causal-design provenance |
| `causal_design` | [`measurement_structure` transition](measurement-structure.md) | [`CausalDesign`](measurement-structure.md#causaldesign) with identifiability status, measurement structure, and outcome construct designation |
| `identification_report` | [`measurement_structure` transition](measurement-structure.md) | Positive identification artifact naming the estimable treatments and outcome |
| `question` | User | Original research question for grounding the opening commentary |

`posterior` transition provided the posterior and diagnostics; `measurement_structure` transition provided the identifiability verdicts. `baseline_report` transition is the first point where posterior samples are translated into causal decision quantities.

Before that translation, the transition constructs one `IdentifiedEstimand` per
treatment and validates that each proof, the loaded `CausalDesign`, and the
`ParticleMCMCPosterior` reference the same workspace-local causal-design
version. Numeric intervention code accepts only the joined
`CertifiedCausalAnalysis`; Laplace/IEKS warmup output and cross-design evidence
cannot reach it.

## Process

`baseline_report` transition runs in two phases: a deterministic intervention computation that produces the baseline ranking, followed by a single LLM generation that produces opening commentary. After completion the transition exposes three interactive tools for follow-up exploration.

```mermaid
flowchart LR
    B[Baseline\nranking] --> C[LLM commentary] --> T([Interactive tools])
```

**Baseline ranking:** For each treatment that remains after the [`measurement_structure` transition identifiability screen](measurement-structure.md), the transition computes a steady-state interventional effect under `do(treatment = baseline + 1)`. For each posterior draw the baseline steady state η\* solves the vector-field root `f(η*) = 0` (numerically, via Levenberg-Marquardt); for the [affine special case](../reference/estimation.md#1-ct-sde-formulation) this reduces to η\* = −**A**⁻¹**c** for [drift matrix **A** and continuous intercept **c**](../reference/estimation.md#1-ct-sde-formulation). An intervention clamps the treatment equation and re-solves the modified system, comparing the intervened and baseline outcome values. The default `do(treatment = baseline + 1)` is vmapped over all posterior draws to produce the full posterior treatment-effect distribution.

- *Temporal forward simulation:* When temporal information is available (either from the [`model_clock`](measurement-structure.md#observation_window-and-model_clock) or the median observed timestep), the transition also runs a 30-day forward simulation for each treatment, discretizing the continuous-time system from the baseline steady state with the treatment clamped at each step. The mean trajectory across posterior draws is summarized into a [`TemporalEffect`](#temporaleffect) at requested elapsed-day horizons (1, 7, and 30 days by default), plus the signed peak effect and time-to-peak. Requests outside the simulated interval are rejected.
- *Manifest-level decomposition:* When posterior draws of the [loading matrix](../reference/estimation.md#1-ct-sde-formulation) λ are available, the transition projects each treatment's outcome-level effect through the loadings to produce per-manifest effects: `manifest_effect[i] = λ[i, outcome_idx] × effect_mean`.
- *Ranking:* Treatments are sorted by |mean(`posterior_draws`)| descending.

**LLM commentary:** A single LLM generation receives the top-5 ranked effects, any diagnostic warnings, excluded non-identifiable treatments, and a summary of the follow-up capabilities. The LLM produces plain Markdown commentary for the user, persisted as `final_summary`.

**Interactive tools:** After the baseline ranking completes, `baseline_report` transition exposes three read-only tools for follow-up exploration within the same conversation.

- *`get_model_info`:* Returns a structured read-only summary of the fitted model and its diagnostics. An optional `names` filter restricts the response to specific constructs or indicators.
- *`simulate_intervention`:* Runs an interventional query on rung 2 of Pearl's ladder[^pearl2019], generalizing the baseline ranking to arbitrary intervention values, trajectory horizons, and manifest projections. Returns a posterior summary (mean, median, 95% CI, `prob_positive`) with any relevant [posterior assessment](inference.md#posteriorassessment) warnings.
- *`simulate_counterfactual`:* Runs a counterfactual query on rung 3 of Pearl's ladder[^pearl2019], conditioning on observed data before asking "what would have happened if we had intervened?" The caller specifies the evidence boundary (`start` — either an observed `time_index` or an ISO-8601 `time`, not both, defaulting to the final retained fitted latent state), the treatment and intervention mode, and the estimand (`"end_state"` or `"trajectory"` with `horizon_days` and projection level). Computation follows the standard abduction-action-prediction procedure in Pearl, Glymour, and Jewell (2016)[^pearl2016].
  - *Abduction:* recovers the latent state at the evidence boundary
  - *Forward simulation:* from the abducted state, simulates both a baseline path and a counterfactual path (treatment clamped)
  - *Output:* reports the difference as the causal effect with posterior summary, effect trajectory, and abduction warnings

### Example

For a study of agricultural practices and crop yield where `latent_structure` transition posited constructs `Irrigation Frequency`, `Soil Nitrogen`, `Pest Pressure`, and `Crop Yield`, `baseline_report` transition might rank `Soil Nitrogen` first with mean(posterior_draws)=+0.38 and a temporal peak at 12 days (`peak_effect=+0.41`), while `Pest Pressure` ranks second with mean(posterior_draws)=−0.22. Any [posterior assessment](inference.md#posteriorassessment) warnings from `posterior` transition are referenced in the LLM commentary.

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | list\[[`TreatmentEffect`](#treatmenteffect)\] | Treatments ranked by absolute `summary.mean` descending |
| `saved_scenarios` | `list[SavedScenario]` \| null | Scientific queries with zero or more posterior-specific evaluations |
| `final_summary` | `str` \| null | LLM-generated opening commentary |

### `TreatmentEffect`

| Field | Type | Description |
|---|---|---|
| `treatment` | `str` | Display name of the treatment |
| `treatment_id` | `ConstructId` | Persistent treatment identity |
| `summary` | `EffectSummary` \| null | Backend mean, median, 95% interval, and probability of a positive effect; null when unavailable |
| `histogram` | `list[HistogramBin]` | Backend bin boundaries, centers, and counts |
| `posterior_draws` | `list[float]` \| null | Full posterior distribution of the treatment effect; display summaries are computed by the backend |
| `temporal` | [`TemporalEffect`](#temporaleffect) \| null | Forward-simulation summary at explicitly requested horizons |
| `manifest_effects` | `dict[str, float]` \| null | Per-manifest outcome decomposition via [loading-matrix](../reference/estimation.md#1-ct-sde-formulation) projection; keys are manifest names, values are `λ[manifest, outcome] × effect_mean` |

Identifiability status and [posterior assessments](inference.md#posteriorassessment) are not duplicated here — consumers derive them from [`measurement_structure` transition](measurement-structure.md#identifiabilitystatus) and [`posterior` transition](inference.md) outputs respectively.

### `TemporalEffect`

| Field | Type | Description |
|---|---|---|
| `horizons` | list\[[`EffectTrajectoryPoint`](#effecttrajectorypoint)\] | Nonempty, strictly increasing requested horizons within the simulated interval |
| `peak_effect` | `float` | Signed effect at the largest absolute departure during the trajectory |
| `time_to_peak_days` | `float` | Days from intervention onset to peak effect |

### `EffectTrajectoryPoint`

| Field | Type | Description |
|---|---|---|
| `day` | `float` | Nonnegative elapsed days from the start of the rollout |
| `effect` | `float` | Signed causal effect at that time |

### `ScenarioQuery`

| Field | Type | Description |
|---|---|---|
| `id` | `ScenarioQueryId` | Content identity of the scientific question, independent of model, posterior version, and display labels |
| `start` | `ScenarioStartInput` | Baseline or observed-history start rule; execution resolves the index and timestamp |
| `clamps` | `list[ScenarioClamp]` | Timed interventions with persistent construct targets |
| `outcome` | `ConstructRef` | Persistent outcome identity |
| `readout` | `ScenarioQueryInput` | Estimand, horizon, and projection |

### `ScenarioEvaluation`

| Field | Type | Description |
|---|---|---|
| `id` | `ScenarioEvaluationId` | Content identity of the query and its exact execution basis |
| `query_id` | `ScenarioQueryId` | Identity of the scientific [question](#scenarioquery), shared across refits |
| `model` | `ModelRef` | Workspace owning the fitted model |
| `posterior` | `ArtifactRef` | Exact [posterior version](inference.md#outputs) used for this evaluation |

### `ScenarioResult`

| Field | Type | Description |
|---|---|---|
| `evaluation_id` | `ScenarioEvaluationId` | Identity of the [evaluation](#scenarioevaluation) that produced these outputs |
| `start` | `ScenarioStartResult` | Resolved initial-state source, evidence index, and timestamp |
| `outcome_label` | `str` | Outcome display name at execution time; identity belongs to the query |
| `summary` | `EffectSummary` | Posterior mean, median, 95% interval, and probability of a positive effect |
| `effect_trajectory` | list\[[`EffectTrajectoryPoint`](#effecttrajectorypoint)\] \| null | Effect over the requested forward horizon |
| `trajectory_peak` | [`EffectTrajectoryPoint`](#effecttrajectorypoint) \| null | Signed effect at the largest absolute departure |
| `visualization` | [`BaselineReportVisualization`](#baselinereportvisualization) \| null | Reference and action trajectories on persistent construct axes |
| `manifest_effects` | `dict[str, float]` \| null | Outcome effects projected onto manifest variables |
| `reference_mean` | `float` | Mean reference outcome under the baseline or factual forecast |
| `warnings` | `list[str]` | Execution and diagnostic warnings |

### `SimulateScenarioResult`

| Field | Type | Description |
|---|---|---|
| `query` | [`ScenarioQuery`](#scenarioquery) | Scientific inputs and persistent targets |
| `evaluation` | [`ScenarioEvaluation`](#scenarioevaluation) | Execution basis whose `query_id` must match `query.id` |
| `result` | [`ScenarioResult`](#scenarioresult) | Computed outputs whose `evaluation_id` must match `evaluation.id` |

### `SavedScenario`

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Human-readable scenario name |
| `query` | [`ScenarioQuery`](#scenarioquery) | Preserved scientific question, reusable across fits |
| `evaluations` | list\[[`ScenarioEvaluationResult`](#scenarioevaluationresult)\] | Unique evaluations of this question; an empty list preserves an unevaluated question |
| `summary` | `str` \| null | Optional narrative |

### `ScenarioEvaluationResult`

| Field | Type | Description |
|---|---|---|
| `evaluation` | [`ScenarioEvaluation`](#scenarioevaluation) | Model and exact posterior used to answer a saved question |
| `result` | [`ScenarioResult`](#scenarioresult) | Computed outputs belonging to this evaluation |

### `BaselineReportVisualization`

| Field | Type | Description |
|---|---|---|
| `reference_node_trajectories` | `dict[ConstructId, list[float]]` \| null | Reference trajectories keyed by persistent construct identity |
| `action_node_trajectories` | `dict[ConstructId, list[float]]` \| null | Intervention trajectories on the same construct axes |
| `node_effect_trajectories` | `dict[ConstructId, list[float]]` \| null | Effect trajectories on the same construct axes |
| `start_state` | `dict[ConstructId, float]` \| null | Starting state on the same construct axes |

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^pearl2019]: Pearl, J. (2019). The Seven Tools of Causal Inference, with Reflections on Machine Learning. *Communications of the ACM*, 62(3), 54–60. [Bibliography entry](../reference/bibliography.md)
[^pearl2016]: Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley. [Bibliography entry](../reference/bibliography.md)
