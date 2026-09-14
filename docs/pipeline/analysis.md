# Intervention Analysis

| Modality | Interactive | Produces |
|---|---|---|
| Hybrid | Yes | [`TreatmentEffect`](#treatmenteffect) list, interactive simulation tools |

Applies steady-state interventional-effect and trajectory-simulation semantics to the [`posterior` transition fitted model](inference.md#fittedartifact), ranks treatments by causal effect size, generates LLM commentary, and exposes two interactive tools for follow-up interventional (rung 2) and counterfactual (rung 3) queries in Pearl's ladder of causation[^pearl2019] [^pearl2009]. This is the terminal transition; interactive simulations are live backend requests, with results explicitly retained in reports when needed.

## Inputs

| Input | Source | Description |
|---|---|---|
| `model` | [`posterior` transition](inference.md) | Conditioned ModelSpec with the joint parameter/trajectory law and time grid; exact engine evidence comes from its producing transition log |
| `model` | [`measurement_structure` transition](measurement-structure.md) | [`ModelSpec`](latent-structure.md#modelspec) with owned measurements and the default outcome; findings are separately sourced |
| `identification_report` | [`measurement_structure` transition](measurement-structure.md) | Positive and negative identification findings; numerical effects require a positive treatment verdict |
| `question` | User | Original research question for grounding the opening commentary |

`posterior` transition provided the posterior and diagnostics; `measurement_structure` transition provided the identifiability verdicts. `baseline_report` transition is the first point where posterior samples are translated into causal decision quantities.

Before that translation, the transition constructs one `IdentifiedEstimand` per
treatment and validates that each proof, the loaded `ModelSpec`, and the
`ParticleMCMCPosterior` reference the same workspace-local Model
version. Numeric intervention code accepts only the joined
`CertifiedCausalAnalysis`; Laplace/IEKS warmup output and cross-design evidence
cannot reach it.

## Process

`baseline_report` transition runs in two phases: a deterministic intervention computation that produces the baseline ranking, followed by a single LLM generation that produces opening commentary. After completion the transition exposes two interactive tools for follow-up exploration.

```mermaid
flowchart LR
    B[Baseline\nranking] --> C[LLM commentary] --> T([Interactive tools])
```

**Baseline ranking:** For each treatment that remains after the [`measurement_structure` transition identifiability screen](measurement-structure.md), the transition computes a steady-state interventional effect under `do(treatment = baseline + 1)`. For each posterior draw the baseline steady state η\* solves the vector-field root `f(η*) = 0` (numerically, via Levenberg-Marquardt); for the [affine special case](../reference/estimation.md#1-ct-sde-formulation) this reduces to η\* = −**A**⁻¹**c** for [drift matrix **A** and continuous intercept **c**](../reference/estimation.md#1-ct-sde-formulation). An intervention clamps the treatment equation and re-solves the modified system, comparing the intervened and baseline outcome values. The default `do(treatment = baseline + 1)` is vmapped over all posterior draws to produce the full posterior treatment-effect distribution.

- *Temporal forward simulation:* When temporal information is available (either from the [`measurement_clock`](measurement-structure.md#observation_window-and-measurement_clock) or the median observed timestep), the transition also runs a 30-day forward simulation for each treatment, discretizing the continuous-time system from the baseline steady state with the treatment clamped at each step. The mean trajectory across posterior draws is summarized into a [`TemporalEffect`](#temporaleffect) at requested elapsed-day horizons (1, 7, and 30 days by default), plus the signed peak effect and time-to-peak. Requests outside the simulated interval are rejected.
- *Manifest-level decomposition:* When posterior draws of the [loading matrix](../reference/estimation.md#1-ct-sde-formulation) λ are available, the transition projects each treatment's outcome-level effect through the loadings to produce per-manifest effects: `manifest_effect[i] = λ[i, outcome_idx] × effect_mean`.
- *Ranking:* Treatments are sorted by |mean(`posterior_draws`)| descending.

**LLM commentary:** A single LLM generation receives the top-5 ranked effects, any diagnostic warnings, excluded non-identifiable treatments, and a summary of the follow-up capabilities. The LLM produces plain Markdown commentary for the user, persisted as `final_summary`.

**Interactive tools:** The fitted model exposes two read-only tools. They require no LLM generation or refit.

- *`get_model_info`:* Returns the fitted model, variables, identification evidence, diagnostics, and available baseline effects. An optional `names` filter selects constructs or indicators.
- *`simulate`:* Accepts one [ScenarioRequest](#scenariorequest) with persistent construct targets. A baseline start uses each posterior draw's deterministic equilibrium. An abducted start selects the retained fitted latent state at an observed index or timestamp, defaulting to the last retained observation. The backend integrates reference and clamped paths over the requested horizon with the true nonlinear drift and returns a [SimulationResult](#simulationresult).

These interactive paths include posterior parameter uncertainty and, for abducted starts, retained initial-state uncertainty. They do not sample future process noise. For nonlinear drift, their average is generally different from the stochastic process's expected trajectory. The response identifies this calculation as `nonlinear_drift_v1`; it must not be described as a full posterior-predictive distribution.

The backend retains at most two loaded fitted contexts per process. Native posterior arrays, packed dynamics and baseline equilibrium solves are reused for the same immutable fit. Every request independently checks current identification and freshness; stale supporting inputs reject simulation even on a cache hit. The requested time grid and actual solver settings travel with the response.

Exploratory requests and responses remain in UI memory. Results included in a report remain in its `simulation_results`, with their original request and execution provenance. The request can be rerun against the current fit. The report's artifact pins remain distinct from each included response's fitted basis. Retained generic conversation traces may also contain tool responses.

### Example

For a study of agricultural practices and crop yield where `latent_structure` transition posited constructs `Irrigation Frequency`, `Soil Nitrogen`, `Pest Pressure`, and `Crop Yield`, `baseline_report` transition might rank `Soil Nitrogen` first with mean(posterior_draws)=+0.38 and a temporal peak at 12 days (`peak_effect=+0.41`), while `Pest Pressure` ranks second with mean(posterior_draws)=−0.22. Any [posterior assessment](inference.md#posteriorassessment) warnings from `posterior` transition are referenced in the LLM commentary.

## Outputs

| Output | Type | Description |
|---|---|---|
| `intervention_results` | list\[[`TreatmentEffect`](#treatmenteffect)\] | Treatments ranked by absolute `summary.mean` descending |
| `simulation_results` | list\[[`SimulationResult`](#simulationresult)\] | Responses explicitly retained in this report, each with its original fitted basis |
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

### `ScenarioRequest`

| Field | Type | Description |
|---|---|---|
| `start` | `ScenarioStartInput` | Baseline or observed-history start rule; observed index and timestamp are mutually exclusive |
| `clamps` | `list[ScenarioClamp]` | Timed interventions, each with one persistent `ConstructRef` target and set, shift, ramp or trajectory values |
| `outcome` | `ConstructRef` | Persistent outcome identity |
| `readout` | `ScenarioQueryInput` | End-state or trajectory readout, horizon in days, and projection |

### `SimulationResult`

| Field | Type | Description |
|---|---|---|
| `request` | [`ScenarioRequest`](#scenariorequest) | The reusable request supplied to the computation |
| `provenance` | [`SimulationProvenance`](#simulationprovenance) | Fitted model and numerical settings used for this response |
| `labels` | `dict[ConstructId, str]` | Display names at execution time; requests contain identities only |
| `summary` | `EffectSummary` | Posterior summary of the declared drift-path contrast |
| `effect_trajectory` | list\[[`EffectTrajectoryPoint`](#effecttrajectorypoint)\] \| null | Mean contrast over the forward horizon |
| `trajectory_peak` | [`EffectTrajectoryPoint`](#effecttrajectorypoint) \| null | Signed effect at the largest absolute departure |
| `visualization` | [`BaselineReportVisualization`](#baselinereportvisualization) \| null | Reference and action trajectories on persistent construct axes |
| `manifest_effects` | `dict[str, float]` \| null | Outcome effects projected through measurement loadings |
| `reference_mean` | `float` | Mean reference outcome across the deterministic drift paths |
| `warnings` | `list[str]` | Diagnostic warnings |

### `SimulationProvenance`

| Field | Type | Description |
|---|---|---|
| `model` | `ModelRevision` | Exact scientific model and owning workspace |
| `posterior` | `ArtifactRef` | Exact [posterior version](inference.md#outputs) supplying joint draws |
| `engine` | `str` | `nonlinear_drift_v1` for live deterministic drift calculations; `illustrative_fixture` identifies retained demo values |
| `solver` | `str` | `Tsit5` integration of the true nonlinear drift |
| `rtol`, `atol`, `max_steps` | Numeric settings | Actual solver tolerances and step limit |
| `draw_count` | `int` | Number of retained posterior draws used |
| `time_grid_days` | `list[float]` | Actual output grid, starting at zero and ending at the requested horizon |
| `start_time_index`, `start_time` | Observed index and timestamp \| null | Resolved abducted start; absent for baseline starts |

### `BaselineReportVisualization`

| Field | Type | Description |
|---|---|---|
| `reference_node_trajectories` | `dict[ConstructId, list[float]]` \| null | Reference trajectories keyed by persistent construct identity |
| `action_node_trajectories` | `dict[ConstructId, list[float]]` \| null | Intervention trajectories on the same construct axes |
| `node_effect_trajectories` | `dict[ConstructId, list[float]]` \| null | Effect trajectories on the same construct axes |
| `start_state` | `dict[ConstructId, float]` \| null | Starting state on the same construct axes |

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^pearl2019]: Pearl, J. (2019). The Seven Tools of Causal Inference, with Reflections on Machine Learning. *Communications of the ACM*, 62(3), 54–60. [Bibliography entry](../reference/bibliography.md)
