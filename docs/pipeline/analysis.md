# Runtime Intervention Analysis

The pipeline ends with [inference](inference.md). Interventional (rung 2) and counterfactual (rung 3) scenarios are runtime API queries against the conditioned model, following Pearl's ladder of causation[^pearl2019] [^pearl2009]. They produce ephemeral responses rather than persisted reports.

## Inputs

| Input | Source | Description |
|---|---|---|
| `model` | [`posterior` transition](inference.md) | Conditioned ModelSpec with the joint parameter/trajectory law and time grid; exact engine evidence comes from its producing transition log |
| `identification_report` | [`measurement_structure` transition](measurement-structure.md) | Positive and negative identification findings; numerical effects require a positive treatment verdict |
| `request` | API caller | Start rule, timed clamps, outcome identity, horizon, and readout |

The backend joins the current identification evidence, conditioned model, and committed inference evidence before computing numerical effects. The simulation context is named `analysis`; the web client dispatches its tools through `POST /api/tools/dispatch`. The [tool server](../../apps/data-pipeline/src/nof1_causal_lab/tool_server.py) owns freshness checks and numerical execution.

## Runtime queries

**Interactive tools:** The fitted model exposes two read-only tools. They require no LLM generation or refit.

- *`get_model_info`:* Returns the fitted model, variables, identification evidence, diagnostics, and capabilities. An optional `names` filter selects constructs or indicators.
- *`simulate`:* Accepts one [ScenarioRequest](#scenariorequest) with persistent construct targets. A baseline start uses each posterior draw's deterministic equilibrium. An abducted start selects the retained fitted latent state at an observed index or timestamp, defaulting to the last retained observation. The backend integrates reference and clamped paths over the requested horizon with the true nonlinear drift and returns a [SimulationResult](#simulationresult).

These interactive paths include posterior parameter uncertainty and, for abducted starts, retained initial-state uncertainty. They do not sample future process noise. For nonlinear drift, their average is generally different from the stochastic process's expected trajectory. These results must not be described as a full posterior-predictive distribution.

The backend retains at most two loaded fitted contexts per process. Native posterior arrays, packed dynamics and baseline equilibrium solves are reused for the same immutable fit. Every request independently checks current identification and freshness; stale supporting inputs reject simulation even on a cache hit. The shared time grid and resolved observed start travel with the response.

Requests and responses remain in UI memory. A request can be rerun against the current fit; each response references the exact fitted model revision so the UI can detect older results. Simulation does not write an artifact, advance the episode journal, or add a pipeline transition. The model viewer offers default intervention requests for backend-identified treatments and displays results only after an API call.

## Responses

The [runtime contracts](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/scenarios.py) define the request and response fields below. These values are not registered artifact types.

### `EffectTrajectoryPoint`

| Field | Type | Description |
|---|---|---|
| `day` | `float` | Nonnegative elapsed days from the start of the rollout |
| `effect` | `float` | Signed causal effect at that time |

### `ScenarioRequest`

| Field | Type | Description |
|---|---|---|
| `start` | `ScenarioStartInput` | Baseline or observed-history start rule; observed index and timestamp are mutually exclusive |
| `clamps` | `list[ScenarioClamp]` | Timed interventions, each with one persistent `ConstructId` target and set, shift, ramp or trajectory values |
| `outcome` | `ConstructId` | Persistent outcome identity |
| `readout` | `ScenarioQueryInput` | End-state or trajectory readout, horizon in days, and projection |

### `SimulationResult`

| Field | Description |
|---|---|
| `request` | The reusable [`ScenarioRequest`](#scenariorequest) supplied to the computation |
| `model` | `ModelRevision` identifying the exact fitted model and owning workspace |
| `time_grid_days` | Shared output grid for every construct's reference and action means, starting at zero and ending at the requested horizon |
| `start_time_index`, `start_time` | Resolved abducted start index and available observed timestamp; absent for baseline starts |
| `labels` | Display names keyed by persistent construct ID at execution time; requests contain identities only |
| `summary` | Posterior summary of the declared drift-path contrast |
| `effect_trajectory` | Optional [`EffectTrajectoryPoint`](#effecttrajectorypoint) list reporting the mean outcome contrast over the forward horizon |
| `trajectory_peak` | Optional [`EffectTrajectoryPoint`](#effecttrajectorypoint) at the largest absolute departure |
| `trajectories` | One [`SimulationTrajectory`](#simulationtrajectory) per simulated construct ID, including every requested clamp target and outcome; available for both end-state and trajectory readouts |
| `manifest_effects` | Optional outcome effects projected through measurement loadings |
| `reference_mean` | Mean reference outcome across the deterministic drift paths at the final horizon |
| `warnings` | Diagnostic warnings |

### `SimulationTrajectory`

| Field | Description |
|---|---|
| `reference_mean` | Mean latent no-clamp path across simulated draws, including day zero |
| `action_mean` | Mean latent path under the requested clamps across the same simulated draws, including day zero |

Both series are required, contain finite values, and have exactly one value per entry in `time_grid_days`. Each trajectory key must resolve through the result's `labels`. The selected outcome's optional `effect_trajectory` is a separate readout and does not define the time axis for these paths.

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^pearl2019]: Pearl, J. (2019). The Seven Tools of Causal Inference, with Reflections on Machine Learning. *Communications of the ACM*, 62(3), 54–60. [Bibliography entry](../reference/bibliography.md)
