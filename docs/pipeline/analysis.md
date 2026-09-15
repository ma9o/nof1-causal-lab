# Intervention Simulation

Interventional (rung 2) and counterfactual (rung 3) scenarios use the [`simulate` action](../reference/scientific-actions.md) with a causal design against a selected conditioned model, following Pearl's ladder of causation[^pearl2019] [^pearl2009]. The action persists its design, generated arrays and certified result in the episode journal.

## Inputs

| Input | Source | Description |
|---|---|---|
| `model_version` | [`fit` result](inference.md) | Selected ModelSpec with the joint parameter/trajectory law and time grid; exact engine evidence comes from its producing transition log |
| `design.query` | API caller | [ScenarioRequest](#scenariorequest): start rule, timed clamps, outcome identity, horizon, and readout |
| `design.draws`, `design.seed` | API caller | Draw count and random seed |
| `design.process_noise`, `design.observation_noise` | API caller | Process diffusion and sampled emissions; both default to `false` for causal designs |

Submit `{action: "simulate", model_version: ..., design: {kind: "causal", query: ...}}` through the action endpoint or the matching `scientific` tool. The [causal action](../../apps/data-pipeline/src/nof1_causal_lab/actions/scenarios.py) derives identification from the selected immutable model and joins it to that model's committed production-fit evidence. A positive treatment/outcome verdict and matching exact-engine evidence are required before numeric effects are reported.

## Execution

The action requires no LLM generation or refit. A baseline start uses each draw's nonlinear drift equilibrium. An abducted start selects a retained fitted latent state at a model time index or timestamp, defaulting to the last retained state. Parameter and state draws remain paired.

The common [nonlinear generator](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py) integrates reference and clamped paths on the same grid. With process noise enabled, the pair shares Brownian streams and integration segments. Observation contrasts use the true emission model, including measurement windows. Windows requiring unavailable prehistory produce absent contrasts and an explicit warning.

The default deterministic design includes parameter uncertainty and, for abducted starts, retained initial-state uncertainty, but excludes future process and observation noise. For nonlinear drift, its average generally differs from the stochastic process's expected trajectory. The report records these choices; predictive uncertainty requires the relevant noise choices.

The `analysis` context retains the read-only `get_model_info` tool for model inspection. All simulations use the four-action surface, including the web viewer's intervention controls.

Every successful simulation writes a journal record with immutable array references and the exact supporting revisions. Historical results remain available after later edits. The model viewer offers requests for backend-identified treatments and displays results after their durable action completes.

## Outputs

The [simulation report](../reference/scientific-actions.md#simulation-report) owns the design, arrays and measurements. Its `causal_result` contains the fields defined by the [scenario contracts](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/scenarios.py) below.

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
| `summary` | Posterior summary of the declared endpoint contrast under the selected noise policy |
| `effect_trajectory` | Optional [`EffectTrajectoryPoint`](#effecttrajectorypoint) list reporting the mean outcome contrast over the forward horizon |
| `trajectory_peak` | Optional [`EffectTrajectoryPoint`](#effecttrajectorypoint) at the largest absolute departure |
| `trajectories` | One [`SimulationTrajectory`](#simulationtrajectory) per simulated construct ID, including every requested clamp target and outcome; available for both end-state and trajectory readouts |
| `manifest_effects` | Optional endpoint indicator contrasts generated by the actual emission model and measurement windows |
| `reference_mean` | Mean reference outcome across generated paths at the final horizon |
| `warnings` | Diagnostic warnings |

### `SimulationTrajectory`

| Field | Description |
|---|---|
| `reference_mean` | Mean latent no-clamp path across simulated draws, including day zero |
| `action_mean` | Mean latent path under the requested clamps across the same simulated draws, including day zero |

Both series are required, contain finite values, and have exactly one value per entry in `time_grid_days`. Each trajectory key must resolve through the result's `labels`. The selected outcome's optional `effect_trajectory` is a separate readout and does not define the time axis for these paths.

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../reference/bibliography.md)
[^pearl2019]: Pearl, J. (2019). The Seven Tools of Causal Inference, with Reflections on Machine Learning. *Communications of the ACM*, 62(3), 54–60. [Bibliography entry](../reference/bibliography.md)
