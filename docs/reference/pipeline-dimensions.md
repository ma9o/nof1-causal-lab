# Pipeline Dimensions

This page maps concerns shared by the four [scientific actions](scientific-actions.md) and optional authoring recipes. Artifact definitions remain in their owning docs.

## Artifact Lineage

Actions refine one scientific model and produce separately versioned data and evidence. The layers below can be revisited in any order allowed by their required inputs.

| Layer | Primary artifact | Produced in | Owner doc | Purpose |
|---|---|---|---|---|
| Research intent | Natural-language question | `ModelSpec.question` | [ModelSpec](../pipeline/latent-structure.md#modelspec) | States the causal research aim for that model revision |
| Theoretical causal structure | `ModelSpec` | `edit_model` or latent-authoring recipe | [../pipeline/latent-structure.md](../pipeline/latent-structure.md) | Defines constructs, edges, and the designated outcome |
| Measurement and identification | `ModelSpec` and identification findings | `edit_model` or measurement-authoring recipe | [../pipeline/measurement-structure.md](../pipeline/measurement-structure.md) | Adds owned indicators and derives separately sourced identification findings |
| Observational evidence | `ObservationRecord`s and the encoded observation table (`data_for_model`) | `prepare_data` | [../pipeline/extraction.md](../pipeline/extraction.md) | Converts source data into time-indexed indicator values |
| Data quality and compatibility | Data profile and validation report | Submission derivations | [../pipeline/extraction-validation.md](../pipeline/extraction-validation.md) | Separates empirical data facts from model-dependent findings |
| Functional specification | `ModelSpec` with current laws | `edit_model` or statistical-authoring recipe | [../pipeline/statistical-model-spec.md](../pipeline/statistical-model-spec.md) | Chooses likelihoods, parameters, and beliefs |
| Conditioned scientific model | `ModelSpec` with updated joint distributions | `fit` | [../pipeline/inference.md](../pipeline/inference.md) | The same scientific type; inference diagnostics live in the transition log |
| Generated evidence | Simulation report and arrays | `simulate` | [scientific-actions.md](scientific-actions.md#simulation-report) | Measures trajectories and optionally compares them with data |
| Interventional and counterfactual effect summaries | `SimulationResult` within the simulation report | `simulate` with a causal design | [../pipeline/analysis.md](../pipeline/analysis.md) | Answers certified interventional and counterfactual queries |

## Temporal Semantics

Time appears in five distinct roles across the pipeline. They answer different questions and should not be collapsed into a single notion of granularity.

| Concept | Primary owner | What it answers |
|---|---|---|
| **`model_clock`** | [`measurement_structure` transition](../pipeline/measurement-structure.md#observation_window-and-model_clock) | What is the shared tick width used for extraction, discretization, and the default lag unit? A global setting (e.g. `"1d"`) that aligns all indicators onto a common grid. |
| **`observation_window`** | [`measurement_structure` transition](../pipeline/measurement-structure.md#observation_window-and-model_clock) | Over what support interval is a single indicator value measured or aggregated? May differ per indicator (e.g. daily mood vs. weekly incident count) as long as windows align back onto the `model_clock`. |
| **`anchor_time`** | [`measurements` transition](../pipeline/extraction.md#observationrecord) | Which timestamp attaches the extracted value to the latent grid? Derived from the indicator's [`anchor_policy`](../pipeline/measurement-structure.md#derived-observation-semantics) — usually `support_end` for interval summaries, `support_start` for `first`. |
| **`dt`** | [estimation.md](estimation.md#2-discretization-ct-to-dt) | What is the elapsed time between consecutive prepared grid points, including support boundaries? It scales the nonlinear Euler–Maruyama transition and process noise in particle inference. |
| **Intervention horizon** | [Causal simulation](../pipeline/analysis.md) | How far forward is an intervention projected? The design supplies the horizon; model-clock steps and intervention boundaries define the output grid. |

### Worked example: one observation through the pipeline

Consider a study with `model_clock = "1d"` and an indicator *daily mean mood* (`aggregation = mean`, `observation_window = "1d"`).

| Stage | What happens | Temporal artifact |
|---|---|---|
| **Edit model** | The measurement structure declares `aggregation = mean` → `support_kind = interval`, `anchor_policy = support_end`. | `observation_window = "1d"` committed |
| **Prepare data** | The extractor averages mood values from 2025-03-01 00:00 to 2025-03-02 00:00, producing value 6.2. | `ObservationRecord(anchor_time = 2025-03-02, support_start = 2025-03-01, support_end = 2025-03-02)` |
| **Fit** | Consecutive prepared grid points are one day apart; the estimator evaluates the true nonlinear drift with `dt = 1.0 day`. | `dt = 1.0` day feeds the particle transition |
| **Simulate** | A certified intervention `do(exercise = baseline+1)` is simulated forward 30 days at 1-day steps from the baseline equilibrium, retaining [`SimulationResult`](../pipeline/analysis.md#simulationresult) with an effect trajectory and peak timing. | Horizon = 30 d at `model_clock` resolution |

The key invariant: `model_clock` sets the resolution; `observation_window` says how much real-world time each datum summarizes; `anchor_time` places it on the grid; `dt` discretizes the SDE between grid points; the intervention horizon projects the fitted model forward on that same grid.

## Assurance Surface

The pipeline has several kinds of checks. They target different failure modes and should not be conflated.

| Assurance target | Action or dependency | Question being answered |
|---|---|---|
| Causal identification and model capability | Model submission | Is the requested effect identified, and which operations can the model support? |
| Data quality | Observation panel | Are the observed series numerically usable and temporally coherent? |
| Compatibility | Model and panel | Do the observations satisfy this model's measurement and fitting requirements? |
| Fit diagnostics | `fit` | Are sampler behavior and fit-based diagnostic measurements satisfactory? |
| Generative behavior and predictive checks | `simulate` | Are generated quantities scientifically plausible and compatible with selected comparison observations? |

## Assumption Map

| Assumption | Primary owner | Main consumers | Detail page |
|---|---|---|---|
| A1. Reflective measurement structure | Model measurement definitions | Editing, compatibility, fit and simulation | [measurement-structure/assumptions.md](measurement-structure/assumptions.md) |
| A3. Markov property for temporal dynamics | Model dynamics | Editing, fit and simulation | [latent-structure/assumptions.md](latent-structure/assumptions.md) |
| A3a. Latent confounders have bounded temporal reach | Model causal structure | Identification | [causal-design/identifiability.md](causal-design/identifiability.md) |
| A4. Acyclicity within time slice | Model causal structure | Editing and identification | [latent-structure/assumptions.md](latent-structure/assumptions.md) |
| A4b. Endogenous time-varying directed effects are drift-mediated | Model dynamics | Editing, fit and simulation | [latent-structure/assumptions.md](latent-structure/assumptions.md) |
| A5. Time-invariant latents as subject-level static states | Model constructs | Editing, fit and simulation | [latent-structure/assumptions.md](latent-structure/assumptions.md) |
| A6. Measurement error handling depends on indicator count | Model measurement definitions | Editing and fit capability | [measurement-structure/assumptions.md](measurement-structure/assumptions.md) |
| A7. Measurement structure identification enables causal identification | Model measurement and causal structure | Identification | [causal-design/identifiability.md](causal-design/identifiability.md) |
| A8. Indicator residuals are temporally independent | Model observation laws | Editing, fit and simulation | [measurement-structure/assumptions.md](measurement-structure/assumptions.md) |
| A9. Single-indicator constructs absorb measurement error | Model measurement definitions | Editing and fit capability | [measurement-structure/assumptions.md](measurement-structure/assumptions.md) |
