# Scientific actions

The scientific vocabulary is `edit_model`, `prepare_data`, `fit`, and `simulate`. Submit the [typed requests](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts.py) to `POST /api/episodes/{workspace_id}/actions`, or use the matching tools in `/api/tools/scientific`. The [registry](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py) defines their responsibilities; the [check catalog](model-checks.md) lists measurements and costs.

| Action | Inputs | Result |
| --- | --- | --- |
| `edit_model` | Replacement `model` and `expected_version`. | Saved model revision with applicable specification, identification and compatibility findings. |
| `prepare_data` | `source: "files"`; or `source: "raw_data"` with `raw_data_version`, `model_version`, and optional `max_windows`. | Versioned sources or observations, empirical profiles and compatibility findings. |
| `fit` | `model_version`, `panel_version`, optional numerical `settings`. | Conditioned model retaining joint parameter/state uncertainty, fit diagnostics and summaries. |
| `simulate` | `model_version`, explicit `design`, optional `comparison_panel_version`. | Immutable generated arrays and a separately recorded simulation report. |

## Editing and data

Structure, measurements, mechanisms, fixed values, estimable parameters and laws can be edited together or interleaved. Valid incomplete and causally unidentified candidates remain editable. Schema violations reject submission; missing execution inputs produce `not_evaluated` findings. [Specification checks](../../apps/data-pipeline/src/nof1_causal_lab/actions/checks.py) execute without fitting or generating trajectories.

The [derivation cascade](../../apps/data-pipeline/src/nof1_causal_lab/machine/derivations.py) separates three dependency sets:

- Model-only: execution capability and causal identification.
- Data-only: a `data_profile` depends on its observation panel, with empirical summaries, timestamp parsing and numeric/sample-size checks.
- Model/data: `validation_report` joins the profile to expected indicators, declared value domains, temporal assumptions, measurement support and fit preflight checks. Missing observations are reported here, outside simulation.

Changing model laws reuses the empirical profile while refreshing affected compatibility findings. Changing measurement definitions can require preparing observations again. Profiles remain meaningful when a panel is incompatible with a later model. Importing sources is independent of model authoring. [Extraction validation](../pipeline/extraction-validation.md) owns the data report definitions.

## Selecting and comparing revisions

`GET /api/episodes/{workspace_id}/revisions` lists stored model, source and observation versions. The `revisions/model/{version}` read returns an earlier scientific definition. Explicit selections are resolved against immutable storage; fitting checks the selected model/panel pair's measurement provenance. Editing uses `expected_version` to prevent overwriting concurrent changes. [Selection resolution](../../apps/data-pipeline/src/nof1_causal_lab/machine/selection.py) runs inside durable activities.

To refit an earlier definition, select that `model_version` directly in `fit`. To revise it, load its definition and submit an edit against the current `expected_version`. Fitting never silently restores earlier priors. The current particle engine accepts independent scalar parameter laws; simulation accepts both scalar and joint laws. Unsupported fitting laws are capability findings, not restrictions on saving the model.

`GET .../revisions/compare?before=1&after=2` reports added, removed, pinned, released and revised parameters, changed scientific dependencies, specification findings, and any recorded fit/simulation evidence for each revision. Check the reported data selections and simulation designs before interpreting differences. The [comparison implementation](../../apps/data-pipeline/src/nof1_causal_lab/actions/revisions.py) performs this work in Python; the web view presents the result.

## Fitting

The [fit action](../../apps/data-pipeline/src/nof1_causal_lab/actions/fit.py) retains the nonlinear particle target, joint parameter/state uncertainty, sampler telemetry, parameter summaries and configured PSIS-LOO. `settings` can override sample count, warmup count, chain count, particle count and seed. It produces no predictive batch. The [inference report](../pipeline/inference.md#inferencereport) owns those outputs.

## Simulation designs

A trajectory design uses `kind: "trajectory"`, strictly increasing `times` in model days, `draws` and `seed`. The [contract](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/simulation.py) makes the following choices explicit:

| Choice | Meaning |
| --- | --- |
| `initial_state: "new_study"` | Draw a fresh state from the declared initial-state law. |
| `initial_state: "retained"` | Draw parameters and a retained state together from the current joint law (conditioned on all data used in that fit); `state_time` must match a retained model time and the first design time. |
| `initial_state: "fixed"` | Use `state_values`, naming every state. Parameter uncertainty remains sampled. |
| `initial_state: "equilibrium"` | Compute each draw's nonlinear drift equilibrium. |
| `process_noise` | Include process diffusion, or follow deterministic nonlinear drift. |
| `observation_noise` | Sample the true emission density, or return conditional emission means. |
| `interventions` | Timed set, shift, ramp or trajectory clamps; include their boundaries in the design grid. These are model explorations unless certified as causal below. |
| `context`, `checks` | Record exploration, calibration or prediction intent; select dynamics, measurement and/or data comparisons. |
| `confinement_growth_ratio`, `confinement_failure_fraction` | Explicit criteria for interpreting confinement measurements. |
| `edge_contrasts` | Add paired edge-off calibration experiments using the same parameter draws and random streams. |
| `comparison_time_offset` | Align the comparison panel's relative time origin to the design. |

The [law sampler](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/parameters.py) samples each joint event once, preserving parameter/state dependence. The [generator](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/registry_runtime.py) and [intervention orchestration](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/counterfactual/orchestration.py) use the true nonlinear field and Diffrax. Paired interventions share initial draws, integration segments and process/emission random streams; hard clamps remove diffusion in the clamped coordinate.

Comparison panels must match the design's prepared grid, including support boundaries. Time zero in a panel is its first observation anchor; preceding support boundaries have negative times. All designs honor declared measurement windows; outputs requiring unavailable prehistory are absent. Comparisons honor missing observations and interval support. Predictive calibration requires sampled emission noise and comparison observations; otherwise it is explicitly unevaluated. Ordinary checks reuse the generated batch. [Shared measurements](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/simulation_checks.py) do not decide whether a candidate may be saved.

A causal design uses `kind: "causal"` and a `query` containing start, clamps, outcome and readout, plus draw/seed/noise choices. It uses the same generator. [Causal certification](../../apps/data-pipeline/src/nof1_causal_lab/actions/scenarios.py) checks identification and the committed production-fit evidence for the selected immutable model before reporting numeric effects. The baseline start uses equilibrium; an abducted start uses the aligned retained state. Deterministic drift is the default and is recorded explicitly; enabling process noise generates paired stochastic paths.

## Simulation report

| Field | Description |
| --- | --- |
| `model`, `comparison_panel_version` | Exact supporting revisions. |
| `design` | Conditioning, schedule, noise, interventions, context and measurement criteria. |
| `state_ids`, `indicator_ids` | Scientific identities in array-axis order. |
| `parameter_draws`, `latent_paths`, `observations` | Immutable workspace array references. |
| `reference_latent_paths`, `reference_observations` | Paired reference arrays for interventions. |
| `findings` | Measurements and their criteria; `passed: null` means unevaluated. |
| `predictive_checks` | Optional calibration, residual, spread, overlay and replicated-statistic comparisons. |
| `causal_result` | Certified scenario readout when requested. |

Reports live in simulation journal records and under `findings.simulation`. Later input revisions mark earlier reports historical. Their original evidence remains available in the timeline. Causal results are durable; they are no longer ephemeral analysis-tool responses.

## Optional recipes and retained workspaces

The web surface leads with the four actions and offers the observational-study recipe separately. `POST .../recipes/observational-study` runs that optional orchestration. [Incremental authoring](../../apps/data-pipeline/src/nof1_causal_lab/recipes/incremental_model.py) and [construct admission](../../apps/data-pipeline/src/nof1_causal_lab/recipes/construct_authoring.py) own their proposal, acceptance and repair loops. The durable machine retains revision checks, job execution, cancellation and provenance.

`InferenceReport` contains no predictive checks. The DEMO measurements remain in [predictive_checks.json](../../data/DEMO/fixture/predictive_checks.json) because their original arrays were not retained. For other workspaces, run the [offline migration](../../apps/data-pipeline/scripts/migrate_scientific_actions.py) with source and new destination directories. `--check` inventories changes. It copies the workspace, removes legacy `ppc` fields from fit reports, preserves their contents and provenance under `episode/predictive-archive/`, and records source hashes. Source files remain intact; the migration creates no fictitious arrays or simulation designs. There is no runtime compatibility adapter.
