# Scientific actions

The scientific vocabulary is `edit_model`, `prepare_data`, `fit`, and `simulate`. Submit the [typed requests](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts.py) to `POST /api/episodes/{workspace_id}/actions`, or use the matching tools in `/api/tools/scientific`. The [registry](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py) defines their responsibilities; the [check catalog](model-checks.md) lists measurements and costs.

The four [action flowcharts](action-flows.md) show conditional checks, execution and publication inside each call, alongside the broader [Bayesian workflow overview](../assets/bayesian-workflow.svg).

| Action | Inputs | Result |
| --- | --- | --- |
| [`edit_model`](action-flows.md#edit_model) | Replacement `model` and `expected_revision`. | Saved model revision with applicable specification, identification, compatibility and automatic predictive findings. |
| [`prepare_data`](action-flows.md#prepare_data) | `source: {files: [...]}` with `preparation` and optional `max_windows`, or `source: {revision, replicate}` for one recorded simulation draw. | Prepared observations with their own schema, scoring instructions, provenance and numerical data profile; no model required. |
| [`fit`](action-flows.md#fit) | `model_revision`, `panel_revision`, optional numerical `settings`. | Conditioned model retaining joint parameter/state uncertainty, fit diagnostics and summaries. |
| [`simulate`](action-flows.md#simulate) | `model_revision`, `end`, optional `start`, `interventions` (default `[]`) and `comparison_panel_revision`. | Immutable generated arrays, simulation checks and optional predictive comparisons with the selected observations. |

## Dispatch and polling

All four actions use the same [asynchronous response contract](../../apps/data-pipeline/src/nof1_causal_lab/actions/results.py). Dispatch waits for durable acceptance and returns HTTP `202` with only `{"attempt_id":"<UUID>"}`. It never includes the scientific result, even when an edit completes quickly.

Poll `GET /api/episodes/{workspace_id}/actions/{attempt_id}`. Its response has three fields:

| Field | Meaning |
| --- | --- |
| `done` | Whether the attempt has finished. |
| `body` | The action's scientific result after successful publication; otherwise `null`. Includes immutable revision references and the applicable model, data or simulation report. |
| `messages` | All labels emitted so far, in order. Each has an ISO UTC `timestamp`, `level` (`debug`, `info`, `warn`, `error`) and `SCREAMING_SNAKE_CASE` `label`. |

Poll the same attempt until `done` is true. Each poll returns the complete message list; replace the prior list rather than appending it. A warning can accompany a saved result, such as `MODEL_INCOMPLETE` on an edit. An unsuccessful attempt returns `body: null` with an error label and leaves the scientific branch unchanged. Schema and authorization errors reject dispatch before acceptance through ordinary HTTP errors. Unknown attempt IDs return `404`.

The scientific tools return the same receipt inside `result`. Use `poll_action` with `{"attempt_id":"<UUID>"}` to read the same polling response. Published results and journaled messages remain readable without Temporal. An infrastructure failure that prevents writing history remains available through the durable workflow's polling query. V2 polls and displays action records as a read-only inspector; action submission belongs to the external agent.

## Editing and data

Structure, measurements, mechanisms, fixed values, estimable parameters and laws can be edited together or interleaved. Valid incomplete and causally unidentified candidates remain editable. Schema violations reject submission; missing execution inputs produce `not_evaluated` findings. [Specification checks](../../apps/data-pipeline/src/nof1_causal_lab/actions/checks.py) execute without fitting or generating trajectories.

The [action-owned evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py) runs a fixed sequence of applicable checks. Each group records a policy-versioned input fingerprint. An unchanged group reuses its report from the selected snapshot, retaining its original source revisions:

| Group | Relevant inputs |
| --- | --- |
| Specification | Execution definitions and current probability laws. |
| Identification | Causal graph and measurement definitions. |
| Compatibility | Measurement definitions, laws, panel and its data profile. |
| Predictive | Execution definitions, current laws, compatible panel, simulation policy and known conditioning history. |

`prepare_data` owns the empirical profile. Changing model laws reuses that profile while refreshing affected compatibility and predictive findings. Changing observation definitions can require preparing a new dataset. Profiles remain meaningful when a panel is incompatible with a later model. Preparation is independent of model authoring. [Extraction validation](../pipeline/extraction-validation.md) owns the data report definitions.

Explicit Python authoring helpers remain available for constructing [expressions](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/expressions.py), [conditional laws](../../apps/data-pipeline/src/nof1_causal_lab/models/likelihoods.py), [parameter-law revisions](../../apps/data-pipeline/src/nof1_causal_lab/models/model_distributions.py) and [measurement standardization choices](../../apps/data-pipeline/src/nof1_causal_lab/models/model_semantics.py). They are caller choices; `edit_model` does not complete or repair a submitted definition.

### Automatic predictive checks

`edit_model` runs one exact, whole-model simulation batch when the model supports simulation and a compatible prepared panel supplies the observation schedule. The [automatic evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/predictive_checks.py) uses 200 draws and seed 0, the model's joint state law at the start of the observed schedule, or its declared initial-state law when no histories are retained, nonlinear Diffrax dynamics and the true emission density. This is a reproducible default budget, not a calibrated Monte Carlo guarantee. Every selected C1–C5 measurement shares that batch. Predictive coverage, residual, spread and replicated-statistic comparisons consume that same batch and are recorded with the model checks. C4b edge-knockout experiments remain a separate internal analysis because each edge requires another batch.

A missing model component, incompatible panel, unsupported simulation law or inadequate schedule produces a typed `not_evaluated` reason. Readiness is specific to each check: an empirical joint law unsupported by the fitting engine can still be simulated. Non-finite paths produce failed findings and skip dependent reductions. Scientific failures save with the edit; unexpected execution failures terminate the action without publishing its candidate.

The result body's `predictive` report records the input key, source model/panel revisions, exact design, findings and law provenance. Known fitted laws are distinguished from authored or mixed laws; comparisons against the fitting panel are marked as in-sample posterior predictive. Imported joint laws with no known fitting history remain unknown. A different panel revision is labeled `posterior_predictive`; it does not prove the observations were held out. These comparisons do not establish held-out predictive performance.

Progress uses `MODEL_CHECKS_STARTED`; completion can include `PREDICTIVE_CHECK_FAILED`, `SIMULATION_CHECK_NOT_EVALUATED` and `PREDICTIVE_CHECKS_REUSED`. Messages contain only timestamp, level and label. Details stay in the typed result. The model, checks and logs publish in one Git commit, with `checks.json` retaining the selected snapshot's reports. `fit` refreshes inexpensive checks and produces no automatic predictive batch.

### Preparing uploaded files

Supply uploaded filenames explicitly in `source.files`. One action ingests those files, applies the supplied [DataPreparationSpec](../pipeline/extraction.md#datapreparationspec), runs the existing computed/semantic worker workflow, encodes observations and computes numerical data checks. The source shape selects this branch; there is no separate public raw-data preparation phase and no implicit newest-upload selection.

The preparation spec gives stable variable IDs, scoring rubrics, source columns, extraction modes, windows and codebooks. It is stored with the resulting data, so a stress score extracted from a diary retains its definition and worker traces. Neither a causal graph nor likelihoods or priors are needed. Simulation sources skip ingestion and extraction because their observations and definitions are already recorded.

Both branches return the same observations, metadata and profile. A later model binds its indicators to these stable IDs; `edit_model` and `fit` check that dtype, summaries, codebooks and windows agree. A model can use a subset of a dataset. New preparation leaves previous fit and simulation evidence in history, with its original input references.

### Preparing simulated observations

`prepare_data` accepts two data origins: provided data and an existing simulation. For simulated data, `source.revision` identifies the Git commit of an applied `simulate` action in the same study, and `source.replicate` selects one zero-based draw from that batch. The [materializer](../../apps/data-pipeline/src/nof1_causal_lab/actions/prepare_data.py) preserves that replicate's emitted values, missing observations, numeric category codes and measurement windows in the standard [observation panel](../pipeline/extraction.md#observationrecord). It uses a synthetic UTC calendar origin of 1970-01-01 for model days. It does not run the generator again or pool independent replicates.

The preparation commit records the exact simulation reference and replicate. Its metadata copies the observation schema, codebooks, support windows and mask recorded by the simulation, without loading the generating model. The source report retains the generating model, design, seed, true parameter draws and latent paths. Only observations enter the panel. The report records the selected window, interventions, resolved coordinates, draw count and seed.

Pass the resulting `panel_revision` to an ordinary `fit` request with the chosen inference model. Compatible measurement definitions permit different priors or dynamics. Reusing an existing panel is a direct input selection for fitting, not another `prepare_data` source. This closes the simulation-to-fitting path; recovery scoring, repeated calibration experiments and SBC rank diagnostics require separate orchestration. See the [SBC requirements](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html): generating and fitted joint models must agree when testing sampler calibration, while comparisons between Diffrax generation and Euler–Maruyama inference also assess discretization error.

## Selecting and comparing revisions

`GET /api/episodes/{workspace_id}/revisions` lists stored model, source and observation versions. The `revisions/model/{revision}` read returns an earlier scientific definition. Explicit selections are resolved against immutable storage; fitting checks the selected model/panel pair's measurement compatibility. Editing uses `expected_revision` to prevent overwriting concurrent changes. [Selection resolution](../../apps/data-pipeline/src/nof1_causal_lab/machine/selection.py) runs inside durable activities.

To refit an earlier definition, select that `model_revision` directly in `fit`. To continue an earlier complete study state, [fork its Git checkpoint](../design/study-history.md#agent-api), then submit actions with `branch` and the branch's `expected_head`. Editing checks `expected_revision` against the model on that branch. Fitting never silently restores earlier priors. The current particle engine accepts independent scalar parameter laws; simulation accepts both scalar and joint laws. Unsupported fitting laws are capability findings, not restrictions on saving the model.

`GET .../revisions/compare?before=<commit_id>&after=<commit_id>` selects two committed checkpoints and reports added, removed, pinned, released and revised parameters, changes in the selected graph, specification findings, and the fit/simulation evidence available at each checkpoint. The graph comparison includes exclusions established by execution dispositions. Check the reported data selections and simulation designs before interpreting differences. The [comparison implementation](../../apps/data-pipeline/src/nof1_causal_lab/actions/revisions.py) performs this work in Python; the web view presents the result.

The [workflow diagram](../design/study-history.md#choose-your-focus) shows how these choices repeat within and across branches. Action logs, diagnostics and traces belong to their owning Git commits.

## Fitting

The [fit action](../../apps/data-pipeline/src/nof1_causal_lab/actions/fit.py) retains the nonlinear particle target, joint parameter/state uncertainty, sampler telemetry, parameter summaries and configured PSIS-LOO. `settings` can override sample count, warmup count, chain count, particle count and seed. It produces no predictive batch. The [inference report](../pipeline/inference.md#inferencereport) owns those outputs.

## Simulation

The [request](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts.py) selects a model revision and an absolute `end` time in model days. `start` is optional and `interventions` defaults to an empty array. There are no separate prior, posterior or causal generation modes.

```json
{
  "action": "simulate",
  "model_revision": "<model tree OID>",
  "end": 30,
  "interventions": []
}
```

Without comparison data, an omitted start uses the latest retained joint state time, or day zero when the model supplies only its initial-state law. At an explicit start, the [generator](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/predictive/simulation.py) uses the jointly sampled retained state at or before that time and advances it when needed. A request before the first retained state is invalid. Without retained histories, the declared initial-state law supplies the state at the requested start. Historical starts use the uncertainty in the selected model, including any later evidence that conditioned it; select an earlier model revision to use earlier beliefs.

Without comparison data, the framework derives output times from the model clock and inserts intervention times. Explicit simulations use 100 draws and seed 0. All use nonlinear stochastic Diffrax dynamics and the true emission model. Measurement windows extending before the simulation start produce absent observations. These execution choices are framework policy; the report records the resolved times, draw count and seed.

An intervention has one shape, with an absolute model timestamp:

```json
{"target": "<construct ID>", "time": 10, "value": 3}
```

It assigns the state once at `time`, after which the model's dynamics resume. A non-varying state retains the value; a varying state evolves from it. Later events may assign the same state again. Values and times must be finite, intervention times must fall within the simulation window, and two assignments to the same state at the same time are rejected.

With interventions, the [paired generator](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/counterfactual/orchestration.py) shares parameter/state draws, integration segments and random streams between reference and action histories. Empty interventions generate an ordinary history without a redundant reference batch.

With `comparison_panel_revision`, the action first binds the dataset schema to the model, then generates on the actual observation times and support windows, preserving missingness. An omitted start uses the first comparison time; the requested end must match the last comparison time. Comparison requests have no interventions. The same batch supplies C5a/C5b and predictive coverage, residual, spread and replicated-statistic checks, recorded in `SimulationReport.predictive_checks`. With a fitted model this is a posterior predictive check. The report pins the comparison dataset and records whether the laws are authored, fitted, mixed or unknown; using the fitting panel is explicitly in-sample. Automatic edit comparisons use the same generator and reducers and remain in `ModelPredictiveReport.predictive_checks`. The [causal reducer](../../apps/data-pipeline/src/nof1_causal_lab/actions/scenarios.py) uses the generated paired arrays to summarize the model's default outcome only when identification and matching production-fit evidence support it. Otherwise `causal_unavailable_reason` records why no numeric causal effect is reported, while the generated histories remain available.

## Simulation report

The single [SimulationReport](../pipeline/analysis.md#simulationreport) records the submitted window and interventions, resolved execution settings, immutable arrays and derived findings. Reports live in simulation journal records and under `findings.simulation`. Later model revisions mark earlier reports historical; their evidence remains available in the timeline. [Causal effect fields](../pipeline/analysis.md#causaleffectresult) are optional derived readouts of the same histories.

## Optional recipes and retained workspaces

Only the four actions submit scientific work. External agents author whole-model revisions through `edit_model`; there is no internal proposal, construct-admission, repair or checkpoint-rebase loop. The durable executor retains revision checks, job execution, polling, atomic publication and action history. Automatic predictive checks belong to model edits and never dispatch a nested `simulate` action. Data preparation runs only data checks. Fitting does not launch prediction; use `simulate` with a comparison panel for an explicit post-fit check.
