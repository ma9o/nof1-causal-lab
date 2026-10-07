# Goals and design

This is a framework to facilitate bayesian modeling and inference with LLM, trying to fight their tendency to overfit a modeling problem solution to its specification via defaults and DSL restrictions.

The DSL is 7 actions:
- edit_question
- edit_model
- prepare_data
- fit
- simulate
- data_diff
- model_diff

The LLM user can only interface with the problem with these 7 core actions. Nothing else. 

Each action materializes some new state. No action is read-only, regardless of whether its materialized result is a simple or complex derivation from its inputs. 

The lineage fo actions is tracked with git. Inputs and outputs appear as git reference hashes. An input can be specified as 'latest' and will pull the latest VALID input for the correct input type.

Actions are cached and indepondent so calling one wiht the same arguments will return the result without recomputing.

A human user can only read the progress the LLM makes from the web UI. If the LLM needs steering it will interact with the cli harness that is under the hood.

# Code practices

Immutability, functional programming, strong typing e2e. Parse, don't validate. reudce implific dependencies as much as possible.

# Core actions

Every POST request uses the envelope `{action, input, reasoning}`. `action` selects one of the seven actions and its typed `input` schema. Action-specific arguments belong inside `input`; optional `reasoning` is a top-level field describing why the action is being called. Parse requests as a discriminated union so each action accepts only its own input type.

Actions are run with a polling model: the LLM starts an action via POST. POST and polling GET responses use the same envelope:

- `call_id`: a stable identifier for `action` and the canonical, resolved scientific arguments in `input`. The top-level `reasoning` field does not affect this identifier. Identical resolved calls have the same identifier.
- `action`: the action name, such as `edit_model`.
- `status`: `running`, `failed`, or `success`.
- `commit_id`: the Git hash of the recorded outcome, or null until publication. A recorded failure also has a commit; a failure before publication may leave this null.
- `body`: null while running or failed; the action-specific scientific outputs on success.
- `messages`: the accumulated execution log, including lifecycle messages, structured progress, LLM/tool traces, and full failure details.

The client polls `GET /api/studies/{workspace_id}/{action}/{call_id}`. Polling reads the existing call without starting or retrying execution. Inputs such as `latest` are resolved when POST accepts the call; polling keeps that same call even if newer revisions appear.

`messages` is the only execution-log field. Each response returns the accumulated entries so far, including after success or failure. Preserve the structured progress and trace information in those entries, rather than reducing them to labels. There are no separate `events`, `traces`, or `error` fields, and execution logs do not move into `body`.

## edit_question

- Sets the question the study wants to answer in natural language (just for tracking), and most importantly defines the outcome node of the study and the intervention nodes with the value changes needed to answer the question. 
- Each edit_model will need to include the specified nodes in its model specification.
- If the LLM user wants to restructure the study, they will need to invoke edit_question again.

## edit_model

Used to specify the SSM. It can be used for structural (nodes and edges) and/or mathematical (setting priors and mechanisms) specification.

The `input.parent_ref` field selects a question revision for an initial model or a model revision to edit. `latest` selects the current model, otherwise the current question; an exact Git hash pins either parent without requiring it to be the current head. The resulting action commit is returned as `commit_id`.

The LLM will decide here based on the study quesiton which parematers will be fixed and hwich will be free - inferred during fit.

Another importantn paramter is model_clock, which sets the grenularity of the causal model.

Depending on the type of specification, edit_model will run structural or mathematical checks. None of the checks depend on data.

The strucural checks ensure that the SSM specifed adheres to a restricted class of SSMs:

- Node types: known inputs, latent constructs, and indicators.
- One shared model clock sets the default observation window and reference interval for time-based priors. Indicators may have different windows and irregular observation times. What this means is that arrows between nodes must have a causal effect plausible within at most the time slice of the set model_clock. Any other effect beyond the model clock is encoded in the dynamics, instead of as an arrow.
- No cycles within a time slice. Feedback between evolving constructs across time is allowed.
- Known inputs are treated as measured without error, have no incoming causal arrows, and have no modeled dynamics or noise of their own. 
- The current state contains all modeled memory. Explicit delays and higher-order lags are not supported.
- Time-invariant constructs stay fixed and cannot have time-varying parents.
- Measurement arrows go from constructs to indicators. Indicators have no causal effects or dynamics of their own.
- Measurement errors are independent across indicators and over time, conditional on the constructs.
- Hidden common causes must appear as explicit latent nodes.
- The model must include the question’s outcome and intervention nodes. The outcome must be a modeled construct, and every other construct must have a directed path to it.


The mathematical checks verify some basic properties fo the model once the priors are populated:

- All required parameters have priors or fixed values, and the equations refer to defined quantities.
- Priors respect the constraints of their parameters, such as positive noise scales and valid correlations.
- Probability distributions have the right dimensions for the quantities they describe.
- Each hidden construct has a fixed reference for its zero point and scale.
- Detectable cases where parameters cannot be learned separately are flagged, such as coefficients that only appear through their sum or product.
- Priors expressed per time step are checked against their continuous-time conversion, using a simplified linear reference.

## data_diff

Used to compare saved data assets, with a semantic return appropriate to the inputs:

Prior simulations and observed data: prior predictive checks, assessing whether the model’s assumptions generate plausible observations.

Posterior simulations and observed data: posterior predictive checks, assessing agreement between simulated and observed behavior.

Two observed datasets: differences in recorded values, missingness, observation schedules, measurement definitions, and descriptive statistics. This includes comparing revisions produced by different data preparation choices.

Prior and posterior simulations: comparison of predictive behavior before and after conditioning on data.

Two simulation ensembles: comparison of predictive behavior across model specifications, priors, fitted models, or intervention scenarios.

Two individual histories, observed or simulated: observation-level additions, removals, and changes.

## model_diff

...

## prepare_data

...


# fit

Accepts a model gitref, a data gitref, and a mandatory, zero-based replicate_index.

- For prepared user data, replicate_index must be 0.
- For simulation output, it selects one recorded history and must be within the simulation’s replicate count
- The data gitref and replicate index are part of the call’s cache identity and retained provenance.

Fits the model's free parameters.

# simulate
