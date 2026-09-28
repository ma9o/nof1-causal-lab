# Actions, Operations, and Artifacts

The LLM calls four [scientific actions](../reference/scientific-actions.md): `edit_model`, `prepare_data`, `fit`, and `simulate`. The [typed requests](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts.py) enter the [episode workflow](../../apps/data-pipeline/src/nof1_causal_lab/machine/temporal/workflow.py) directly. The workflow serializes execution, applies effects, and records each applied, rejected, or raised attempt.

The four [action flowcharts](../reference/action-flows.md) show each request's conditional checks, execution and publication.

## One Scientific Definition

The canonical [ModelSpec](../pipeline/latent-structure.md#modelspec) gains detail while its scientific entities retain identity. Structure, measurements, mechanisms, parameters, and laws can be edited together or interleaved. The caller supplies the candidate to `edit_model`; that action validates and stores it without invoking an internal authoring prompt builder.

| Entity | Owned detail |
| --- | --- |
| Construct | Indicators, intrinsic dynamics, and execution usage |
| Indicator | Likelihood and measurement choices |
| Causal edge | Additive mechanisms, each with a stable identity |
| Parameter | Fixed value or current native law and supporting evidence |

Identification, predictive checks, and posterior findings are separately sourced results. They describe an exact model revision.

## Execution and History

| Concept | Contract |
| --- | --- |
| Scientific action | One of the four typed requests |
| Implementation job | A private execution detail chosen from the request's inputs |
| Artifact | An immutable payload, identified by `ArtifactId` and its Git tree OID |
| Action check | An applicable check evaluated or reused before publication, including automatic predictive simulation for model edits |
| Journal record | Action name, recorded inputs, status, effects, diagnostics, and trace identities |

There is no run/write command union or actor classification. Historical records retain their private `operation_id` for locating existing traces and findings; it is not a public execution choice.

| Action | Execution |
| --- | --- |
| `edit_model` | Validate the supplied model, evaluate or reuse applicable checks, and publish a revision with its findings |
| `prepare_data` | Infer the source branch, ingest and extract uploaded files or materialize one recorded replicate, then run data-only numerical checks |
| `fit` | Condition an executable model on the selected observation panel |
| `simulate` | Generate trajectories, predictive comparisons or certified causal scenarios from the selected model and design |

The [job graph](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py) describes implementation dependencies. Latent, measurement, and statistical authoring modules remain private implementation code for authoring experiments and historical readers. The public workflow does not dispatch those authoring jobs or an automatic recipe.

Fitting records diagnostics without launching a predictive batch. Prediction and [intervention simulation](../pipeline/analysis.md) both use `simulate`. Numerical causal effects additionally require positive identification and matching production-fit evidence.

## Atomic Model Edits

A model can begin with a research question and no edges. `edit_model` names its `expected_revision`: `null` for creation, otherwise the selected model's Git tree OID. A stale base rejects the action. The [model writer](../../apps/data-pipeline/src/nof1_causal_lab/actions/edit_model.py) stages the candidate, then the [check evaluator](../../apps/data-pipeline/src/nof1_causal_lab/actions/model_checks.py) completes applicable checks before the branch advances.

| Derived artifact | Inputs |
| --- | --- |
| `identification_report` | Model |
| `data_profile` (owned by `prepare_data`) | Panel and its own observation schema |
| `validation_report` | Panel, model, and data profile |

Incomplete scientific choices remain valid during authoring. Identification reports preserve negative findings. Missing prerequisites can retract a derived output; that retraction remains visible in history. Failed publication leaves the previous study snapshot current.

## Input Revisions and Freshness

Every computed result records its exact input revisions. Model consumers also record fingerprints of the scientific fields they used. A prior edit can preserve extraction and identification while invalidating inference. Existing results keep their original input references.

An edit refreshes affected model checks, including an automatic predictive batch when its inputs changed and its prerequisites hold. Data preparation refreshes only the empirical profile; it does not load a model or refresh fitting evidence. Fitting and additional simulation designs remain explicit actions. The [model reader](model-snapshot.md) exposes the selected definition, observations, and findings, with current estimates requiring compatible fresh inputs. Earlier results remain available for inspection through [study history](study-history.md).

## Transport

HTTP exposes `POST /api/episodes/{id}/actions` and the equivalent four `scientific` tools through generated Python-owned contracts. There is no alternate scientific write endpoint. Reads, uploads, and branch management support those actions. The Next.js layer forwards requests and presents results.

Existing histories require the [offline history migrations](study-history.md#migrating-existing-local-studies). Runtime readers accept the current contract only.
