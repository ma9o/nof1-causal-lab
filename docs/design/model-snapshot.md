# Model Access at a Committed Revision

`GET /api/episodes/{workspace_id}/model` reads one committed journal position. `?at_seq=N` selects an applied move; zero selects the empty episode. Rejected and raised attempts remain activity and cannot be selected as model revisions. The default is the latest applied move.

The reader loads the journal once and reads the artifact versions selected by that prefix. Historical facts retain their contents and freshness after later writes or retractions. Reading a model requires neither compilation nor inference.

## Identity and Ownership

The canonical [ModelSpec](../pipeline/latent-structure.md#modelspec) owns constructs, causal edges, parameters, the measurement clock, shared distributions, and trajectory time points. Constructs own [indicators](../pipeline/measurement-structure.md#indicator), intrinsic dynamics, and usage choices. Indicators own their likelihood. Edges own an additive collection of [mechanisms](../pipeline/statistical-model-spec.md#dynamicsmechanism).

Preserve an entity's ID while enriching or renaming it. A mechanism also retains its ID when reordered among other terms. Estimated coefficients reference scientific parameters whose owners include that mechanism. Fixed coefficients stay on their mechanism.

The compiler binds these identities to execution coordinates. It does not duplicate scientific definitions in a semantic catalog. Execution readiness is derived directly from the selected Model revision. It cannot retain an earlier revision’s approval after an edit.

## Read Contract

[ModelSnapshot](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshot_models.py) contains the canonical value and four distinct read groups:

| Field | Meaning |
|---|---|
| `model` | Optional `Sourced[ModelSpec]`, directly reusing the current scientific hierarchy |
| `context` | Workspace, selected journal sequence, artifact state, freshness, installation positions, retractions, and backend simulation availability |
| `data` | Independently sourced question, uploaded table profile, and extracted measurements |
| `findings` | Identification, structural dispositions, validation, admission, fit summaries, and baseline results |
| `findings.execution` | Unmet execution requirements or anchor certificates, sourced to the selected Model |

Each optional sourced value carries an artifact or transition-log reference, JSON pointer, and freshness. The default outcome is `model.value.default_outcome`; an indicator's likelihood is on the defining `cause` or `effect` endpoint under `model.value.edges[i]`, then `indicators[j].likelihood`. Sampler diagnostics are under `findings.fit.value.report.inference_diagnostics`; predictive checks are under `findings.fit.value.report.assessment`.

The model version and journal sequence are different identities. A model write increments the former; any journaled attempt advances the latter. Use `context.seq` to coordinate several reads of the same committed episode.

[ModelReader](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshots.py) exposes collection accessors backed by that same Model:

| GET endpoint below `/api/episodes/{workspace_id}/model` | Response |
|---|---|
| `/definition` | Optional `Sourced[ModelSpec]` |
| `/inference-report` | Optional `Sourced[InferenceReport]` from the transition log |
| `/constructs` | `Construct[]`, derived from unique graph endpoints |
| `/edges` | `CausalEdge[]` |
| `/indicators` | `Indicator[]`, obtained from construct ownership |
| `/parameters` | `ParameterSpec[]` |

Every accessor accepts `at_seq`. Collection reads do not construct batch views. Python accessors return the canonical objects; generated TypeScript consumes their serialized schema directly. The Next.js server forwards these typed HTTP reads.

## Results and Freshness

Scientific parameters carry a native NumPyro distribution or a reference to a shared law. The conditioned ModelSpec retains joint parameter and trajectory uncertainty; authoring evidence and inference metadata remain in logs. Ephemeral bindings derive logical element IDs and numerical coordinates. Scalar posterior findings refer to parameter and element IDs. Display names do not establish relationships.

The compiler declares category contrasts and ordinal cutpoints from category labels. Padded tensor coordinates are execution details. A scalar identity survives an execution-axis reorder; a Cholesky component includes its ordered basis because changing that basis changes the represented quantity.

Freshness follows each consumer's [actual Model inputs](../../apps/data-pipeline/src/nof1_causal_lab/models/model_inputs.py). A distribution edit preserves extraction, identification, and structural compilation inputs while invalidating inference-dependent findings. Labels can affect a consumer's input when it uses those labels. Original artifact pins are retained when a result is reusable.

Stale results remain available for inspection with their source marked stale. Current graph estimates and parameter attachments require fresh observational inputs and compatible structural findings. Inference freshness follows its scientific Model and panel inputs. Invalidation does not fit or simulate a replacement. A persistence prior converted to decay remains an exact transformed law; a posterior decay mean is not presented as a persistence mean.

All statistical summaries and composed artifact views are computed in Python. Raw-table profiles, extraction counts, histograms, and predictive comparisons use immutable input versions. The browser controls presentation and selection.

## Writes and Operation History

`PUT /api/episodes/{workspace_id}/model` accepts `{expected_version, model}`. Use version `0` for initial creation. The machine validates the full candidate, checks the base version, and commits the Model and required derivations together. A conflict returns HTTP 409; an invalid candidate returns HTTP 422. Failed writes cannot publish partial model or finding versions.

`latent_structure`, `measurement_structure`, `statistical_model_spec`, and `posterior` remain operation names. Each enriches the same Model through the shared commit boundary. `GET /api/episodes/{workspace_id}/operations/{operation_id}/traces` locates that operation's trace independently of later Model authorship.

The [offline migration](../guides/additive-model-migration.md) preserves retained scientific histories and numerical outputs while replacing the old persisted schemas.

## Generated Contracts

Run `bun run codegen` after Python contract changes and `bun run types:graph` to refresh the graph. The semantic view is `bun run types:graph --view semantic`. Generated clients are checked by `bun run codegen:check`.
