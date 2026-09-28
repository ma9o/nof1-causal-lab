# Model Access at a Committed Revision

`GET /api/episodes/{workspace_id}/model` reads the selected branch's Git head. `?at=<commit_id>` selects an immutable applied commit, including the empty study root. Rejected and raised attempts retain their own logs without advancing scientific state.

The reader derives artifact selections from the commit's native tree and finds earlier evidence through Git ancestry. Historical facts retain their contents and freshness after later writes or retractions. Reading a model requires neither compilation nor inference.

## Identity and Ownership

The canonical [ModelSpec](../pipeline/latent-structure.md#modelspec) owns the research question, constructs, causal edges, parameters, the measurement clock, shared distributions, and trajectory time points. Constructs own [indicators](../pipeline/measurement-structure.md#indicatorspec), intrinsic dynamics, and usage choices. Indicators own their likelihood. Edges own an additive collection of [mechanisms](../pipeline/statistical-model-spec.md#dynamicsmechanismspec).

Preserve an entity's ID while enriching or renaming it. A mechanism also retains its ID when reordered among other terms. Estimated coefficients reference scientific parameters whose owners include that mechanism. Fixed coefficients stay on their mechanism.

The compiler binds these identities to execution coordinates. It does not duplicate scientific definitions in a semantic catalog. Numerical operations [validate the selected definition](../reference/compilation.md) and their actual inputs when preparing execution. Snapshot reads preserve partial authoring values without running those execution checks.

## Read Contract

[ModelSnapshot](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshot_models.py) contains the canonical value and four distinct read groups:

| Field | Meaning |
|---|---|
| `model` | Optional `Sourced[ModelSpec]`, directly reusing the current scientific hierarchy |
| `context` | Workspace, branch, commit OID, display sequence, derived artifact state, freshness and backend simulation availability |
| `data` | Independently sourced uploaded table profile and extracted measurements |
| `findings` | Identification, structural dispositions, validation, prior-predictive results, and fit summaries |
| `findings.graph` | Backend-selected construct and edge IDs: authored structure before execution dispositions exist, then the retained component connected to the default outcome; the full scientific definition remains in `model` |
| `findings.prior_predictive` | [Prior-predictive samples and checks](../pipeline/statistical-model-spec.md#priorpredictiveresult), sourced to the model-authoring journal record and checked against its model and data inputs |

[Git study history](study-history.md) owns snapshots, branches and commit-local logs. `context.commit_id` is the immutable Git identity; `context.branch` records the requested branch.

Each optional sourced value carries one `GitRef` (`workspace_id`, `revision`, `path`), a JSON pointer and freshness. `context.commit_id` selects the Git snapshot; `context.seq` is an activity label. The research question is `model.value.question`, including in a question-only initial revision. The default outcome ID is `model.value.default_outcome`; an indicator's likelihood is on the defining `cause` or `effect` endpoint under `model.value.edges[i]`, then `indicators[j].likelihood`. Sampler diagnostics are under `findings.fit.value.report.inference_diagnostics`; [predictive checks and held-out evaluation](../pipeline/inference.md#inferencereport) are under `findings.fit.value.report.ppc` and `findings.fit.value.report.loo_diagnostics`.

Git tree OIDs identify immutable artifacts; commit OIDs identify complete study snapshots and their action logs. Use `context.commit_id` to coordinate reads of the same state.

`GET /api/episodes/{workspace_id}/revisions/compare?before=<commit_id>&after=<commit_id>` compares any two committed checkpoints containing a model. It uses the same graph selection as snapshot reads, including structural-to-execution exclusions, and returns the simulation and fit evidence available at each checkpoint. Marginalized and unsupported entities remain in the scientific model; comparison findings carry their execution dispositions and reasons.

The [backend selection](../../apps/data-pipeline/src/nof1_causal_lab/models/model_structure.py) drops components disconnected from the default outcome after structural projection, including their execution indicators and parameters. Shared latent causes, measurement loadings, coefficients, and joint laws preserve statistical connections. Without a default outcome, execution covers all measured components. Disconnected constructs remain in the scientific definition with an exclusion reason available through the same history comparison. Unresolved causal parents of retained states still fail execution checks.

[ModelReader](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshots.py) exposes collection accessors backed by that same Model:

| GET endpoint below `/api/episodes/{workspace_id}/model` | Response |
|---|---|
| `/definition` | Optional `Sourced[ModelSpec]` |
| `/inference-report` | Optional `Sourced[InferenceReport]` from the transition log |
| `/constructs` | `ConstructSpec[]`, derived from unique graph endpoints |
| `/edges` | `CausalEdgeSpec[]` |
| `/indicators` | `IndicatorSpec[]`, obtained from construct ownership |
| `/parameters` | `ParameterSpec[]` |

Every accessor accepts `at` and `branch`. Collection reads do not construct batch views. Python accessors return the canonical objects; generated TypeScript consumes their serialized schema directly. The Next.js server forwards these typed HTTP reads.

## Workbench UI Logic

[`lib/model-asset`](../../apps/web/src/lib/model-asset) owns the workbench's shared UI behavior. Its [entity resolver](../../apps/web/src/lib/model-asset/entities.ts) derives indexes, labels, relationships, and editable fields from the selected graph; inspection and editing use the same resolution. [Scoped edits](../../apps/web/src/lib/model-asset/edit.ts) assemble a whole-model candidate while preserving endpoint ownership and references.

The [workbench hook](../../apps/web/src/lib/model-asset/use-workbench.ts) owns selection, revision navigation, comparison previews, and open actions. The [form hook](../../apps/web/src/lib/model-asset/use-action-form.ts) owns drafts and request assembly; the [submission hook](../../apps/web/src/lib/model-asset/use-scientific-action.ts) owns input revisions, optimistic writes, results, and cache refresh. Components render these values, dispatch interactions, and manage DOM focus. Scientific validation and numerical computation remain on the server.

[Graph UI logic](../../apps/web/src/lib/dag) owns layout inputs, comparison placement, selection highlighting, zoom, and playback of recorded trajectories. [Inspector selectors](../../apps/web/src/lib/model-asset/inspector.ts) format findings using parameter ownership supplied by their caller. [Timeline navigation](../../apps/web/src/lib/model-asset/use-revision-timeline.ts) retains the selected branch while browsing ancestors. [Admission presentation](../../apps/web/src/lib/admission) owns live report selection and uses the check modes recorded by the server. [Table hooks](../../apps/web/src/lib/tables) own filtering, sorting, grouping, expansion, virtualization, and keyboard navigation in displayed row order. These libraries do not import rendering components; ESLint enforces that dependency direction.

## Results and Freshness

Scientific parameters carry a native NumPyro distribution or a reference to a shared law. The conditioned ModelSpec retains joint parameter and trajectory uncertainty; authoring evidence and inference metadata remain in logs. Ephemeral bindings derive logical element IDs and numerical coordinates. Scalar posterior findings refer to parameter and element IDs. Display names do not establish relationships.

The compiler declares category contrasts and ordinal cutpoints from category labels. Padded tensor coordinates are execution details. A scalar identity survives an execution-axis reorder; a Cholesky component includes its ordered basis because changing that basis changes the represented quantity.

Freshness follows each consumer's [actual Model inputs](../../apps/data-pipeline/src/nof1_causal_lab/models/model_inputs.py). A question edit invalidates extraction, whose LLM workers consume the text, while preserving numerical compilation and identification inputs. Refitting and prior authoring recover the original unconditioned law through model ancestry and retain the selected revision’s question, so an intervening question edit cannot turn a posterior into a fresh prior. A distribution edit preserves extraction, identification, and structural compilation inputs while invalidating inference-dependent findings. Labels can affect a consumer's input when it uses those labels. Original artifact pins are retained when a result is reusable.

Stale results remain available for inspection with their source marked stale. Current graph estimates and parameter attachments require fresh observational inputs and compatible structural findings. Inference freshness follows its scientific Model and panel inputs. Invalidation does not fit or simulate a replacement. A persistence prior converted to decay remains an exact transformed law; a posterior decay mean is not presented as a persistence mean.

All statistical summaries and composed artifact views are computed in Python. Raw-table profiles, extraction counts, histograms, and predictive comparisons use immutable input versions. The browser controls presentation and selection.

## Writes and Operation History

Agents create and continue branches under the hood. The workbench follows the latest committed work and lets users inspect and compare checkpoints. Its timeline presents the server's Git ancestry instead of inferring branches from model input pins.

`PUT /api/episodes/{workspace_id}/model` accepts `{expected_revision, model}`. Use `null` for initial creation; otherwise use the selected model tree OID. The machine validates the full candidate, checks the expected tree OID, and commits the Model and required derivations together. A conflict returns HTTP 409; an invalid candidate returns HTTP 422. Failed writes cannot publish partial model or finding versions.

`latent_structure`, `measurement_structure`, `statistical_model_spec`, and `posterior` remain operation names. Each enriches the same Model through the shared commit boundary. `GET /api/episodes/{workspace_id}/operations/{operation_id}/traces` locates that operation's trace independently of later Model authorship.

The [offline migration](../guides/additive-model-migration.md) preserves retained scientific histories and numerical outputs while replacing the old persisted schemas.

## Generated Contracts

Run `bun run codegen` after Python contract changes and `bun run types:graph` to refresh the graph. The semantic view is `bun run types:graph --view semantic`. Generated clients are checked by `bun run codegen:check`.
