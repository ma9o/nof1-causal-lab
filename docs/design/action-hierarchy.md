# Actions, Operations, and Artifacts

The navigator chooses an operation; the machine applies one committed effect; delegated authoring contexts make scientific decisions. These are distinct responsibilities. Their current contracts are defined by the [machine graph](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py) and [context hierarchy](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py).

## One Scientific Definition

The canonical [ModelSpec](../pipeline/latent-structure.md#modelspec) gains detail while its scientific entities retain identity. Latent structure, measurement structure, and statistical specification describe authoring work on this value. Each operation submits a whole candidate through the same optimistic revision boundary.

| Entity | Owned detail |
|---|---|
| Construct | Indicators, intrinsic dynamics, and execution usage |
| Indicator | Likelihood and measurement choices |
| Causal edge | Additive mechanisms, each with a stable identity |
| Parameter | Quantity, owners, native NumPyro prior, and supporting evidence |

Identification, admission, and posterior findings are separately sourced results. They describe a pinned Model rather than becoming another scientific definition.

## Machine Vocabulary

| Concept | Contract |
|---|---|
| Operation | A callable unit of work, identified by `OperationId` |
| Artifact | An immutable, versioned payload, identified by `ArtifactId` |
| Run move | `{kind: "run", operation_id: ...}` |
| Write move | `{kind: "write", artifact_id: ..., provenance: ...}`; Model writes also require `expected_model_version` |
| Derivation | Deterministic computation from declared current artifacts, completed within a committing move |
| Journal record | One applied, rejected, or raised attempt, including effects and trace identities |

Operation names and artifact names have separate types. Several operations produce a new version of `model`. Operation ordering is therefore distinct from artifact dependency ordering.

## Operations

| Operation | Inputs | Main effects | Scientific definition |
|---|---|---|---|
| `raw_data` | Uploaded source files | Raw Arrow table with column descriptions | [Ingestion](../pipeline/ingestion.md) |
| `latent_structure` | Question; existing Model when present | Propose or revise constructs and edges on the Model | [Latent authoring](../pipeline/latent-structure.md) |
| `measurement_structure` | Question, raw data, Model | Add owned indicators and usage to the Model | [Measurement authoring](../pipeline/measurement-structure.md) |
| `measurements` | Question, raw data, Model | Extract observations and, when usable, a panel | [Extraction](../pipeline/extraction.md) |
| `statistical_model_spec` | Question, Model, plan, identification, panel, validation | Commit the completed Model and admission report | [Statistical authoring](../pipeline/statistical-model-spec.md) |
| `posterior` | Executable Model and panel | Joint posterior and fit diagnostics | [Inference](../pipeline/inference.md) |
| `baseline_report` | Posterior, Model, identification | Identified effect summaries and commentary | [Analysis](../pipeline/analysis.md) |

The default navigator uses this order to find missing or stale work. Legal moves are determined by declared inputs; the machine permits explicit navigation when those inputs exist. Numerical effect computation also requires a positive identification verdict for the selected treatment and outcome.

## Derivations and Writes

| Artifact | Derivation inputs |
|---|---|
| `identification_report` | Model |
| `validation_report` | Panel and Model |

The writable roots are question and Model. Baseline reports also permit an explicit authored write, including retained simulation results with their requests and provenance. Scientific authoring operations use the same Model writer as the public HTTP API.

A Model write names its expected base version (`0` for creation), validates all ownership and references, and completes required derivations before publication. If validation or a final write fails, its partial versions are removed and the previous state stays current. Incomplete scientific choices remain valid during authoring; executable outputs appear only when their requirements are met.

Identification reports preserve negative findings. Missing prerequisites can retract a derived output; that retraction remains visible in history.

## Freshness and Provenance

Every result pins its actual input versions. Model consumers additionally record a fingerprint of the scientific input they used. A prior revision can preserve extraction and identification while invalidating compilation and inference. Reuse preserves original pins instead of rewriting a result's provenance to the newest version.

Current Model state and historical results remain distinct. Invalidating a result neither deletes it nor automatically starts expensive computation. An explicit run or the default auto-run policy determines subsequent work.

The [Model reader](model-snapshot.md) selects a committed journal prefix and exposes the canonical definition alongside sourced inputs and findings. Current estimates require compatible fresh sources. Historical artifacts remain available at their original versions.

## Delegated Contexts

The outer operation owns machine inputs and outputs. Its delegated context owns prompts, allowed tools, and checkpoints. Context-local progress is not an additional artifact hierarchy.

Statistical authoring has the most involved inner workflow: independent ready constructs may run concurrently, feedback components remain sequential, accepted branches merge, and a shared full-model barrier must pass. Its [state-machine reference](../reference/statistical-model-spec/state-machine.md) owns those detailed rules. Partial admission checkpoints cannot become a public completed Model.

The journal preserves each operation's trace even after another operation updates the Model. The trace endpoint uses the operation identity; artifact inspection uses the artifact version.

## Transport

HTTP exposes the machine through generated Python-owned contracts. `PUT /api/episodes/{id}/model` accepts the expected version and complete candidate; read accessors select the same canonical value at `at_seq`. The Next.js layer forwards requests and owns presentation behavior.

The [offline migration](../guides/additive-model-migration.md) converts retained histories from the former separate scientific artifact schemas. The application accepts one runtime contract.
