# Actions, Operations, and Artifacts

The navigator chooses among four [scientific actions](../reference/scientific-actions.md): `edit_model`, `prepare_data`, `fit`, and `simulate`. The durable machine applies their effects and records provenance. Optional authoring recipes compose proposals and checks. Their contracts are defined by the [action requests](../../apps/data-pipeline/src/nof1_causal_lab/actions/contracts.py), [machine graph](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py) and [context hierarchy](../../apps/data-pipeline/src/nof1_causal_lab/machine/hierarchy.py).

## One Scientific Definition

The canonical [ModelSpec](../pipeline/latent-structure.md#modelspec) gains detail while its scientific entities retain identity. Latent structure, measurement structure, and statistical specification describe authoring work on this value. Each operation submits a whole candidate through the same optimistic revision boundary.

| Entity | Owned detail |
|---|---|
| Construct | Indicators, intrinsic dynamics, and execution usage |
| Indicator | Likelihood and measurement choices |
| Causal edge | Additive mechanisms, each with a stable identity |
| Parameter | Quantity, owners, fixed value or current native NumPyro law, and supporting evidence |

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

## Internal Jobs and Optional Recipes

| Operation | Inputs | Main effects | Scientific definition |
|---|---|---|---|
| `raw_data` | Uploaded source files | Raw Arrow table with column descriptions | [Ingestion](../pipeline/ingestion.md) |
| `latent_structure` | Model containing its research question | Propose or revise constructs and edges on the Model | [Latent authoring](../pipeline/latent-structure.md) |
| `measurement_structure` | Raw data and Model | Add owned indicators and usage to the Model | [Measurement authoring](../pipeline/measurement-structure.md) |
| `measurements` | Raw data and Model | Extract observations and, when usable, a panel | [Extraction](../pipeline/extraction.md) |
| `statistical_model_spec` | Model, identification, panel, validation | Commit the completed Model and record prior-predictive results in its operation history | [Statistical authoring](../pipeline/statistical-model-spec.md) |
| `posterior` | Executable Model and panel | Joint posterior and fit diagnostics | [Inference](../pipeline/inference.md) |
| `simulate` | Selected Model, design and optional comparison panel | Durable generated arrays, measurements and optional certified causal result | [Simulation](../reference/scientific-actions.md#simulation-designs) |

The observational-study recipe uses the authoring order to find missing or stale work. Direct actions depend on their actual inputs: model editing can interleave all scientific choices, fitting requires an executable model and compatible observations, and simulation requires a model and design. Numerical causal effects additionally require positive identification and matching production-fit evidence.

Fitting records fit diagnostics without launching a predictive batch. Both ordinary prediction and [intervention simulation](../pipeline/analysis.md) use `simulate`; each result persists in its own journal record. The `analysis` context offers only read-only `get_model_info`.

## Derivations and Writes

| Artifact | Derivation inputs |
|---|---|
| `identification_report` | Model |
| `data_profile` | Panel |
| `validation_report` | Panel, Model and data profile |

The writable root is Model, which can begin with a research question and no edges. Scientific authoring operations use the same Model writer as the public HTTP API.

A Model write names its expected base version (`0` for creation), validates all ownership and references, and completes required derivations before publication. If validation or a final write fails, its partial versions are removed and the previous state stays current. Incomplete scientific choices remain valid during authoring; executable outputs appear only when their requirements are met.

Identification reports preserve negative findings. Missing prerequisites can retract a derived output; that retraction remains visible in history.

## Freshness and Provenance

Every result pins its actual input versions. Model consumers additionally record a fingerprint of the scientific input they used. A prior revision can preserve extraction and identification while invalidating compilation and inference. Reuse preserves original pins instead of rewriting a result's provenance to the newest version.

Current Model state and historical results remain distinct. Invalidating a result neither deletes it nor automatically starts expensive computation. An explicit action or requested recipe determines subsequent work. Model-only, data-only and compatibility checks refresh according to their separate dependency sets.

The [Model reader](model-snapshot.md) selects a committed journal prefix and exposes the canonical definition alongside sourced inputs and findings. Current estimates require compatible fresh sources. Historical artifacts remain available at their original versions.

## Delegated Contexts

The outer operation owns machine inputs and outputs. Its delegated context owns prompts, allowed tools, and checkpoints. Context-local progress is not an additional artifact hierarchy.

The optional statistical authoring recipe has the most involved inner workflow: independent ready constructs may run concurrently, feedback components remain sequential, accepted branches merge, and a shared full-model barrier must pass before the recipe succeeds. Its [state-machine reference](../reference/statistical-model-spec/state-machine.md) owns those rules. Direct model edits do not depend on recipe admissions; valid incomplete candidates remain savable.

The journal preserves each operation's trace even after another operation updates the Model. The trace endpoint uses the operation identity; artifact inspection uses the artifact version.

## Transport

HTTP exposes `POST /api/episodes/{id}/actions` and the equivalent `scientific` tools through generated Python-owned contracts. `GET .../revisions` lists selectable immutable inputs; comparisons report changes in fixed/free decisions, laws and supporting evidence. Lower-level `/moves` and model-write endpoints remain machine interfaces. The Next.js layer forwards requests and owns presentation behavior.

The [scientific-action migration](../reference/scientific-actions.md#optional-recipes-and-retained-workspaces) archives legacy predictive checks outside fit reports. The earlier [model migration](../guides/additive-model-migration.md) converts histories from separate scientific artifact schemas. These are explicit offline conversions; the application accepts one runtime contract.
