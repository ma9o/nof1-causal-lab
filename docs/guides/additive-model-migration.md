# Converting Retained Model Histories

ModelSpec replaces the separately persisted latent, measurement, causal-design, statistical-specification, and fitted-model schemas. The running application accepts the canonical contract. Conversion is an explicit offline operation.

Use a source workspace that is no longer being written. Create a separate staging root and preserve the workspace directory name, because retained posterior and scenario identities include it. From `apps/data-pipeline`:

```bash
uv run python -m scripts.migrate_additive_model ../../data/WORKSPACE ../../.local/additive-migration/WORKSPACE
```

The destination must be absent and outside the source. A failed conversion removes its temporary output. The source remains unchanged.

## Preserved Information

| Retained information | Conversion |
|---|---|
| Scientific histories | Resolve each original source context into a canonical Model revision; preserve every explicit parent pin |
| Identification | Preserve positive and negative findings; restore reports that previously existed only inside causal designs |
| Journal | Preserve sequences, applied/rejected/raised outcomes, errors, and trace identities; move extraction worker outcomes here from the retired measurements artifact |
| Retractions | Remove obsolete findings and reveal a retained Model ancestor when a former detail layer was removed |
| Structural plans | Keep execution order and dispositions; remove copied semantic definitions |
| Compilers | Validate historical priors and anchors against ModelSpec, then discard receipts and their lineage pins; derive readiness, numerical blocks, and coordinates when needed |
| Posterior payloads | Preserve joint draws, latent paths, and observation times as a shared native law on a new ModelSpec revision; move engine evidence and diagnostics to the inference log |
| Reports without full samples | Retain report values and historical metadata in the log; do not invent a conditioned model or enable simulation |
| Source tables and traces | Copy unchanged |
| Authoring checkpoints | Archive their files; retire resumes that use the old submission schema |

`migration.json` records SHA-256 hashes for the original files, the scientific revision mapping, and restored identification reports. Conflicting source pins or ambiguous anonymous mechanisms fail conversion; the script does not infer missing historical decisions from current state.

## Converting the Former Parameter Catalogue

The [component-slot converter](../../apps/data-pipeline/scripts/migrate_component_slots.py) moves a retained flat parameter catalogue into construct-owned coefficient expressions and indicator-owned likelihood coefficients. It preserves parameter IDs and native laws, removes stored quantity/owner metadata and root policy flags, and places referenced parameter IDs directly in operand values. The full history converter above includes this step.

For standalone model JSON files already using the additive entity hierarchy, validate the conversion first, then write the selected files:

```bash
uv run python -m scripts.migrate_component_slots PATH/model.json
uv run python -m scripts.migrate_component_slots --write PATH/model.json
```

This command accepts the former catalogue schema. It is an offline conversion tool, not a runtime compatibility path. Rebuild derived fixture views after converting their source model.

Models with the former `innovation` and `initial_state` containers can use the [construct-coefficient converter](../../apps/data-pipeline/scripts/migrate_construct_coefficients.py). It moves their scalar uses into the shared [coefficient expressions](../pipeline/statistical-model-spec.md#construct-coefficients), preserving parameter IDs, values, noise families, and joint construct references:

```bash
uv run python -m scripts.migrate_construct_coefficients PATH/model.json
uv run python -m scripts.migrate_construct_coefficients --write PATH/model.json
```

Models already using expression operands with nested `FixedCoefficient` or `ParameterCoefficient` records can use the [operand-value converter](../../apps/data-pipeline/scripts/migrate_coefficient_values.py):

```bash
uv run python -m scripts.migrate_coefficient_values PATH/model.json
uv run python -m scripts.migrate_coefficient_values --write PATH/model.json
```

It preserves roles, fixed values including zero, parameter IDs, joint construct references, and unassigned operands. Current runtime schemas accept only the direct `value` field.

## Validation and Cutover

Point the local data root at the staging directory and run the [lineage validator](../../apps/data-pipeline/scripts/validate_run.py) with `--workspace-id WORKSPACE`. Inspect historical snapshots, retained result provenance, and the migration manifest before installing the converted store and journal.

Archive the original store and journal before replacement. Keep source data, existing fixture projections, and unrelated workspace files. The conversion does not run inference, simulations, evaluations, or paid services.

The normal API then uses `PUT /api/episodes/{id}/model` with an expected base version (`0` for a first Model) and a complete candidate value. Subsequent authoring operations enrich this same value through the [shared revision boundary](../design/model-snapshot.md#writes-and-operation-history).

## Connected graph payloads

The current [ModelSpec](../pipeline/latent-structure.md#modelspec) derives constructs
from edge endpoints. To convert a retained flat ModelSpec JSON file into a separate
file, run from `apps/data-pipeline`:

```bash
uv run python -m scripts.migrate_connected_graph source-model.json connected-model.json
```

The [converter](../../apps/data-pipeline/scripts/migrate_connected_graph.py) preserves
entity IDs and owned scientific values, defines each construct once at an endpoint,
and writes references for shared endpoints. It refuses an existing destination,
isolated constructs, undefined endpoints, and disconnected graphs. Those structures
require an explicit scientific revision; conversion does not invent causal edges.
This command converts a model value, not an episode journal or retained numerical
coordinate arrays.

## Retired construct usage

`ConstructUsage` is no longer an authored field. Exact measurements belong in an
indicator’s [Delta likelihood](../reference/statistical-model-spec/likelihoods.md);
[recording semantics](../pipeline/measurement-structure.md#source-recording-and-exactness)
determine how extraction handles empty windows. Constructs retain their initial
and transition laws. Admission scopes are operation-local projections.

Run the offline converter with complete reviewed construct replacements for every
active declaration:

```bash
cd apps/data-pipeline
uv run python -m scripts.migrate_construct_usage old-model.json new-model.json --revisions construct-revisions.json
```

The revisions file maps construct IDs to complete revised construct definitions.
The converter preserves the source file and rejects unreviewed active declarations.
Null declarations need no replacement. Removing a scientific-only exclusion may
expose required unsupported structure; the result remains a valid partial model
with explicit execution requirements.

## Scalar identity fields

Older JSON payloads wrap scenario targets, outcomes, and model default outcomes in
`ConstructRef` records, and wrap snapshot workspace IDs in `ModelRef` records.
Convert those files on a separate copy of a retained workspace:

```bash
cd apps/data-pipeline
uv run python -m scripts.migrate_identity_references ../../.local/migration/WORKSPACE/store/model/v1/model.json
```

The command reports changed files by default; add `--write` to convert the supplied
JSON files. Include every retained model version and any saved scenario, snapshot,
or validation-report JSON payloads. The converter preserves scientific IDs, model
version numbers, journal sequence references, and provenance pins. It leaves shared
graph endpoint references intact. Validation issues become indicator-specific or
dataset issues; construct-level explanations retain the construct ID in the message.
The running application accepts the current contract without compatibility parsing.

## Inline distribution laws

The [distribution-reference converter](../../apps/data-pipeline/scripts/migrate_distribution_references.py) moves inline parameter and trajectory laws into `ModelSpec.distributions`. Quantities retain only their law IDs. The converter preserves native constructors, shared joint laws, and the source file; the destination must be new.

```bash
cd apps/data-pipeline
uv run python -m scripts.migrate_distribution_references source-model.json converted-model.json
```

## Research question ownership

The [question converter](../../apps/data-pipeline/scripts/migrate_model_question.py) folds each retained `question` artifact into the same [ModelSpec history](../design/model-snapshot.md) as graph and statistical edits. Run it on an offline workspace with the current scientific schema, preserving the workspace directory name:

```bash
cd apps/data-pipeline
uv run python -m scripts.migrate_model_question ../../data/WORKSPACE ../../.local/question-migration/WORKSPACE
```

The destination must be new. Initial question writes become models with empty edges; later question writes preserve the existing scientific definition. Model version numbers and dependent pins are remapped together, while journal positions, retained findings, arrays, and source data remain intact. `model-question-migration.json` records the revision mapping. Rebuild any generated fixture snapshots from the converted values. The full historical stage-schema converter above includes this step.
