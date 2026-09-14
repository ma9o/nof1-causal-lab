# Code Generation

Generated API artifacts and generated documentation have separate ownership and commands.

## API Artifacts

`bun run codegen` runs two independent export branches, then generates the model read client:

- **Contracts**: [`artifacts/catalog.py`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py) and the models owned by `artifacts/` are the source of truth. `export_schemas.py` writes the JSON schemas; `generate.ts` then feeds them through [`json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) to write the TypeScript models, tool definitions, and metadata.
- **Agent API**: `export_agent_api.py` writes the OpenAPI schema and the generated `nof1-episode-api` skill from the FastAPI application.
- **Model client**: [`generate-client.ts`](../../packages/api-types/scripts/generate-client.ts) uses [openapi-typescript](https://openapi-ts.dev/node) to generate request paths, parameters, and response types. Response declarations reference the existing domain types; [openapi-fetch](https://openapi-ts.dev/openapi-fetch/) supplies the runtime client.

Native NumPyro distributions use the shared [JSON codec](../../apps/data-pipeline/src/nof1_causal_lab/numpyro_json.py) on scientific parameters and compiled sites. Export derives constructor signatures from native distribution arguments and constraints. There is no separate prior-parameter class hierarchy.

```bash
bun run codegen       # regenerate API artifacts
bun run codegen:check # verify API artifact drift
```

Generated API files are committed. Run `codegen` after editing an artifact, read model, machine record, tool contract, or facade response.

The combined `contracts.json` includes all registered JSON artifact payloads, facade responses, machine records, and tool results. `panel` is a Parquet artifact whose file layout is declared in the generated metadata. OpenAPI remains the HTTP operation description; it is not a second source of domain types.

## Type Diagrams

`bun run types:graph` reads current Python schemas and writes a compact SVG and
Graphviz DOT file under `.local/type-system/`. Graphviz must provide `dot`.
Sections group types by concern; colors identify their roles.
One **Scientific model** section contains observed data, model specification,
model checks and compilation, and inference and causal analysis. Its types share
one layout without internal subsections, so their references determine placement.
References between top-level sections remain visible without stretching their layouts.

| Command | View | Output stem |
|---------|------|-------------|
| `bun run types:graph` | All exported backend contracts | `backend-types` |
| `bun run types:graph --view semantic` | `ModelSnapshot` and its dependencies | `semantic-model` |
| `bun run types:graph --view artifacts` | Stored JSON payloads and their dependencies | `stored-artifacts` |
| `bun run types:graph --view machine` | Machine and transport types, with external dependencies shown at the boundary | `machine-transport` |

Stored payload roots come from the schema references exported by
[`ARTIFACT_CONTRACTS`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py).
The machine view labels external types and leaves their fields unexpanded.

Compact diagrams put scalar IDs, validated scalar aliases, named enums, and provenance references into
field annotations. Generic sourced values become annotations or dashed arrows
through their `.value` fields; their `FactSource` provenance remains explicit.
Solid arrows represent ordinary field types. Green dotted arrows show references
by identity, resolved from nominal IDs to authored entities that declare those
IDs as their `id` field. Compact views fold simple entity-reference records into
these arrows while retaining their types and field paths in annotations and tooltips.
View selection follows both field and identity relationships. Role descriptions and folded
field details remain available in tooltips. These are display transformations;
the exported contracts and schema analysis retain their distinct types and field dependencies.

Add `--detail full` to retain every type in the selected view and show role
sentences inside nodes. Detailed output stems end in `-full`. Use `--root TypeName`
to follow a particular type instead of selecting a view, and `--output path/stem`
to choose the destination.

```bash
bun run types:graph --detail full
bun run types:graph --view semantic --detail full
bun run types:graph --root ParameterSpec
```

## Documentation Artifacts

`bun run docs:codegen` updates the generated distribution reference sections, then rewrites math in `README.md` and `docs/` (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) as SVG embeds under [`docs/assets/generated/latex`](../assets/generated/latex). The source is retained in nearby `docs-latex` metadata comments because GitHub math rendering is unreliable across Markdown contexts.

```bash
bun run docs:codegen # regenerate documentation artifacts
bun run docs:check   # verify documentation drift, Markdown, and spelling
```

`bun run check` runs both drift checks alongside the repository's lint, type, test, and build tasks.

## Type Naming Conventions

Name types for the role their instances serve. Python contracts and generated
TypeScript exports use the same names.

| Role | Convention | Examples |
|------|------------|----------|
| Declarative model or configuration definition, consumed by validation or compilation | `...Spec` | `ModelSpec`, `ParameterSpec`, `LikelihoodSpec`, `DynamicsSpec` |
| Symbolic formula or formula node | `...Expression` | `StateExpression`, `CoefficientExpression`, `BinaryExpression` |
| Compiled implementation or execution state | `Compiled...` or `...Runtime` | `CompiledDynamics`, `CompiledObservationModel`, `ObservationSupportRuntime` |
| Executable mathematical operation | Name the operation | `ObservationKernel`, `ObservationOperator`, `VectorField` |
| Validation, identification, or inference findings | `...Report` or a specific finding name | `IdentificationReport`, `InferenceReport`, `ValidationIssue` |
| Completed operation output | `...Result` | `SimulationResult`, `PriorPredictiveResult` |
| Recorded observation or event | `...Record` or `...Event` | `ObservationRecord`, `RuntimeEvent` |
| Persistent scalar identity | `...Id` | `ConstructId`, `IndicatorId`, `ParameterId` |
| Structured reference | `...Ref` | `ConstructRef`, `ArtifactRef`, `ParameterRef` |
| Exact model version | `...Revision` | `ModelRevision` |
| API request and its input values | `...Request` or `...Input` | `ScenarioRequest`, `ScenarioStartInput` |

The scientific definition types are `ModelSpec`, `ConstructSpec`, `CausalEdgeSpec`,
`IndicatorSpec`, `DynamicsMechanismSpec`, `LikelihoodSpec`, `ObservationLawSpec`,
and `ParameterSpec`. A spec may be partial during authoring, complete before
execution, or enriched with conditioned distributions after inference. Its suffix
describes its declarative role throughout those revisions.

Use the scientific nouns in prose and UI labels: "construct", "indicator", and
"observation law". Keep identity names and serialized field names tied to those
concepts, such as `ConstructId`, `ConstructRef`, `indicators`, and `law`.
The suffix belongs to the definition type's name.

Choose names by meaning rather than by the base class or whether an object is
serializable. An expression represents a formula; a runtime object supplies
execution operations; a report records findings. These roles keep their own names.
When renaming a public type, update its consumers and regenerate the API artifacts
without retaining compatibility aliases.

### Identity and revision conventions

Use a scalar ID when a field identifies one known kind of entity: scenario
`target` and `outcome`, model `default_outcome`, and validation `indicator_id`.
Keep tagged references for mixed entity kinds and for shared graph endpoints.
Python edges hold canonical `ConstructSpec` objects; their
[JSON representation](../design/additive-model.md#proposed-ownership) defines a
shared construct once and refers to it at subsequent endpoints.

Scientific IDs are nominal Python `NewType` values with Pydantic format
constraints. Construct IDs explicitly when allocating trusted identities; use
Pydantic model validation or `TypeAdapter` when accepting serialized input.
The `NewType` constructor alone does not validate a prefix or prove that an entity
exists. Resolve membership against the selected model. Named JSON Schema
definitions preserve the corresponding TypeScript ID types.

Entity identity survives renames and revisions. Exact provenance remains separate:
`ModelRevision` records a workspace and model artifact version; `ArtifactRef`
records an artifact and version; `TransitionRef` records a journal sequence in the
enclosing workspace. A journal sequence is not a model artifact version.
Snapshots carry `context.workspace_id` and the selected journal sequence, and pin
each sourced value to its supporting version. See
[model snapshots](../design/model-snapshot.md) for historical reads and freshness.

## Changing the schema

Workflow: **edit Python → `bun run codegen` → commit both**.

Follow the [type naming conventions](#type-naming-conventions). `ModelSpec` is the
directly persisted scientific definition; `ConstructSpec`, `IndicatorSpec`, and
`ParameterSpec` retain their canonical ownership inside it. `ToolDefinition`
describes a callable tool, while `ArtifactPayload` supplies the shared validation
base. Server-composed views and machine records share this export. Frontend code
owns presentation state only.

- **New/changed field**: edit the owning Python model.
- **New artifact contract**: add the payload class in `artifacts/`, register it in `ARTIFACT_CONTRACTS`, add re-export in `index.ts`.
- **New/changed tool**: update the owning transition’s `ToolDefinition` and its registration in `flows/context_tools.py`.

## File ownership

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated — do not edit |
| `src/generated/tools.ts` | Generated — do not edit |
| `src/generated/model-api.ts` | Generated model read operations referencing canonical domain declarations |
| `src/client.ts` | Runtime client factory using the generated operation signatures |
| `src/generated/metadata.ts` | Generated artifact IDs, file layout, machine description, and distribution metadata |
| `src/index.ts` | Hand-written re-exports |
| `src/run.ts`, `src/transitions.ts` | Hand-written |

## Client Version

`openapi-fetch` is pinned to 0.16.0 because [0.17.0's response mapping](https://github.com/openapi-ts/openapi-typescript/blob/main/packages/openapi-typescript-helpers/index.d.ts) widens fixed
tuples into arrays. The client type test checks that a fetched batch retains the
canonical `ModelSnapshot` contract, including posterior draw dimensions.

## Troubleshooting

- **Optional vs required mismatch**: `_make_defaults_required()` in the export script promotes defaulted fields to required, but nullable fields (`default=None`) stay optional.
- **Spurious named type aliases** (e.g. `type RHat = number`): `stripFieldTitles()` in `generate.ts` strips Pydantic's per-field `title` annotations that cause these.
- **Circular imports**: artifact contracts import other contracts; numerical implementations import those contracts. Keep the `artifacts` package initializer free of re-exports.
- **tanstack-table column errors**: cast generated column defs as `ColumnDef<T, unknown>[]`.
