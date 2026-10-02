# Code Generation

Generated API artifacts and generated documentation have separate ownership and commands.

## API Artifacts

`bun run codegen` exports the Python contracts and API together, then generates the TypeScript types and facade client:

- **Contracts**: [`artifacts/catalog.py`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py) and the domain owners in `artifacts/` and [`study/records.py`](../../apps/data-pipeline/src/nof1_causal_lab/study/records.py) are the source of truth. [`export_api.py`](../../apps/data-pipeline/scripts/codegen/export_api.py) writes the JSON schemas; `generate.ts` then feeds them through [`json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) to write the TypeScript models and metadata.
- **Agent API**: The same exporter writes the OpenAPI schema and the generated `nof1-study-api` skill from the FastAPI application.
- **Facade client**: [`generate-client.ts`](../../packages/api-types/scripts/generate-client.ts) uses [openapi-typescript](https://openapi-ts.dev/node) to generate every exported OpenAPI operation, including actions, polling, traces and multipart uploads. Response declarations reference the existing domain types; [openapi-fetch](https://openapi-ts.dev/openapi-fetch/) supplies the runtime client. Upload inputs use native `Blob` values, serialized as `FormData` by the caller.

Native NumPyro distributions use the shared [JSON codec](../../apps/data-pipeline/src/nof1_causal_lab/numpyro_json.py) on scientific parameters and compiled sites. Export derives constructor signatures from native distribution arguments and constraints. There is no separate prior-parameter class hierarchy.

```bash
bun run codegen       # regenerate API artifacts
bun run codegen:check # verify API artifact drift
```

Generated API files are committed. Run `codegen` after editing an artifact, read model, study record, tool contract, or facade response.

The combined `contracts.json` includes all registered JSON artifact payloads, facade responses, study records, and tool results. `panel` is a Parquet artifact whose file layout is declared in the generated metadata. OpenAPI remains the HTTP operation description; it is not a second source of domain types.

## Documentation Artifacts

`bun run docs:codegen` updates the generated distribution reference sections, then rewrites math in `README.md` and `docs/` (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) as SVG embeds under [`docs/assets/generated/latex`](../assets/generated/latex). The source is retained in nearby `docs-latex` metadata comments because GitHub math rendering is unreliable across Markdown contexts.

```bash
bun run docs:codegen # regenerate documentation artifacts
bun run docs:check   # verify documentation drift, Markdown, and spelling
```

`bun run check` runs both drift checks alongside the repository's lint, type, test, and build tasks.

### Action-flow charts

Each chart in [`docs/assets/action-flows`](../assets/action-flows) is a native Excalidraw scene, and the scene is the source. After editing a scene, export its SVG, then render and inspect the result. Use `edit_model` as the visual reference and keep its individual check nodes.

Keep the shared styling: 16 px sans-serif body text, monospace titles and outcomes, solid 2 px outlines, rounded stacked cards, slate connectors and pale gray section panels. Requests are indigo, warnings amber, failures red and commits green. Skipped or unavailable results and subsequent actions use dashed gray cards.

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
| Completed operation output | `...Result` | `CausalEffectResult`, `PriorPredictiveResult` |
| Recorded observation or event | `...Record` or `...Event` | `ObservationRecord`, `RuntimeEvent` |
| Persistent scalar identity | `...Id` | `ConstructId`, `IndicatorId`, `ParameterId` |
| Structured reference | `...Ref` | `ConstructRef`, `GitRef`, `ParameterRef` |
| Exact study version | `...Revision` | `StudyRevision` |
| API request and its input values | `...Request` or `...Input` | `SimulateRequest`, `SimulationSpec` |

The scientific definition types are `ModelSpec`, `ConstructSpec`, `CausalEdgeSpec`,
`IndicatorSpec`, `DriftMechanismSpec`, `PotentialMechanismSpec`, `LikelihoodSpec`,
`ObservationLawSpec` (the closed union of per-family law specifications),
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
Python edges hold canonical `ConstructSpec` objects; their JSON representation
defines a shared construct once and refers to it at subsequent endpoints.

Scientific IDs are nominal Python string subclasses whose constructors and
Pydantic schemas enforce the same grammar. Construct IDs explicitly when
allocating identities; parse serialized input at its owner or external boundary.
The constructor proves the ID grammar. Resolve entity membership against the
selected model. Named JSON Schema
definitions preserve the corresponding TypeScript ID types.

Entity identity survives renames and revisions. Exact provenance remains separate:
`GitRef` records a workspace, a Git object ID and a path within that object;
`StudyRevision` records one commit of the study history with its action log. A
commit is not a model artifact version.
Snapshots carry `context.workspace_id` and the selected journal sequence, and pin
each sourced value to its supporting version.

## Changing the schema

Workflow: **edit Python → `bun run codegen` → commit both**.

Follow the [type naming conventions](#type-naming-conventions). `ModelSpec` is the
directly persisted scientific definition; `ConstructSpec`, `IndicatorSpec`, and
`ParameterSpec` retain their canonical ownership inside it. `ToolDefinition`
describes a callable tool, while [`Value`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/base.py) supplies the shared immutable
base. Server-composed views and study records share this export. Frontend code
owns presentation state only.

Owned values expose tuples and read-only mappings. Generated outputs have
readonly properties, arrays and tuples; an ID-keyed map preserves its named key
type as `Readonly<Partial<Record<Id, T>>>`. A lookup may be absent. A serialized
defaulted field is required, including a nullable field emitted as `null`.
Validation schemas describe inputs independently, so an input default may be
omitted. Fields excluded from serialization do not appear in output schemas.

[Assessments](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/checks.py) carry a producer's typed subject and evidence, or its explicit reason for unavailable evaluation. Consume those alternatives directly. Scientific classifications and plot series come from the backend. [Inference reports](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/posterior.py) compose a compact core with full plot detail; snapshot fields declare the core type, so serialization omits detail without filtering or reparsing owned values.

- **New/changed field**: edit the owning Python model.
- **New artifact contract**: add the payload class in `artifacts/`, register it in `ARTIFACT_CONTRACTS`, add re-export in `index.ts`.
- **New/changed tool**: update its `ToolDefinition` and its registration in [`tool_contracts.py`](../../apps/data-pipeline/src/nof1_causal_lab/tool_contracts.py).

## File ownership

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated — do not edit |
| `src/generated/model-api.ts` | Every OpenAPI operation, referencing canonical domain declarations |
| `src/client.ts` | Runtime client factory using the generated operation signatures |
| `src/generated/metadata.ts` | Generated artifact IDs, file layout, and distribution metadata |
| `src/index.ts` | Hand-written re-exports |

## Client Version

`openapi-fetch` is pinned to 0.16.0 because [0.17.0's response mapping](https://github.com/openapi-ts/openapi-typescript/blob/main/packages/openapi-typescript-helpers/index.d.ts) widens fixed
tuples into arrays. The client type test checks that a fetched batch retains the
canonical `ModelSnapshot` contract, including posterior draw dimensions.

The [client type test](../../packages/api-types/src/model-snapshot.type-test.ts)
also compares all generated paths and methods with the exported OpenAPI schema.
Web and API consumers enable `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes`: sparse lookups retain absence, and omitted request
options stay absent rather than being assigned `undefined`.

[Fixture generation](../../apps/data-pipeline/scripts/fixtures/study.py) emits
`.d.json.ts` declarations alongside canonical JSON projections. They reference
the production domain types without copying payloads or declaring another
schema. `fixture:build` and `fixture:check` own both outputs. Stored trace and
predictive-check bytes pass through their Python contract owners before export;
the declarations preserve discriminators and scientific IDs in JSON imports.

## Troubleshooting

- **Optional vs required mismatch**: `Value` owns serialization presence through Pydantic configuration; check the field's validation and serialization schemas separately.
- **Spurious named type aliases** (e.g. `type RHat = number`): `stripFieldTitles()` in `generate.ts` strips Pydantic's per-field `title` annotations that cause these.
- **Circular imports**: artifact contracts import other contracts; numerical implementations import those contracts. Keep the `artifacts` package initializer free of re-exports.
