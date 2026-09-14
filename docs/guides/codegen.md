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
model checks and compilation, and inference and causal analysis. Internal groups
keep these readable, with measurement choices alongside dynamics, causal structure,
and parameters. References between groups remain visible without stretching their layouts.

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
Solid arrows represent ordinary field references. Role descriptions and folded
field details remain available in tooltips. These are display transformations;
the exported contracts retain their distinct types.

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

## Changing the schema

Workflow: **edit Python → `bun run codegen` → commit both**.

Type names describe their role. `ModelSpec` is the directly persisted scientific definition; `Construct`, `Indicator`, and `ParameterSpec` retain their canonical ownership inside it. Log reports use names such as `InferenceReport`. Tool responses use names such as
`SimulationResult` and `ToolError`. `ToolDefinition` describes a callable
tool, while `ArtifactPayload` supplies the shared validation base. Generated
TypeScript exports use the same names as Python. Server-composed views and machine records share this export. Frontend code owns presentation state only.

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
