# Model Access at a Committed Revision

`GET /api/episodes/{workspace_id}/model` returns a typed model view from one
committed journal position. `?at_seq=N` selects a particular applied move; zero
selects the empty model. Rejected and raised attempts remain activity and cannot
be selected as model revisions. The default is the latest applied move.

The reader loads the journal once and reads only the artifact versions selected
by that prefix. Historical facts therefore retain their historical contents and
freshness, even after later writes or derivation retractions. The endpoint works
on the read-only facade and does not require a compiled or fitted model.

## Identity and Ownership

Identity starts in the authored [constructs and edges](../pipeline/latent-structure.md#construct).
[Indicators](../pipeline/measurement-structure.md#indicator) carry their owner's
persistent construct ID. IDs are required and use a kind prefix plus an opaque
token. Preserve an ID when editing the same entity; assign a new ID for a new
entity. Names, ordering, and lag changes do not allocate new identities.

The compiler carries authored IDs into the
[structural catalog](../pipeline/measurement-structure.md#structuralplan).
Constructs and edges belong to the model; indicators belong to a construct.
Artifact producers validate these relationships; the snapshot checks its source-version references.
An older, stale indicator definition can retain its owner through a rename. If
that owner was removed, the indicator belongs to the earlier revision and is
omitted from the current graph; its artifact and historical snapshot remain
readable.

This is a required contract change. Existing payloads without IDs are invalid;
the checked-in examples and DEMO fixtures have been updated explicitly.

## Read Contract

`ModelSnapshot` contains revision context and independently sourced aggregate reads.
It reuses the existing domain hierarchy, including the specification's mechanisms
and the posterior's nested assessment. It has no enriched entity subclasses or
public indexes duplicating those aggregates.

| Field | Meaning |
|---|---|
| `model`, `seq`, `state` | Workspace identity, committed journal position, and selected artifact versions |
| `question` | Canonical question artifact |
| `latent_structure` | Canonical constructs, edges, and default outcome |
| `measurement_structure` | Canonical measurement artifact: indicators, clock, known inputs, and scientific-only declarations |
| `identification` | Canonical identification status from the selected causal design |
| `dispositions` | Canonical disposition collection from the selected structural plan |
| `graph_status` | Python-derived construct status across identification and structural dispositions |
| `raw_data`, `measurements` | Server-composed uploaded-table profile and extraction view |
| `validation_report` | Canonical validation aggregate with indicator audits and dataset issues |
| `specification` | Canonical specification artifact: model mechanisms, likelihoods, authored parameter definitions, priors, and admission findings |
| `compiled_parameters` | Scientific parameter catalog from the selected compiler |
| `fit` | Canonical `PosteriorArtifact`, predictive-check counts, and graph estimates |
| `baseline_report` | Canonical treatment effects and narrative aggregate |
| `saved_scenarios` | Independently versioned canonical scenario collection |
| `artifacts`, `installed_at`, `retracted` | Freshness and journal history for version navigation |

Each optional aggregate has one `Sourced` boundary carrying its artifact version,
JSON pointer, and freshness. For example, the default outcome is
`latent_structure.value.default_outcome`; sampler diagnostics are
`fit.value.posterior.assessment.mcmc_diagnostics`. `FitSummary` adds only display
findings computed by Python to the canonical posterior.

Latent and measurement reads remain separate because either can be missing or
have a different freshness status. Identification reads the selected causal
design's existing `IdentifiabilityStatus`; it does not construct a new
`CausalDesign` from independently selected versions. The structural plan's
execution topology and duplicate semantic catalog stay in artifact inspection.
Its disposition collection is enough for model scopes.

The authored parameter catalog belongs to `specification`; the compiled catalog
belongs to `compiled_parameters`. These retain their distinct provenance. An
uncompiled model still exposes its authored definitions. Parameter tables use
the compiled catalog and compatible resolved prior and posterior findings.

[`ModelReader`](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshots.py)
provides aggregate and collection accessors at the same pinned revision:

| GET endpoint below `/api/episodes/{workspace_id}/model` | Response |
|---|---|
| `/latent-structure` | Optional `Sourced[LatentStructure]` |
| `/measurement-structure` | Optional `Sourced[MeasurementStructureArtifact]` |
| `/specification` | Optional `Sourced[StatisticalModelSpecArtifact]` |
| `/posterior` | Optional `Sourced[PosteriorArtifact]` |
| `/constructs` | `Construct[]` |
| `/edges` | `CausalEdge[]` |
| `/indicators` | `Indicator[]` with surviving owners |
| `/parameters` | `ParameterSpec[]` |

Each accessor accepts `at_seq` and reads its defining artifacts and required
ownership inputs. It does not materialize the batch or table views. For several
reads, select a revision once and pass its `seq` to each request. The model page
uses the batch endpoint so graph and detail panels share one request.

Names live on the definitions that own them. Edges, indicator ownership, known
inputs, and scientific-only declarations reference persistent IDs. Renaming one
construct changes only that definition. The graph uses the same IDs for node
selection, temporal copies, and edge slots.

Fact sources carry `validity: "fresh" | "stale"`. The
[identification aggregate](../pipeline/measurement-structure.md#identifiabilitystatus)
retains persistent IDs for treatments, confounders, and instruments.

Observations, likelihoods, validation issues, and predictive findings carry explicit
indicator or construct IDs. Parameter definitions carry a quantity, scientific
owners, and a parameter ID. Priors reference that ID; scalar posterior findings
reference both the parameter ID and a logical element ID. Display names do not
establish relationships.

The compiler maps each logical element to a `ParameterCoordinate(site_name, indices)`.
It declares category contrasts and ordinal cutpoints from their category labels,
and marks padded tensor coordinates as execution-only. A scalar parameter's
identity survives a runtime-axis reorder. A Cholesky component includes its
ordered basis because changing that basis changes the represented quantity.
Unknown owners, overlapping bindings, and unbound scientific findings are errors
at production. The snapshot joins IDs directly and never searches historical
catalogs or parses runtime labels to recover ownership.

Structural findings exist only while their selected structural plan exists.
Posterior parameter findings attach only when the posterior's `compiled_ssm`
input pin matches the selected compiler. Resolved priors attach only when the
compiler's specification pin matches the selected specification. These existing artifact pins provide revision provenance;
individual parameter references do not repeat it.

Graph annotations use the same `PosteriorEstimate` fields as
[posterior marginals](../pipeline/inference.md#posteriormarginal): mean, interval
bounds, interval kind, and probability mass.

The parameter definition also records the authoring transform. A continuous-time
decay posterior remains a decay rate; its mean must not be transformed and
presented as a daily-persistence posterior.

Artifact views are computed in Python. Uploaded-table profiles, extraction
counts, empirical histograms, and predictive comparisons use immutable table
versions. A panel and its worker outcomes must come from the same applied move.
The measurement/design view and likelihood diagnostics require matching input
pins. Incompatible joins are absent while their individual artifacts remain
available for inspection. The model page fetches the semantic snapshot when the selected revision changes.
Artifact inspection uses `GET /api/episodes/{workspace_id}/model/views/{artifact_id}`
separately. Aggregate reads retain canonical types while projecting compatible
members: removed indicator owners are omitted, incompatible resolved priors and
posterior coordinates are absent, and historical fit metadata and predictive
checks remain available. The original immutable payload remains inspectable.
Provenance points to the supporting aggregate, rather than to an array position
in the filtered read.

Saved scenarios use a structured [query contract](../pipeline/analysis.md#scenarioquery)
whose identity includes the model, posterior version, persistent intervention
targets, resolved start, and requested readout. Display labels do not affect its
identity. Saving a historical result preserves its original posterior version;
a saved collection can contain queries from several fits. Simulation overlays
are shown only against the fit that produced the query. Trajectory maps use
persistent construct IDs, so a label change does not detach a saved overlay.

## Generated Types and Diagram

The Pydantic models in
[`snapshot_models.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/snapshot_models.py)
are exported through the existing [API code-generation pipeline](../guides/codegen.md).
The generated TypeScript reuses the canonical definitions and owner-reference
types. Backend validators additionally enforce reference integrity and freshness.
The [OpenAPI client generator](../../packages/api-types/scripts/generate-client.ts)
generates operation signatures and links response schemas to those existing
declarations. The React Query hook uses `createModelClient`; the Next.js route
forwards the Python response body, status, and revision query unchanged.

```bash
bun run types:graph
bun run types:graph --root ModelSnapshot
```

The script reads current Python schemas for API-exported contracts and their
nested types, then writes standalone SVG and Graphviz DOT files under
`.local/type-system/`. Install Graphviz to provide `dot`.
Each node displays the type name and the opening role paragraph from its Python
description. Hovering reveals the full description. Every exported type must
have a description beginning with a complete role sentence; diagram generation
rejects missing descriptions. Every type declares one Python owner and one of
seven conceptual layers: identity, authored values, artifact payloads, derived
findings, machine records, read models, or transport. The diagram includes a
color legend for these layers.

Arrows represent actual schema references, including persistent ID aliases,
containers, and union branches. Recursive JSON values retain their cycles in
this type graph. These are data dependencies; the scientific causal structures
remain DAGs with explicit latent confounders.
