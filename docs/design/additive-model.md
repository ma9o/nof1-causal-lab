# One Model, Built Incrementally

Status: approved for implementation. One model accumulates scientific knowledge,
and latent structure, measurement structure, and statistical specification exist
only conceptually. The type sketch shows ownership, rather than final class names
or constructor signatures. Detailed mappings are completed with each implementation
increment under the contracts below.

## Recommendation

Use one canonical `Model` from the first construct declaration through a complete
scientific specification. Enrich its existing entities as decisions are made:
constructs acquire measurements and dynamics, indicators acquire likelihoods,
edges acquire mechanisms, and parameters acquire priors.

Pipeline stages become operations on this model. Latent structure, measurement
structure, and statistical specification remain vocabulary for explaining aspects
of it and the work being performed. Accessors return its canonical entities and
their relationships. Each accepted operation creates an immutable revision of the
same model type.

This goes further than putting the current structures under one parent. Their
overlapping catalogs and ownership references disappear. The model owns the
scientific definition; numerical programs, observations, identification findings,
and fits retain their own input provenance.

## Required Architectural Boundary

`LatentStructure`, `MeasurementStructure`, `StatisticalModelSpec`, and the composed
`CausalDesign` cease to be domain types and separately persisted scientific
definitions. Their artifact wrappers, API schemas, and reconstruction adapters
are removed with them. Renamed equivalents, aliases, stage subclasses, or three
corresponding fields under `Model` would preserve the separation this refactor is
intended to remove.

The code expresses scientific entities, their owned components, relationships,
and operations. Documentation, workflow labels, and UI explanations can still
discuss causal assumptions, measurement design, and statistical specification.
Those concepts do not determine the shape of the stored model or require their
own Python or TypeScript value objects.

This boundary is an acceptance criterion for the refactor: the generated domain
graph must have one scientific model root and its entity hierarchy. Separately
typed numerical programs and sourced findings represent computation outputs.

## Proposed Ownership

```text
Model
├── constructs: Construct[]
│   ├── id, name, description, role, temporal_status
│   ├── indicators: Indicator[]
│   │   ├── id, name, measurement recipe, categories, ...
│   │   └── likelihood?                 family, link, standardization, evidence
│   ├── dynamics: NodeMechanism[]       intrinsic drift and coefficient references
│   └── usage?                         known input or scientific-only declaration
├── edges: CausalEdge[]
│   ├── id, cause_id, effect_id, lag, evidence
│   └── mechanisms: EdgeMechanism[]     additive edge functions, coefficient references
├── parameters: ParameterSpec[]
│   ├── id, quantity, owners, name, description
│   └── prior?                         native law, elicitation context, evidence
├── measurement_clock?
├── default_outcome?
└── policies                           initialization and observation-intercept choices
```

An indicator has one owning construct under the current domain rules. Its
containment establishes that relationship, so a stored `construct_id` on the
indicator is redundant. Likewise, a nested likelihood does not repeat its
indicator ID; a nested node or edge mechanism does not repeat its containing
entity's ID. Collections use each entity's existing ID, with private indexes for
lookup. The proposed shape reuses the values already defined by
[`Construct`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/latent_structure.py),
[`Indicator`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/measurement_structure.py),
and the [mechanism variants](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/mechanism.py).

`edge.mechanisms` is an additive collection, preserving the flexibility of the
current [mechanism lowering](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/mechanisms.py).
Linear and Hill functions are individual term kinds and can contribute together
on an edge. The native [vector field](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/dynamics/vector_field.py)
already accumulates component contributions. Attaching terms to their owning edge
must preserve that composition, every coefficient's fixed or estimated meaning,
and its scientific parameter references.

Node dynamics remain an additive collection as well. Preserve collection order in
serialization and lower all terms deterministically. Validation checks each term
and its references without imposing a new one-term-per-edge restriction. An empty
collection can represent an incomplete model; compilation still requires complete
coverage of the retained dynamic states and edges. Existing coefficient identity
rules and supported execution capabilities remain in force.

References across owners remain explicit. Edges reference their endpoints;
estimated coefficients reference parameters; shared likelihood parameters can
reference several indicators. Parameters therefore retain a model-wide registry
instead of being copied into every coefficient slot. `construct.parameters()`
and `model.indicator(id)` resolve those relationships as accessors.

`usage` records an authored known-input or scientific-only choice. Whether a
latent confounder is projected out of the executable state remains a derived
decision. Structural assumptions continue to be DAGs with explicit latent
confounders; this proposal does not introduce authored covariance edges.

Model-wide policies stay at the root because their scope is model-wide. The
measurement clock remains distinct from the numerical integration step.

The proposed ownership rules for the shared and exceptional cases are:

| Case | Rule |
|---|---|
| Indicator | Exactly one containing construct per revision; likelihood and measurement recipe live on that indicator. |
| Edge functions | The edge owns an additive `mechanisms` collection. Each term retains its kind and coefficients; several supported terms can contribute to the same edge. |
| Known input | The construct owns the declaration. Its source indicator must belong to that construct. The declaration contains the scale and missing-data policy, and is mutually exclusive with a scientific-only declaration. |
| Latent confounder | An ordinary construct with explicit causal edges. Executable projection is a derived decision. |
| Shared parameter | One entry in `model.parameters`; every use references its ID. Its owner set records all the scientific entities it belongs to. Existing family-sharing rules remain in force. |
| Parameter prior | The parameter owns the native law, elicitation context, and supporting evidence. The compiler owns only the derived law attachment and runtime coordinates. |
| Observations and results | Separate versioned inputs and outputs, related to the model through provenance. They do not become mutable attributes of scientific entities. |

Keep entity IDs through renames. A prior change keeps its parameter ID because
the quantity and owner set are unchanged. Changing a parameter's quantity or owner
set creates a different parameter ID under the existing identity rule. Old
posterior components cannot attach to that new identity merely because its label
matches.

## How the Same Object Grows

For a model involving stress and sleep, successive revisions could contain:

| Revision | Contribution to the same `Model` |
|---|---|
| 1 | Stress and sleep constructs, a directed causal edge, and the default outcome |
| 2 | Indicators attached to those constructs, extraction recipes, and a measurement clock |
| 3 | Likelihoods on the indicators, additive dynamics on the constructs, and additive mechanisms on the edge |
| 4 | The complete parameter inventory and explicit priors, including scientific defaults |

These are examples of contributions, not four mandatory global states. An
operation may specify a mechanism and its prior together. Another construct may
still need a likelihood. Every revision is a `Model`; there is no sequence of
`LatentModel`, `MeasuredModel`, and `StatisticalModel` subclasses.

An incomplete model can be valid. Every present entity and reference must be
consistent, while missing scientific choices remain explicit. Readiness for an
operation is computed from the model and that operation's other inputs:

- Extraction requires the relevant measurement definitions and source data.
- Identification requires the relevant causal assumptions and query.
- Compilation requires complete executable choices and priors for its scientific
  parameters.
- Fitting additionally requires compatible observations and the applicable checks.

These checks report missing requirements; they do not populate a second family
of readiness models or silently supply scientific choices. A successful check
allows the operation to use the same model.

Construction is additive while assumptions stay fixed. Corrections and deletions
create a new revision. An edit that leaves dangling edges, coefficient references,
or invalid likelihoods is rejected until the same update includes the necessary
repairs. Earlier consistent revisions remain available.

### Update and Validation Contract

An authoring operation builds a candidate `Model` from a named base revision.
Validate the complete candidate before committing it. The operation may add or
replace several components together; intermediate drafts do not become current.

| Boundary | Checks |
|---|---|
| Component construction | Intrinsic value constraints, native distribution arguments, and the internal consistency of a present likelihood or mechanism |
| Model commit | Unique IDs, valid containment and references, DAG constraints, compatible parameter quantities and owners, and all relationships whose inputs are present |
| Requested operation | Missing choices and inputs, complete parameter inventory, executable capability, and the scientific checks required for that operation |

For example, deleting a construct removes its contained indicators and dynamics
from the candidate. The update must also remove or repair incident edges, parameter
owners, the default outcome, and references from remaining components. Otherwise
the commit fails. Changing indicator categories must similarly replace or clear
any incompatible attached choices in the same update. Missing choices can then be
filled by later operations.

Validation enforces the encoded invariants. Scientific identification and
admission remain explicit findings; a structurally valid model is not itself
evidence that a causal claim is supported. Scientific defaults enter through an
explicit completion operation that records their origin. Validation and numerical
compilation do not silently supply them.

## Parameters and Native Probability Laws

The current compiler can add initial-state and shared likelihood parameters in
[`bind_parameters`](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/prior_compilation.py).
It also copies parameter definitions to add scalar elements and replace their
transform metadata. This prevents the authored catalog from describing the
complete model.

Move scientific parameter completion into the operation that specifies the
model. It must declare the complete inventory and any default priors before
committing a model ready for compilation. Use one shared inventory
planner based on the existing native block descriptors, likelihood membership,
anchor choices, and initialization policies. Compilation binds that inventory
and rejects omissions; it does not invent further scientific parameters.

Defaults retain their existing laws and are identified as policy choices in
their evidence. Shared likelihood parameters keep their current sharing rules.
Finalizing their owner set happens before committing the completed model.

Each parameter owns its native NumPyro prior and the context needed to interpret
that prior. The [native distribution boundary](../../apps/data-pipeline/src/nof1_causal_lab/numpyro_json.py)
already provides the direction for serializing laws. We should use native
constructors, constraints, and transforms rather than introduce another
distribution-parameter type hierarchy.

Scientific coordinates still need an explicit contract. A continuous-time decay
rate is positive, while an elicited interval-persistence prior has support between
zero and one. Its reference interval is scientific information. Keep the prior's
elicitation coordinate and interval together; derive the exact native transform
to the model coordinate. Compiler attachment choices such as a site row belong
to the binding, not to that elicitation contract. Parameter role and support
descriptions should be derived from the quantity and native contracts where they
are redundant.

Compiled bindings own logical element IDs, element labels, and runtime
coordinates. They reference the canonical parameter ID instead of copying
`ParameterSpec`. Execution-only padding stays outside the scientific inventory.
Preserve the [existing identity rules](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/compile/parameter_identity.py):
renames do not change identity, semantic components survive execution-axis
reordering, and changing a covariance basis can change the represented component.

## Numerical Programs and Results

`model.compile()` derives an executable program from the complete scientific
definition. This program owns array layout, sample sites, numerical blocks, and
bindings back to scientific IDs. It is a compiler output, with no independently
authored construct, indicator, or parameter catalog.

The completed model already declares the scientific quantities, all mechanism
terms and their coefficients, prior laws, and scientific defaults. Compilation
lowers these declarations into execution details: component layout, state and
channel axes, sample-site attachment, exact prior transforms, and tensor
coordinates. It preserves every additive contribution and does not create new
scientific parameter identities, choose undeclared scientific defaults, or mutate
the model to complete it. Execution-only padding remains explicitly separate from
scientific quantities.

The current [`SSMSpec`](../../apps/data-pipeline/src/nof1_causal_lab/models/ssm/model.py)
already has useful native blocks. Group its remaining channel and input metadata
with the blocks that consume it, and derive numerical arrays at execution
boundaries. Its serialization should follow that canonical native structure.
The scientific `Model` does not need to imitate every runtime field.

Identification, validation, posterior diagnostics, and causal estimates are
findings about a model revision and its inputs. Keep them as results with that
provenance. Identification must retain both positive and negative findings when
removing `CausalDesign.identifiability` as their storage location.

One model reader can expose the definition and these results together. For
example, `reader.model.construct(id)` returns the canonical entity, while
`reader.posterior(parameter_id)` resolves a compatible finding. The batch API
contains the canonical model plus data, findings, and revision context. The
frontend receives generated contracts and server-derived scientific facts.

The inference algorithms, exact nonlinear drift and emission paths, prior laws,
and scientific admission requirements remain part of the preservation contract.
This design changes ownership and compilation boundaries, not the target model.

## Revisions and Dependencies

One domain object does require revising the current
[artifact dependency graph](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py).
Scientific authoring operations would commit a new `Model` version instead of
installing separate latent, measurement, and statistical catalogs. Keep the
[journal's existing commit rule](storage-lifecycle.md#durable-ledger): immutable
content is written first, and an applied journal record selects it.

Operation identity must be independent of output identity. Currently,
[`Transition.transition_id`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py)
is its produced artifact ID. The new operation registry can have several
authoring operations that all write `model`. Their prerequisites inspect the model
and external inputs; they do not require separate scientific catalogs as evidence
that a previous stage has run.

Store full immutable values at `store/model/vN/`. A scientific edit creates a
model version; recording a result or an activity can advance the journal without
creating another model version. A model revision is identified by its workspace
and model artifact version. `at_seq` continues to select the complete historical
workspace view, including the model version selected by that journal prefix.

An authoring write names its expected base model version. Compare that version
and commit the validated replacement inside the existing serialized move
boundary. Reject a stale base with a conflict; do not merge two full values
automatically. The journal retains the operation and its provenance. Historical
values and their differences provide attribution without a second catalog of
scientific facts.

Derived outputs must not depend indiscriminately on every field of the root.
Otherwise adding a prior would invalidate measurement extraction, and completing
the statistical specification would invalidate the structural plan it just used.

Use explicit projections of canonical model fields for dependency boundaries:

| Consumer | Model facts it depends on, alongside its external inputs |
|---|---|
| Extraction | Indicator recipes, declared categories, and measurement clock |
| Identification | Causal assumptions and the requested effect |
| Execution planning | Causal and measurement structure, authored usage choices |
| Compilation | Executable structure, mechanisms, likelihoods, parameters, priors, and policies |
| Inference and reporting | Compatible compiled program, observations, and required findings |

These are computed input views over canonical components or native execution
values. They do not recreate stage-shaped domain types, even as private classes.
Each result records its exact source model revision and the content identity of
the inputs actually passed to its computation. The same input projection supplies
the computation and its dependency fingerprint, avoiding a separate hand-maintained
list of fields for freshness. A result can remain applicable to a later revision
when those inputs are unchanged, while still displaying its original provenance.
Computation versions and external artifact pins remain part of cache identity.

Begin with the existing dependency granularity. For example, changing a prior
invalidates compilation and inference while preserving extraction. Finer reuse
for presentation-only edits can follow explicit evidence that it is useful.
There is no need for a generic system that tracks arbitrary Python field reads.

| Edit or event | Required result behavior |
|---|---|
| Change a parameter's prior | Retain extraction and causal identification when their inputs are unchanged; invalidate affected compilation and inference. |
| Change a causal edge while keeping indicator recipes and clock | Retain extraction; recompute the affected identification, execution planning, and downstream results. |
| Rename an entity | Preserve identity. Reuse is determined by actual consumed inputs, including labels if a computation used them. |
| Change a shared parameter's owner set | Allocate its new scientific identity; old posterior components remain attached to the earlier parameter. |
| A fit finishes after another model edit | Store it against its original model and data inputs. It is applicable to the current view only if those computation inputs still match. |

Invalidation makes a result unavailable as a current claim; it does not delete the
historical output or automatically launch another expensive computation.

## Storage and API Transition

Use the existing HTTP and generated-client approach. The proposed public contract
is a model read at a journal position and an atomic model write:

| Surface | Contract |
|---|---|
| `GET /api/episodes/{id}/model?at_seq=N` | Return the selected canonical model, its model version, the selected journal position, and sourced data and findings. Before the first model commit, the model is absent. |
| `PUT /api/episodes/{id}/model` | Accept the expected base model version and a candidate `Model`; use a null base only for initial creation. Return the committed revision, a validation error, or a stale-base conflict. |
| Authoring operations | Build candidate values using canonical components and submit through the same commit path. Several operations can update the model. |
| Entity reads | Resolve constructs, indicators, edges, and parameters from that same selected model. They do not return reconstructed stage structures. |

The response envelope owns revision context and result provenance. Python defines
the canonical scientific schema; generated TypeScript uses it directly. The
Next.js layer forwards the contract. Full model writes keep the initial protocol
small; a generic patch language or separate stage-update schema hierarchy is
unnecessary.

Switch readers, writers, generated clients, and fixtures together to the new
schema. Remove the old scientific artifact IDs, wrappers, endpoints, and
composition adapters. Operation labels may retain conceptual stage names in
history and workflow explanations.

Preserve retained workspaces through an explicit offline conversion into a
separate destination. Keep the originals unchanged. Replay historical journal
positions and respect each artifact's pinned inputs when constructing the new
model revisions and result references. Do not combine incompatible historical
fragments into a supposedly coherent model or invent missing scientific choices.
An unmappable revision is a conversion error to resolve before switching that
workspace.

Compare entity IDs, scientific values, native prior laws, result bindings, and
historical reader behavior before cutover. Existing numerical outputs are carried
forward with their source context; conversion does not fit models again.
Development fixtures can be regenerated explicitly. The application has one
runtime schema, with no compatibility aliases or old-format reader fallback.

## Expected Reduction

| Current concept | Proposed destination |
|---|---|
| `LatentStructure` as a separately persisted definition | Constructs and edges in `Model`; graph accessors |
| `MeasurementStructure` and its artifact-level declarations | Indicators and usage on their constructs; model clock |
| `CausalDesign` as a combined copy | The existing model revision; identification remains a sourced result |
| `StatisticalModelSpec` as a parallel catalog | Likelihoods, mechanisms, parameters, and policies on the same model |
| `StructuralSemanticCatalog` | Removed; execution planning reads canonical entities |
| `StructuralPlan` as a scientific root | Derived execution projection, optionally cached for reuse and inspection |
| Compiler-created and copied parameter definitions | Complete canonical inventory plus compiled bindings |
| Separate lists joined by construct, indicator, or edge ID for exclusive ownership | Containment and accessors; shared references remain explicit |
| `ModelSnapshot` fields mirroring the authoring stages | One canonical model value, alongside distinct data and sourced results |

The meaningful reduction is fewer independent declarations, joins, and rules
that keep representations synchronized. The exported graph may still contain
small component types where they express real ownership or alternatives.

## Decisions Before Coding and During Migration

Establish the contracts that cross component boundaries before implementation.
A complete field-by-field map of the repository is not a prerequisite. Produce
that map for each owner as its definitions and consumers are migrated.

| Establish before coding | Recommended contract |
|---|---|
| Canonical ownership | One scientific `Model`; exclusive ownership uses containment, shared relationships use IDs. Parameters have one registry. Programs, data, and findings carry their own provenance. |
| Validity and edits | Missing choices are allowed; contradictory declarations and dangling references are rejected. An update includes all repairs needed to produce a consistent revision. Operation readiness is checked when the operation is requested. |
| Revision and commit semantics | An edit names its base revision and commits one immutable, validated model value through the existing journal. A stale base is rejected. Begin with full stored model values. |
| Result dependencies | Results retain their exact source revision and external inputs. Applicability depends on the canonical inputs consumed: changing a prior must preserve extraction while invalidating affected compilation and inference. |
| Storage and API transition | The completed refactor has one write/read contract and generated clients, with the old scientific schemas removed. Development fixtures are regenerated; retained history is preserved through an explicit offline conversion with originals retained. |

Ownership exceptions deserve early attention because they can change the tree:
shared likelihood parameters, explicit latent confounders, known inputs, and
parameter components with an ordered basis. Their meanings and ownership must be
settled before moving their fields. Existing numerical laws, identity rules, and
scientific admission requirements remain preservation constraints throughout.

During implementation, determine exact field names, small component boundaries,
accessor signatures, module layout, serializer details, and individual validator
placement from the definitions and consumers being migrated. Specify each
consumer's input projection when converting that consumer. Derive the concrete
conversion script from the completed field mappings.

Before switching stored workspaces or API clients, complete the conversion and
verify its historical behavior, generated contracts, and reader agreement. That
is a deployment requirement rather than a reason to finish every mapping before
the first source edit.

The first implementation slice should follow a construct, its indicator and
likelihood, an edge with both linear and Hill contributions, and their declared
parameter priors through validation and compiler binding. Use that slice to check
additive composition, complete scientific declarations before compilation, and
the ownership and revision contracts before expanding across the remaining
components. Any discovery that changes a cross-component contract should update
this design explicitly.

## Implementation Sequence After Review

1. Settle the ownership and revision contracts above, including the shared and
   cross-entity cases. Map fields owner by owner during the following increments.
2. Introduce the canonical value and pure construction operations. Move intrinsic
   validation to its owners and whole-model reference checks to the aggregate.
   Complete the scientific parameter inventory before compilation.
3. Make planning and compilation consume that model through explicit input
   projections. Preserve all existing numerical choices and binding invariants.
4. Switch scientific stage writes, journal selection, readers, and generated API
   contracts together. Delete the superseded catalogs and composition adapters.
5. Consolidate the native runtime metadata and regenerate the type graph to
   measure the actual reduction.

Apply the [storage and API transition contract](#storage-and-api-transition) when
switching the completed implementation.

Verification should use source comparison and cheap contract checks: entity and
parameter identity, completeness errors, native distribution round trips,
preservation of additive terms and their bindings, compiled layout equivalence,
historical reads, dependency invalidation, and generated API agreement. Check that
compilation leaves the scientific inventory and declarations unchanged. Expensive
CPU/GPU tests remain separately tagged and require explicit authorization; no
inference run is needed to review this design.

## Remaining Design Work

The accepted direction makes scientific entities the places where knowledge
accumulates and retires stage-shaped scientific roots. Implement the cross-component
contracts above. Complete the detailed ownership map,
validators, dependency selectors, and conversion alongside the affected code,
then verify their agreement before switching readers and writers. Containment,
valid partial models, and immutable revisions must work together throughout.

The main tradeoff is the size of the storage and workflow change. It offers more
reduction than reorganizing today's wrappers, because it removes the reason those
wrappers and synchronization paths exist.
