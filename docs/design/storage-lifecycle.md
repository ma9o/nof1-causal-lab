# Storage Lifecycle and Commit Boundaries

Each workspace has three storage tiers with different correctness and retention rules:

```text
data/{workspace_id}/
├── input/                         durable user inputs
├── store/                         content-addressed numerical arrays and table blobs
├── episode/history.git/           durable artifact trees, commits, logs and refs
├── cache/                         regenerable computation reuse
└── scratch/
    ├── events/                    live UI telemetry
    └── runs/{run_id}/             run-scoped execution state and checkpoints
```

## Durable Git History

Artifact activities write immutable Git trees before returning their transition effects. An artifact becomes current when an applied action commits its tree under `artifacts/{artifact_id}/` and advances the selected branch. Current state comes directly from that commit's tree. Git owns identity, ancestry and branch heads; the [study history contract](study-history.md) defines the application paths and refs.

Finalized LLM traces follow the same write-before-commit rule. The publication activity discovers the finalized `llm/{subroutine_id}/trace.json` files owned by the attempt's scratch run and places them in its commit under `logs/traces/`. The same commit captures `logs/transition.json` and `logs/events.json`. Failed attempts retain their own commits without advancing the scientific branch. Each commit owns only its action's logs; earlier evidence remains on its Git ancestry.

A raised model-spec transition may contain a typed `(run_id, checkpoint_id)` resume selection. The checkpoint module alone resolves that selection into scratch storage. The collector preserves the selected run until a later model-spec transition supersedes it; artifact lineage and trace references remain closed within the durable tier.

## Scratch and Checkpoints

Everything produced while a transition executes lives below one `scratch/runs/{run_id}/` directory. Conversations, tool exchanges, staged contexts, extraction chunks, and model-spec checkpoints are one collection unit.

Checkpoints are not cache entries. They preserve paid semantic work needed to resume a raised model-spec transition, so run collection protects the selection referenced by the latest raised attempt. A later model-spec attempt releases or replaces that selection. Automatic collection runs under the episode action lock; offline collection is explicit and makes no filename-based liveness inference.

The UI event stream lives under `scratch/events/`. It reports intra-transition progress but never participates in state reconstruction. Once captured in an attempt's commit, its durable events remain available after live telemetry expires.

## Cache

Cache entries are safe to delete at any time. Admission evaluations are content-addressed by all semantic inputs and reject hash collisions. JAX compilation sidecars are guarded by their topology fingerprint and schema version. Cache collection applies both age and total-size bounds.

## Collection and Publication

The episode workflow collects completed run scratch after every attempt is committed while its per-episode action lock is still held. Collection failure is logged but never changes the already-committed action outcome. `uv run nof1-sweep WORKSPACE_ID` expires telemetry and caches without guessing whether a run is active; offline maintenance may additionally pass `--collect-runs`.

[Fixture promotion](../guides/agentic_integration_testing.md#promoting-a-workspace-to-the-demo-fixture) copies the durable workspace and excludes `scratch/` and `cache/`. It exports all Git refs to `episode/history.bundle` for the tracked fixture; numerical payloads remain under `store/`. Both are needed to restore the study. Generated files under `fixture/` are read projections of that stored history.
