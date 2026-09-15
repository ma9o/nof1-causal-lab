# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, episode moves, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

### Test concerns

The specialized suites use [pytest concern markers](../../apps/data-pipeline/pyproject.toml).
Each marker identifies the behavior exercised, independently of runtime or device:

| Marker | Concern |
|--------|---------|
| `inference` | Particle runtimes, exact likelihood targets, and posterior sampling |
| `warmup` | Laplace initialization, local linearization, solver references, and gradients |
| `simulation` | Deterministic/stochastic trajectories, steady states, and interventions |
| `predictive` | Prior/posterior predictive generation and observation sampling |
| `recovery` | Fitted accuracy and uncertainty checked against known generating parameters |
| `admission` | Complete construct and edge admission batteries |
| `workflow` | Complete Temporal episode journeys |

Apply a marker when a test executes that subsystem. Isolated routing, validation,
identity, projection, and small array-reduction contracts remain unmarked in the
default selection. Use function or class markers in mixed files; a module marker
applies only when every test shares the concern. A test can have multiple concerns:
the MAP recovery suite carries both `warmup` and `recovery`.

`bun run --cwd apps/data-pipeline test` and direct `uv run pytest tests/` runs
use all available CPU workers (`-n auto`) and run the default contracts. Select specialized suites explicitly
when requested or directly affected by the change:

```bash
bun run --cwd apps/data-pipeline test -m simulation
bun run --cwd apps/data-pipeline test -m "warmup and not recovery"
bun run --cwd apps/data-pipeline test -m "inference or predictive"
```

Add `--collect-only` to inspect a selection without executing it. `test:all`
includes every concern and requires an explicit request. Use `-n 0` for a serial
run or `-n N` to limit the worker count. Pytest reports durations separately and rejects
unregistered markers. Keep test files beside their subject under `tests/`.

### Local stack

```bash
bun run integration:start
```

Starts the stack under [process-compose](https://github.com/F1bonacc1/process-compose)
supervision (`brew install f1bonacc1/tap/process-compose`; config:
[`process-compose.yaml`](../../process-compose.yaml)): the Temporal dev
server on port `7233` (ephemeral state, binary auto-downloaded on first
use), the episode worker (task queue `nof1-episodes`), the tool server
with the episode facade on port `8100`, and the web app on port `3000`.
Startup order is health-gated (`depends_on` + readiness probes) and
crashed processes restart automatically. The script **stays in the
foreground** — wait until `curl -s http://localhost:8100/api/capabilities`
answers before proceeding (pass `-t=false` to disable the TUI when
redirecting output to a file).

To tear down the stack, kill the script (`Ctrl+C`, or `kill <pid>` if
backgrounded). All child processes are cleaned up automatically.

The process-compose API is pinned to port `8181` for targeted operations
against the running stack:

```bash
process-compose --port 8181 process list
process-compose --port 8181 process restart worker
```

The worker caches `apps/data-pipeline/config.yaml` at first read
(`lru_cache`), so after editing pipeline config, restart the `worker`
process — no need to bounce the whole stack. The Temporal dev server
persists its event history to `.local/agentic-integration-stack/temporal.db`
(the `--db-filename` on its command), so restarting the `temporal`
process — to serve the UI, pick up a change, or recover from a crash —
**resumes** in-flight episode workflows exactly where they left off
rather than orphaning them. To start genuinely fresh, delete that
`temporal.db` before boot and wipe that workspace's `store/`, `episode/`,
`scratch/`, and `cache/` directories. The Web UI is served at
`http://localhost:8233`.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Raw uploaded files for the raw_data transition
│   ├── store/             # Versioned artifact store ({artifact}/v{N}/)
│   ├── episode/           # Durable transition journal and promoted LLM traces
│   ├── cache/             # Evictable admission and compilation reuse
│   └── scratch/           # UI telemetry and run-scoped execution/checkpoints
└── DEMO/                  # Tracked mock fixture workspace (evals + manual sampling)
```

### Promoting a workspace to the DEMO fixture

Keep ordinary data workspaces and the committed fixture separate. Once a candidate
workspace has a complete, fresh artifact chain, promote it explicitly:

```bash
bun run fixture:promote --from <WORKSPACE_ID>
```

The command validates the episode journal, copies the durable workspace into
`data/DEMO`, and rebuilds stable JSON and trace copies under `data/DEMO/fixture/`
for Storybook and tests. It replaces `data/DEMO` as a unit rather than merging,
and excludes `cache/` and `scratch/`.

Retained illustrative artifacts live under `data/DEMO/fixture/`. Recompose their read views with:

```bash
bun run fixture:demo
bun run fixture:demo:check
```

This validates the retained canonical Model and findings and uses the production reader to generate snapshots and artifact views. An incomplete model remains available for inspection; inference report freshness follows its pinned inputs. It preserves the stored episode and all retained numerical results. It does not fit, simulate, or invent missing scientific artifacts. Prior plot viewports use a small deterministic draw from the retained prior laws. A complete promoted episode supplies its own canonical payloads.

## Step-by-Step Flow

### 1. Create the workspace and start the run

```bash
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"

curl -s -X POST http://localhost:3000/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@data/DEMO/input/dsar_bundle.zip"

curl -s -X POST http://localhost:3000/api/runs \
  -H 'Content-Type: application/json' \
  -d "{\"workspaceId\":\"$WORKSPACE_ID\",\"query\":\"$QUESTION\"}"
```

There is no auth: the facade is the source of truth for what is allowed, and
`curl -s http://localhost:8100/api/capabilities` reports whether the move
plane is available (`moves_enabled` is `false` on a read-only facade).

This creates or revises the Model’s `question` field and starts the **auto-run driver**: a
default navigation policy that proposes operation-named `run` moves in dependency
order while transition outputs are missing or stale, stopping when the episode
is quiescent or a move fails.

### 2. Observe the episode

The episode facade (tool server, port `8100`) is the source of truth:

```bash
# Current state: artifact existence/staleness/versions + legal moves
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID | jq '.artifacts'

# The transition journal: every move attempt (applied / rejected / raised)
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | {seq, status, move, error_type}'

# Transition telemetry (extraction worker fan-out, model-spec agent graph)
curl -s "http://localhost:8100/api/episodes/$WORKSPACE_ID/events" | jq '.events[-3:]'
```

### 3. Verify via browser automation

Navigate to `http://localhost:3000/model/{WORKSPACE_ID}` and verify the four action controls, selected input revisions and reported findings. The optional recipe progress view is at `/recipes/observational-study/{WORKSPACE_ID}`.

The persistent model view lives at `http://localhost:3000/model/{WORKSPACE_ID}`: one causal model per workspace with the version scrubber across the top, the graph over the scoped details pane, and the journal as a conversation on the right. Select a scrubber tick to open that move's version scope and its time breakdown; double-click a tick to view the asset as it stood then.

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming After a Transition Failure

A failed transition run is a `"raised"` transition in the journal — artifact
state is unchanged, and the typed error plus diagnostics ride on the record:

```bash
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | select(.status=="raised") | {seq, move, error_type, error_message, resume}'
```

Re-running is just proposing the move again (the machine validates
enabledness; there is no window arithmetic):

```bash
# Run one transition
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/moves \
  -H 'Content-Type: application/json' \
  -d '{"move": {"kind": "run", "operation_id": "statistical_model_spec"}}'

# Or resume the default policy (runs everything enabled and stale/missing)
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/recipes/observational-study \
  -H 'Content-Type: application/json' -d '{}'
```

The question is retained in each [ModelSpec revision](../pipeline/latent-structure.md#modelspec), so subsequent operations read it from their pinned Model input.

## Editing the Model

Read the current Model and its source version, edit the owned scientific entities, and submit the whole candidate with that expected version. The [model write contract](../design/model-snapshot.md#writes-and-operation-history) validates and commits required derivations atomically. Negative identification findings remain explicit reports. Changed inputs invalidate dependent results without launching another expensive run.

```bash
# model-update.json contains {"action": "edit_model", "expected_version": <current version>, "model": <candidate>}
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/actions \
  -H 'Content-Type: application/json' --data-binary @model-update.json
```

Use `0` only when creating the first Model. A stale base rejects the action. The [offline conversion guide](additive-model-migration.md) covers retained episodes from the former scientific schemas.

### Scientific Actions and Internal Jobs

Submit `edit_model`, `prepare_data`, `fit`, or `simulate` through `/actions` using
the [typed contracts](../reference/scientific-actions.md). Explicit revisions let
you fit an earlier definition or compare simulations without changing the current
model first. Causal designs use the same simulation action with additional evidence
requirements.

Dependencies are artifact-level; `GET /api/machine` exposes
`topological_artifact_order` and `topological_transition_order` from
[`machine/graph.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py).
The optional observational-study recipe orders these internal jobs:

```text
raw_data → latent_structure → measurement_structure → measurements → statistical_model_spec → posterior
```

The separate `simulate` job runs only when requested. Direct fitting requires
an executable selected model and compatible observations, without authoring
admissions or positive causal identification. Negative identification findings
remain visible and restrict numeric causal claims. Missing inputs produce
unevaluated checks or an operation-readiness failure.
