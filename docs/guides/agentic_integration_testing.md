# Agentic Integration Testing

## Design Principles

| Concern | Mode | Why |
|---------|------|-----|
| File placement, scientific actions, run registration | **Programmatic** (`curl`) | Reliable, fast, no UI fragility |
| Runtime errors, route discovery, build state | **Next.js devtools MCP** | Reads the running app directly and catches server/runtime errors before UI debugging |
| UI rendering verification, visual regression | **browser automation** (`browser_eval` / Playwright) | Only way to see rendered output |

## Prerequisites

### Test concerns

Every Python test declares at least one [pytest concern marker](../../apps/data-pipeline/pyproject.toml),
including standalone notebook support tests. The top-level groups are `contract`,
`inference`, and `workflow`. Numerical tests declare a specific child of `inference`:

| Group | Child concern | Covers |
|-------|---------------|--------|
| `contract` | — | Isolated routing, validation, schema, identity, and projection without scientific numerical execution |
| `inference` | `sampling` | Parameterization, exact probability targets, particle sampling, and posterior diagnostics |
| `inference` | `warmup` | Laplace and optimizer initialization, local linearization, solver references, and gradients |
| `inference` | `simulation` | State dynamics, deterministic/stochastic trajectories, steady states, and interventions |
| `inference` | `predictive` | Prior/posterior predictive generation, observation sampling, and predictive diagnostics |
| `inference` | `recovery` | Fitted accuracy and uncertainty checked against known generating parameters |
| `workflow` | — | Temporal workflows and worker orchestration |

Numerical tests use `pytest.mark.inference(concern="warmup")`, with the appropriate
child name. Pytest's [native keyword matching](https://docs.pytest.org/en/stable/example/markers.html#marking-test-functions-and-selecting-them-for-a-run)
makes `-m inference` select the whole numerical group and
`-m "inference(concern='warmup')"` select that child alone. No parent markers are
injected. A bare `inference` marker is invalid: every numerical test must name a
child concern. Count each test once when reporting totals for its parent group,
even when it has multiple child owners.

Apply a marker when a test executes that subsystem. Use `contract` for the isolated
checks in the default selection. Small inputs, fixed draws, a mocked optimizer, or
shape tracing do not make scientific execution a contract: exact density and
parameter-gradient checks belong to `sampling`, predictive draws and statistical
checks to `predictive`, and initialization solvers to `warmup`. A test that checks
both exact density and observation draws owns both `sampling` and `predictive`.
Keep routing tests in `contract` when the numerical execution is stubbed, and
validation tests when they reject inputs before execution. Attribute real Temporal
child workflows to `workflow`, even without a complete episode journey.

Use function or class markers in mixed files;
a module's `pytestmark` applies only when every test shares the concern. A test can
have multiple owners: the MAP recovery suite carries both
`pytest.mark.inference(concern="warmup")` and `pytest.mark.inference(concern="recovery")`.
The [collection check](../../apps/data-pipeline/conftest.py) rejects unowned cases
and missing or invalid inference concerns before `-m` or `-k` can deselect them.
Parameterization and timeout markers do not count as owners, and missing ownership
is never assigned automatically.

`bun run --cwd apps/data-pipeline test` and direct `uv run pytest tests/` runs
use all available CPU workers (`-n auto`) and run the default contracts. Select specialized suites explicitly
when requested or directly affected by the change:

```bash
bun run --cwd apps/data-pipeline test -m contract
bun run --cwd apps/data-pipeline test -m inference
bun run --cwd apps/data-pipeline test -m "inference(concern='sampling')"
bun run --cwd apps/data-pipeline test -m "inference(concern='predictive')"
bun run --cwd apps/data-pipeline test -m "inference(concern='warmup') and not inference(concern='recovery')"
bun run --cwd apps/data-pipeline test -m "inference and not inference(concern='recovery')"
```

Add `--collect-only` to inspect a selection without executing it. `test:all`
includes every concern and requires an explicit request. Use `-n 0` for a serial
run or `-n N` to limit the worker count. Pytest reports durations separately and rejects
unregistered markers. Keep test files beside their subject under `tests/`.
The default selection is `not (inference or workflow)`, which excludes every
numerical child, including tests that also carry `contract`. Web unit tests belong to the named `contract` Vitest project;
the separate `visual` project owns screenshot stories. The root
`test:fixture-promotion` command owns fixture-promotion checks.

### Choosing tests for a change

Before testing, identify the behavior changed and its direct consumers, then choose
the smallest set of test files or node IDs that covers them. Select the relevant
contracts and numerical children from the concern table above. Test ownership
follows the behavior executed, so a source directory alone does not determine
which concerns need to run.

| Changed behavior | Relevant selection |
|------------------|--------------------|
| Schema, routing, validation, identity, or projection | Affected `contract` tests |
| Exact density, parameterization, particle sampling, or posterior diagnostics | `inference(concern='sampling')` and affected contracts |
| Initialization, Laplace, optimization, or solver references | `inference(concern='warmup')` and affected contracts |
| Dynamics, trajectories, steady states, or interventions | `inference(concern='simulation')` and affected contracts |
| Prior/posterior prediction, observation draws, or predictive checks | `inference(concern='predictive')` and affected contracts |
| Estimation accuracy or uncertainty against known generating parameters | `inference(concern='recovery')` plus the changed fitting or warmup concern |
| Temporal orchestration | Affected `workflow` tests and routing contracts |
| Test selection or ownership configuration | `tests/test_concerns.py`, then collection without scientific execution |
| Documentation only | `bun run docs:check` |

Include multiple children when the change affects their shared behavior. For
example, an emission change may affect both exact density and predictive draws;
an initialization-only change calls for warmup checks. Use the whole `inference`
group when shared numerical changes affect all its children, or when explicitly
requested. Run recovery tests when the change can affect fitted accuracy or
uncertainty, rather than merely because it touches a fitting-related file.

For a file-scoped Python run, invoke pytest directly from `apps/data-pipeline`:

```bash
uv run pytest tests/models/ssm/test_transition_builder.py -m "contract or inference(concern='warmup')" -n 0
uv run pytest tests/models/ssm/test_emissions.py tests/models/ssm/test_observation_sampling.py -m "contract or inference(concern='sampling') or inference(concern='predictive')"
uv run pytest tests/test_concerns.py -m contract -n 0
```

Always supply the numerical or workflow selector when targeting those tests:
the default excludes them even when their file is named explicitly. The Bun
`test` wrapper includes `tests/` in its command, so appending a file does not
narrow it to that file. Reserve that wrapper for a whole concern suite. Use
`-n 0` for small selections when worker startup would dominate; the configured
default remains `-n auto`.

Use `--collect-only` if the selection is uncertain, and check that the intended
tests are selected. Once focused checks pass, broaden only for affected shared
behavior, failure diagnosis, or an explicit broader request. Record the command,
selected concerns, and outcome in the handoff. Follow the separate required
lint, type, documentation, and duplicate checks when they apply to the change.

### Local study history

Existing studies need the one-time [Git history migration](../design/study-history.md#migrating-existing-local-studies). Keep the study offline during migration. New studies initialize their local bare repository on first use. On a fresh checkout, restore the tracked `DEMO` history bundle before using its backend:

```bash
git clone --mirror data/DEMO/episode/history.bundle data/DEMO/episode/history.git
git --git-dir=data/DEMO/episode/history.git config nof1.format 4
```

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
│   ├── store/             # Content-addressed arrays and external table blobs
│   ├── episode/           # Local Git history with logs and traces in each commit
│   ├── cache/             # Evictable compilation and artifact-read reuse
│   └── scratch/           # UI telemetry and run-scoped execution state
└── DEMO/                  # Tracked mock fixture workspace (evals + manual sampling)
```

### Promoting a workspace to the DEMO fixture

Keep ordinary data workspaces and the committed fixture separate. Once a candidate
workspace has a complete, fresh artifact chain, promote it explicitly:

```bash
bun run fixture:promote --from <WORKSPACE_ID>
```

The command validates the selected Git snapshot, copies the durable workspace into
`data/DEMO`, and rebuilds stable JSON and trace copies under `data/DEMO/fixture/`
for Storybook and tests. It replaces `data/DEMO` as a unit rather than merging,
and excludes `cache/` and `scratch/`. It exports all Git refs and objects to
`episode/history.bundle` so the tracked fixture retains branches, attempts and
artifact trees while its local bare repository remains gitignored. The files in
`store/` retain the external numerical payloads.

The tracked `data/DEMO/episode/history.bundle` and `data/DEMO/store/` are the fixture's authoritative inputs. Files under `data/DEMO/fixture/` are generated projections for Storybook and tests. Regenerate or check them with:

```bash
bun run fixture:demo
bun run fixture:demo:check
```

Both commands restore the bundle into an isolated temporary repository and use the production readers to project artifacts, logs, traces and historical snapshots. They do not read the local `history.git` or the generated projections as inputs. DEMO has no numbered artifact directories or separate journal and trace directories.

The bundle preserves the existing illustrative history and numerical findings. Its original posterior samples were not retained, so the fit remains explicitly report-only. Archived predictive checks belong to that attempt at `logs/predictive_checks.json` and are projected into the fixture directory. Regeneration does not fit, simulate, or invent missing scientific artifacts. Prior plot viewports use a small deterministic draw from the retained prior laws.

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

The facade is the source of truth for action availability. `GET /api/capabilities`
reports `actions_enabled`, which is false on a read-only facade. Creating the run
submits `edit_model` with the question. Subsequent preparation, fitting and custom simulation designs require explicit scientific actions. Model edits automatically run applicable predictive checks when a compatible panel is available; a question-only model has no executable simulation.
Creation returns HTTP `202` with the workspace and `attempt_id`. Poll that attempt
at `GET /api/episodes/{workspace_id}/actions/{attempt_id}` until `done` is true.
Direct `/actions` dispatches return only `{attempt_id}`; scientific bodies and
timestamped labels are available through polling.

### 2. Observe the episode

The episode facade (tool server, port `8100`) is the source of truth:

```bash
# Current state: artifact existence, freshness, revisions, and the four action names
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID | jq '.artifacts'

# The transition journal: every action attempt (applied / rejected / raised)
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | {seq, status, action, error_type}'

# Transition telemetry (extraction worker fan-out and action labels)
curl -s "http://localhost:8100/api/episodes/$WORKSPACE_ID/events" | jq '.events[-3:]'
```

### 3. Verify via browser automation

The two interfaces have separate URL namespaces while the workbench is validated against a fresh study:

- `http://localhost:3000/v1/{WORKSPACE_ID}` preserves the stage-by-stage pipeline interface.
- `http://localhost:3000/v2/{WORKSPACE_ID}` presents the model workbench. This is the default destination from the workspace list and after creating a workspace.

Both read the same backend study. The version links preserve the workspace when switching interfaces. The workbench reads the episode journal and canonical snapshots directly, including an empty initialized study; it does not require a completed recipe or a legacy analysis manifest. Verify the question, graph, owned entity details, data, findings, history, and action log after each backend action. V1 interactivity is not a compatibility requirement. The agent harness submits changes through the backend API; v2 provides read-only inspection and navigation without write controls or a chat composer. Its action log displays timestamped labels from HTTP polling alongside recorded traces.

The persistent model view has a branching timeline across the top, a causal graph in the centre, scoped details beneath the graph, and a chat log at full body height on the right. Selecting a timeline version or recorded chat turn updates the graph, details and chat together. Selecting a graph entity scopes the bottom details pane to that entity. Fit results and simulation evidence are sections of the relevant details. Failed attempts show their recorded error and the unchanged parent version.

Hover or focus another version to overlay its differences directly on the selected graph, including ancestors, other branches and simulation steps. The backend identifies changed constructs, causal connections and owned parameters. Colored outlines and strokes mark additions, removals and revisions; annotations show changes such as `pinned 0 → free` on the affected element. Existing node positions, zoom and scroll remain in place. Added nodes extend the canvas without rearranging the selected graph. The details and chat stay visible. **Keep** holds the overlay; **Escape** or the close button dismisses it. The timeline's small **Compare** control also works by keyboard or touch. Comparisons select Git commit OIDs, so two versions using the same model can carry different simulation evidence.

Before execution dispositions exist, the graph shows the authored structure. Afterwards it shows the backend-selected retained constructs and edges, excluding components disconnected from the default outcome after projection. Compare with an earlier structural version through the same timeline controls to inspect marginalized, unsupported, or disconnected exclusions and their recorded reasons. An identification warning on a retained construct does not hide that construct. Unresolved dependencies of the retained component still fail the backend's specification checks.

Solid lines show Git commit ancestry. **Collapse lineage** shows the selected version's branch in one row, including its shared ancestors and recorded actions; **Expand lineage** restores all branches for comparison. The workbench follows the latest committed work. Agents create branches through the backend API; the UI displays the resulting history. Historical versions remain available for inspection and comparison.

Storybook preserves pipeline stories under **V1 / Pipeline** and has one comprehensive **V2 / Model / Workbench / Complete** story. Extend that scenario when adding workbench features, rather than creating separate feature or state stories. It includes two branches, hover comparison, synchronized version and chat navigation, graph and parameter inspection, and scoped simulation and fit evidence. The story uses isolated mocked API responses and illustrative data. It offers no model writes, scientific job submissions or recipe controls.

The story's pinned model and comparison responses are generated with the production readers by `bun run fixture:workbench`; verify them with `bun run fixture:workbench:check`. This pins one retained DEMO parameter for illustration and runs no fitting or simulation.

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming After a Transition Failure

A failed transition run is a `"raised"` transition in the journal — artifact
state is unchanged, and the typed error plus diagnostics ride on the record:

```bash
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | select(.status=="raised") | {seq, action, error_type, error_message, resume}'
```

Inspect the recorded action and its inputs, correct the cause, and resubmit the
corresponding request to `/actions`. Select fresh revisions when inputs changed.
There is no generic job-submission or automatic-resume endpoint.

```bash
# fit-request.json names the selected model_revision and panel_revision.
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/actions \
  -H 'Content-Type: application/json' --data-binary @fit-request.json
```

The question is retained in each [ModelSpec revision](../pipeline/latent-structure.md#modelspec), so subsequent operations read it from their pinned Model input.

## Editing the Model

Read the current Model and its source version, edit the owned scientific entities, and submit the whole candidate with that expected version. The [model write contract](../design/model-snapshot.md#writes-and-operation-history) checks and commits the model with its findings atomically. Negative identification findings remain explicit reports. Relevant changes trigger one automatic exact predictive batch when a compatible prepared panel exists; unchanged checks are reused.

The workbench displays the entity fields the agent harness edits through the API. An edge shows linked endpoint constructs separately from its description, timing, mechanisms, and sources. Construct details expose the shared construct, including its indicators and intrinsic dynamics. Verify that relocating a shared definition within the serialized graph does not change these views, and that selecting a version without the entity clearly reports its absence.

```bash
# model-update.json contains {"action": "edit_model", "expected_revision": <model tree OID>, "model": <candidate>}
curl -s -X POST http://localhost:8100/api/episodes/$WORKSPACE_ID/actions \
  -H 'Content-Type: application/json' --data-binary @model-update.json
```

Use `null` only when creating the first Model; otherwise supply its Git tree OID. A stale base rejects the action. The [offline conversion guide](additive-model-migration.md) covers retained episodes from the former scientific schemas.

### Scientific Actions and Internal Jobs

Submit `edit_model`, `prepare_data`, `fit`, or `simulate` through `/actions` using
the [typed contracts](../reference/scientific-actions.md). Explicit revisions let
you fit an earlier definition or compare simulations without changing the current
model first. Dated interventions use the same simulation action; causal summaries require identification
and production-fit evidence.

Dependencies are artifact-level; `GET /api/machine` exposes
`topological_artifact_order` and `topological_transition_order` from
[`machine/graph.py`](../../apps/data-pipeline/src/nof1_causal_lab/machine/graph.py).
The episode workflow chooses private jobs from each typed action request.
`edit_model` validates the caller's supplied model directly, and runs applicable checks, including automatic whole-model prediction, without an internal authoring prompt builder. Direct fitting requires
an executable selected model and compatible observations, without authoring
admissions or positive causal identification. Negative identification findings
remain visible and restrict numeric causal claims. Missing inputs produce
unevaluated checks or an operation-readiness failure.
