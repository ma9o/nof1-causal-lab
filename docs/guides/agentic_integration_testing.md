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

New studies initialize their local bare repository on first use. On a fresh checkout, restore the tracked `DEMO` history bundle before using its backend:

```bash
git clone --mirror data/DEMO/episode/history.bundle data/DEMO/episode/history.git
git --git-dir=data/DEMO/episode/history.git config nof1.format 6
```

#### Migrating a local study

Format-3 studies still store extraction instructions on their models. To convert one into a new format-4 copy:

1. Stop work on the study and close its episode workflow.
2. Run the converter with the original uploaded filenames. The destination must be new and outside the source, and the source is left untouched.

   ```bash
   uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_data_preparation \
     ../../data/STUDY /tmp/migrated/STUDY --files diary.csv
   ```

   The converter moves scoring instructions into panel metadata and adds numerical data profiles, without fitting or generating trajectories. It doesn't invent scoring rules or codebooks: variables without retained definitions stay explicit profile findings. `--preparations-json` can supply reviewed panel-revision-to-preparation specs.
3. Review the migrated snapshots, then select the new workspace while offline.
4. Restart the workers with the new code and start a fresh episode workflow from the migrated Git state; don't replay the previous workflow. Regenerate any fixture bundle from the migrated repository.

Older numbered-artifact and format-2 histories have no validated route to the current runtime.

For a format-4 study that still declares `recording` or nested `fill_null`, follow the same stop, review and restart steps with the [null-filling converter](../../apps/data-pipeline/scripts/migrations/migrate_fill_null.py):

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_fill_null \
  ../../data/STUDY /tmp/migrated/STUDY
```

The converter rewrites observation definitions, model-input fingerprints and revision references in a new copy, preserving saved numerical tables and results. It translates retired fields to the shared [`ObservationSpec`](../../apps/data-pipeline/src/nof1_causal_lab/artifacts/observations.py) with flat `fill_null`; subsequent preparation uses [Polars null-filling semantics](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.fill_null.html), including explicitly null values.

For a format-4 study whose fitted models list one point mass per retained posterior draw, follow the same steps with the [posterior-law converter](../../apps/data-pipeline/scripts/migrations/migrate_empirical_laws.py):

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_empirical_laws \
  ../../data/STUDY /tmp/migrated/STUDY
```

It stores each retained posterior as one batched point mass over its saved draws and moves the equal weights into the array store, rewriting model-input fingerprints and revision references in a new copy. Draws, tables and results are reused unchanged.

For a format-4 study whose stored predictive overlays still carry quantile bands, use the [predictive-overlay converter](../../apps/data-pipeline/scripts/migrations/migrate_predictive_overlays.py) the same way:

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_predictive_overlays \
  ../../data/STUDY /tmp/migrated/STUDY
```

It drops the band arrays and keeps evenly spaced sample series, as many as new checks record. Observed values, medians and all other check results are preserved. Run this converter before the format-5 conversion below.

After any necessary format-4 conversions above, create format 5 with the [simulation/preparation converter](../../apps/data-pipeline/scripts/migrations/migrate_simulation_preparation.py) into another new copy:

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_simulation_preparation \
  ../../data/STUDY /tmp/format5/STUDY
```

It records panel and historical fit origins, backfills simulation summaries from saved draws, and rewrites all changed Git references. It preserves historical numerical coordinates under the [time semantics](../assumptions.md#time). Imported panels without a files recipe are rejected before copying; their preparation requires a reviewed scientific decision. No model calls, fitting or simulation run during migration. Review the new copy and start a fresh workflow as above.

The current runtime requires format 6. Convert a stopped format-5 study with the [edge-timing converter](../../apps/data-pipeline/scripts/migrations/migrate_edge_timing.py):

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.migrate_edge_timing \
  ../../data/STUDY /tmp/edge-timing/STUDY
```

It removes `lagged`, recomputes identification from the constructs’ temporal status, and rewrites references in a new copy. Equations and numerical arrays are preserved. Review the copy and restart the workflow as above. The [temporal assumptions](../assumptions.md#model-class) describe the revised interpretation.

For this checkout's switch-over, move the existing local DEMO `episode/history.git` outside `data/` and re-clone its tracked bundle using the [restore commands](#local-study-history). Move STEPWISE and the other older-format local studies (`V2BUILD01`, `ws`, `ws-test`) outside `data/` as archived workspaces, keeping each whole directory and its numerical store. STEPWISE's imported panels have no files recipe and are archived, not converted or rebuilt. A later files-path rebuild requires a separate budgeted end-to-end run.

#### Squashing a local study's action history

Stop work on the study and close its episode workflow, as in the [migration procedure](#migrating-a-local-study). The [history squash script](../../apps/data-pipeline/scripts/migrations/squash_study_history.py) compacts saved effects through an applied commit `R` into a new directory outside the source. Keep the directory's basename, which is the logical workspace ID:

```bash
uv run --directory apps/data-pipeline python -m scripts.migrations.squash_study_history \
  ../../data/STUDY /tmp/squashed/STUDY --at R --dry-run
uv run --directory apps/data-pipeline python -m scripts.migrations.squash_study_history \
  ../../data/STUDY /tmp/squashed/STUDY --at R
```

Replace `R` with the applied commit OID. The dry run lists retained and dropped attempts without copying. The script keeps the root, `R`, the entire suffix, and the dependency closure of the prefix's last artifact/check writers (including retractions) and latest fresh simulation. It preserves saved artifact, check and log objects, all artifact refs, numerical files and authorship links; `squash-mapping.json` maps original commits to new commits or `null` for dropped actions. Sequence numbers and attempt IDs retain their gaps.

At `R` and later commits, artifacts (including absence), checks and fresh reader findings are preserved. Stale findings can disappear, and earlier snapshots can change. This is the smallest closure of the mandatory writers, not a globally minimal history: reused reports and redundant retractions can retain extra actions. There is no optimizer, numerical execution or post-squash equality gate.

Only current-format, single-branch histories are supported. The script refuses legacy `statistical_model_spec`/`report_only` records, simulation-replicate panels anywhere in the preserved catalog, other branches (including successful attempts off the branch), and retained scientific inputs without a recorded producer. Review the new copy, select it offline, then start a fresh workflow and regenerate any fixture bundle using the migration procedure above. The source is unchanged.

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
process — no need to bounce the whole stack. If the supervisor was started with
`up --no-deps web`, `process-compose project update` can disable its dependency
processes; check that Temporal is still running afterward.

For local-study GPU fits, set `inference.compute_backend: modal` in
[`config.yaml`](../../apps/data-pipeline/config.yaml) and restart the worker.
The [Modal compute adapter](../../apps/data-pipeline/src/nof1_causal_lab/flows/modal_fit.py)
uses the installed Modal credentials, creates an ephemeral
`nof1-causal-lab-pipeline` app with the current source, and reuses
`nof1-cached-fit-cache:/jax`. The study remains local; the
[fit chart](../assets/action-flows/fit.svg) owns its failure and publication flow.

The Temporal dev server
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
│   ├── input/             # Raw uploaded files for prepare_data
│   ├── store/             # Content-addressed arrays and external table blobs
│   ├── episode/           # Local Git history with logs and traces in each commit
│   ├── cache/             # Evictable compilation and artifact-read reuse
│   └── scratch/           # UI telemetry and run-scoped execution state
└── DEMO/                  # Tracked mock fixture workspace (evals + manual sampling)
```

Back up the whole workspace directory, including `store/`, because the Git history doesn't hold the numerical arrays. Cache entries are safe to delete at any time. `uv run nof1-sweep WORKSPACE_ID` expires telemetry and caches, and offline maintenance can add `--collect-runs` to remove finished run scratch.

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
bun run fixture:build
bun run fixture:check
```

Both commands restore the bundle into an isolated temporary repository and use the production readers to project artifacts, logs, traces, historical snapshots and workbench comparisons in one pass. They do not read the local `history.git` or the generated projections as inputs. DEMO has no numbered artifact directories or separate journal and trace directories.

The bundle preserves the existing illustrative history and numerical findings. Its original posterior samples were not retained, so the fit remains explicitly report-only. Archived predictive checks belong to that attempt at `logs/predictive_checks.json` and are projected into the fixture directory. Regeneration does not fit, simulate, or invent missing scientific artifacts. Prior plot viewports use a small deterministic draw from the retained prior laws.

### Publishing a workspace

Use the current copier only for a stopped synthetic workspace and a destination with no previous copy. The hosted viewer serves the copied payloads:

```bash
uv run --project apps/data-pipeline nof1-publish SYNTHETIC_WORKSPACE --exclude input
```

`--exclude input` withholds uploaded source files only. `--exclude raw_data` matches `store/raw_data/`; it does not remove current raw-table blobs under `store/blobs/` or their Git history. Existing remote files, including mutable Git refs, are skipped, so repeated uploads do not synchronize study history.

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

`GET /api/capabilities` reports `actions_enabled`, which is false on a read-only facade. Creating the run submits `edit_model` with the question and returns HTTP `202` with the workspace and `attempt_id`. Poll that attempt until `done` is true. Submit further actions as the [`nof1-episode-api` skill](../../.agents/skills/nof1-episode-api/SKILL.md) describes; the [action charts](../../README.md#documentation) show what each one does.

### 2. Observe the episode

The episode facade (tool server, port `8100`) is the source of truth:

```bash
# Current state: artifact existence, freshness, revisions, the four action names,
# and `running`, the attempt the episode's Temporal workflow is executing
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID | jq '.artifacts'

# The transition journal: every action attempt (applied / rejected / raised)
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | {seq, status, action, error_type}'

# Transition telemetry (extraction worker fan-out and action labels)
curl -s "http://localhost:8100/api/episodes/$WORKSPACE_ID/events" | jq '.events[-3:]'
```

### 3. Verify via browser automation

- `http://localhost:3000/v2/{WORKSPACE_ID}` is the model workbench, the default destination from the workspace list. `http://localhost:3000/v1/{WORKSPACE_ID}` keeps the older stage-by-stage interface, whose interactivity is not a compatibility requirement.
- After each backend action, check that the workbench shows the question, graph, entity details, data, findings, history and action log. The workbench is read-only: the agent submits every change through the backend API.
- Storybook's **V2 / Model / Workbench / Complete** story covers the workbench with mocked responses. Extend it rather than adding separate stories. Its pinned responses are included in the [fixture build and check](#promoting-a-workspace-to-the-demo-fixture).

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming after a failed action

A failed action is a `raised` transition in the journal. The scientific branch is unchanged, and the typed error and diagnostics are on the record:

```bash
curl -s http://localhost:8100/api/episodes/$WORKSPACE_ID/timeline \
  | jq '.transitions[] | select(.status=="raised") | {seq, action, error_type, error_message, resume}'
```

Correct the cause and resubmit the action to `/actions`, selecting fresh revisions if its inputs changed. There is no automatic-resume endpoint.
