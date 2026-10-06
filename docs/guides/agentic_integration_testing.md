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
child workflows to `workflow`, even without a complete study journey.

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
git clone --mirror data/DEMO/study/history.bundle data/DEMO/study/history.git
git --git-dir=data/DEMO/study/history.git config nof1.format 20
```

#### Migrating a local study

The migration tools and their tests live in the gitignored `scratchpad/migrations/` directory. Run the commands below from the repository root with those local tools present.

The current runtime requires format 20. Restore only a bundle exported after the offline conversion. Stop the study and close its workflow before converting. The destination must be new and outside the source; the source remains untouched. Convert a format-19 study with the [format-20 converter](../../scratchpad/migrations/migrate_format_20.py):

```bash
uv run --project apps/data-pipeline python -m scratchpad.migrations.migrate_format_20 \
  data/STUDY /tmp/format20/STUDY
```

It wraps retained requests, rebuilds call IDs, and retains the accumulated execution log without scientific execution. Both successful and failed calls are indexed; duplicate historical calls use their first recorded outcome. Unknown original requests remain unknown. If an action's scientific input schema also changed, `--requests` accepts verified current requests keyed by source attempt commit. Retired model-independent preparation calls require that explicit mapping.

Older studies must first reach format 19 using the matching historical contracts and converters. Convert a format-16 study through formats 17 and 18, then format 19:

1. Stop work on the study and close its workflow:

   ```bash
   temporal workflow signal --workflow-id study-STUDY --name close
   ```

2. Run the [format-17 converter](../../scratchpad/migrations/migrate_format_17.py), the [format-18 converter](../../scratchpad/migrations/migrate_format_18.py) and the [format-19 converter](../../scratchpad/migrations/migrate_format_19.py), each into another new destination. The destination must be new and outside the source, and the source is left untouched.

   ```bash
   uv run --project apps/data-pipeline python -m scratchpad.migrations.migrate_format_17 \
     data/STUDY /tmp/format17/STUDY --file-hashes /path/to/verified-file-hashes.json
   uv run --project apps/data-pipeline python -m scratchpad.migrations.migrate_format_18 \
     /tmp/format17/STUDY /tmp/format18/STUDY
   uv run --project apps/data-pipeline python -m scratchpad.migrations.migrate_format_19 \
     /tmp/format18/STUDY /tmp/format19/STUDY
   ```

   The format-19 converter evaluates each applied attempt once, as its action
   publishes, and retains the reports with its record.
   The format-17 converter rebuilds retained call arguments, names each panel originally
   read, removes branch fields and rewrites revision references. Missing upload hashes
   require verified historical SHA-256 values keyed by source call commit; unknown
   historical calls remain unknown; `--requests` accepts verified original arguments
   for calls whose transport was not retained. It performs no scientific execution.
   Convert format 15 with the [format-16 converter](../../scratchpad/migrations/migrate_format_16.py) first.
   Convert a format-14 study with the
   [format-15 converter](../../scratchpad/migrations/migrate_format_15.py)
   first. That conversion requires the original resolved sampler controls via
   `--sampler-settings` when the report did not retain them, and binds overlays
   to their pinned evaluation schedules. Earlier studies first use the
   [format-14](../../scratchpad/migrations/migrate_format_14.py),
   [format-13](../../scratchpad/migrations/migrate_format_13.py), or
   [format-12](../../scratchpad/migrations/migrate_format_12.py)
   converter for their respective source format.

3. Review the migrated snapshots and ref mapping before a live cutover. Then, while offline, back up each whole original under `.local/format-backup-<date>/STUDY`, including `store/`, and replace `data/STUDY` with the migrated repository, keeping one study per ID. Keep backups outside `data/` in durable storage; temporary directories are only converter destinations.

4. Restart the workers with the new code and start a fresh `study-STUDY` workflow from the migrated Git state; don't replay the previous workflow. Export any fixture bundle from the migrated repository, then run `bun run fixture:build`.

Formats before 11 have no route to the current runtime.

#### Squashing a local study's action history

Stop work on the study and close its `study-STUDY` workflow, as in the [migration procedure](#migrating-a-local-study). The [history squash script](../../scratchpad/migrations/squash_study_history.py) compacts saved effects through an applied commit `R` into a new directory outside the source. Keep the directory's basename, which is the logical workspace ID:

```bash
uv run --project apps/data-pipeline python -m scratchpad.migrations.squash_study_history \
  data/STUDY /tmp/squashed/STUDY --at R --dry-run
uv run --project apps/data-pipeline python -m scratchpad.migrations.squash_study_history \
  data/STUDY /tmp/squashed/STUDY --at R
```

Replace `R` with the applied commit OID. The dry run lists retained and dropped attempts without copying. The script keeps the root, `R`, the entire suffix, and the dependency closure of the prefix's last artifact writers (including retractions) and latest fresh simulation. It preserves saved artifact and log objects, all artifact refs, numerical files and authorship links; `squash-mapping.json` maps original commits to new commits or `null` for dropped actions. Sequence numbers and attempt IDs retain their gaps.

At `R` and later commits, artifacts (including absence), supporting evidence and saved action reports are preserved. Readers load retained reports without evaluating checks; missing historical reports remain absent. This is the smallest closure of the mandatory writers, not a globally minimal history: redundant retractions can retain extra actions. There is no optimizer, numerical execution or post-squash equality gate.

Only current-format histories with one main ref are supported. The script refuses recorded data comparisons, simulation-replicate panels anywhere in the preserved catalog, other Git heads, and retained scientific inputs without a recorded producer. Review the new copy, select it offline, then start a fresh workflow and regenerate any fixture bundle using the migration procedure above. The source is unchanged.

### Local stack

```bash
bun run integration:start
```

Starts the stack under [process-compose](https://github.com/F1bonacc1/process-compose)
supervision (`brew install f1bonacc1/tap/process-compose`; config:
[`process-compose.yaml`](../../process-compose.yaml)): the Temporal dev
server on port `7233` (ephemeral state, binary auto-downloaded on first
use), the study worker (task queue `nof1-studies`), the tool server
with the study facade on port `8100`, and the web app on port `3000`.
Startup order is health-gated (`depends_on` + readiness probes) and
crashed processes restart automatically. The script **stays in the
foreground** — wait until `curl -s http://localhost:8100/api/workspaces`
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
The [Modal compute adapter](../../apps/data-pipeline/src/nof1_causal_lab/actions/modal_fit.py)
uses the installed Modal credentials, creates an ephemeral
`nof1-causal-lab-pipeline` app with the current source, and reuses
`nof1-cached-fit-cache:/jax`. The study remains local; the
[fit chart](../assets/action-flows/fit.svg) owns its failure and publication flow.

The Temporal dev server
persists its event history to `.local/agentic-integration-stack/temporal.db`
(the `--db-filename` on its command), so restarting the `temporal`
process — to serve the UI, pick up a change, or recover from a crash —
**resumes** in-flight study workflows exactly where they left off
rather than orphaning them. To start genuinely fresh, delete that
`temporal.db` before boot and wipe that workspace's `store/`, `study/`,
`scratch/`, and `cache/` directories. The Web UI is served at
`http://localhost:8233`.

## Workspace Layout

```text
data/
├── <WORKSPACE_ID>/        # User-facing workspace
│   ├── input/             # Ready-to-use CSV or Parquet tables for prepare_data
│   ├── store/             # Content-addressed arrays and external table blobs
│   ├── study/             # Local Git history with logs and traces in each commit
│   ├── cache/             # Evictable compilation and artifact-read reuse
│   └── scratch/           # Live progress events and run-scoped execution state
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
for Storybook. It replaces `data/DEMO` as a unit rather than merging,
and excludes `cache/` and `scratch/`. It exports all Git refs and objects to
`study/history.bundle` so the tracked fixture retains the main history, attempts and
artifact trees while its local bare repository remains gitignored. The files in
`store/` retain the external numerical payloads.

The tracked `data/DEMO/study/history.bundle` and `data/DEMO/store/` are the fixture's authoritative inputs. Files under `data/DEMO/fixture/` are generated projections for Storybook. General behavior tests use small, test-owned fixtures independent of DEMO. Regenerate Storybook fixtures with:

```bash
bun run fixture:build
```

The command restores the bundle into an isolated temporary repository and uses the production readers to project artifacts, logs, traces, historical snapshots and workbench comparisons in one pass. It does not read the local `history.git` or the generated projections as inputs. DEMO has no numbered artifact directories or separate journal and trace directories.

The bundle preserves the illustrative action history and authored and extracted facts. DEMO retained no posterior samples, so its fit has no numerical evidence or posterior summaries. Reports and checks load the findings retained with [action outcomes](../../apps/data-pipeline/src/nof1_causal_lab/actions/effects.py); missing historical reports remain absent. Regeneration does not fit, simulate, or invent missing scientific artifacts. Prior plot viewports use a small deterministic draw from the retained prior laws.

### Publishing a workspace

Use the current copier only for a stopped synthetic workspace and a destination with no previous copy. The hosted viewer serves the copied payloads:

```bash
uv run --project apps/data-pipeline nof1-publish SYNTHETIC_WORKSPACE --exclude input
```

`--exclude input` withholds uploaded source files only. `--exclude raw_data` matches `store/raw_data/`; it does not remove current raw-table blobs under `store/blobs/` or their Git history. Existing remote files, including mutable Git refs, are skipped, so repeated uploads do not synchronize study history.

## Step-by-Step Flow

### 1. Create the workspace and start the study

```bash
WORKSPACE_ID="T3ST42"
QUESTION="How does screen time affect sleep?"

curl -s -X POST http://localhost:8100/api/upload \
  -F "workspaceId=$WORKSPACE_ID" \
  -F "file=@data/DEMO/input/observations.csv"

curl -s -X POST http://localhost:8100/api/studies/$WORKSPACE_ID/edit_question \
  -H 'Content-Type: application/json' \
  -d "{\"action\":\"edit_question\",\"input\":{\"question\":{\"text\":\"$QUESTION\"}}}"
```

`GET /api/workspaces` includes the `X-Actions-Enabled` capability header. Poll the returned `call_id` using the [action API contract](../../.agents/skills/nof1-study-api/SKILL.md). A read-only facade serves saved calls, including failures and comparisons; unsaved POST calls return 403. The [action charts](../../README.md#documentation) show what each action does.

### 2. Observe the study

The study facade (tool server, port `8100`) is the source of truth:

```bash
# Slim journal: every recorded call's arguments, status, messages and dependencies
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline \
  | jq '.attempts[] | {commit_id, seq: .record.seq, attempt: .record.attempt}'

# Read the running call without submitting work
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline > /tmp/study-timeline.json
ACTION=$(jq -r '.running.action' /tmp/study-timeline.json)
CALL_ID=$(jq -r '.running.call_id' /tmp/study-timeline.json)
curl -s "http://localhost:8100/api/studies/$WORKSPACE_ID/$ACTION/$CALL_ID" \
  | jq '{call_id, action, status, commit_id, messages}'

# The viewer uses the same GET route for saved results and execution logs.
```

### 3. Verify via browser automation

- `http://localhost:3000/v2/{WORKSPACE_ID}` is the model workbench, the default destination from the workspace list.
- After each backend action, check that the workbench shows the question, graph, entity details, data, findings, history and action log. The workbench is read-only: the agent submits every change through the backend API.
- Storybook's **V2 / Model / Workbench / Complete** story covers the workbench with mocked responses. Extend it rather than adding separate stories. Its pinned responses are included in the [fixture build](#promoting-a-workspace-to-the-demo-fixture).

If the UI behaves unexpectedly, check Next.js devtools MCP errors before debugging the browser script.

## Resuming after a failed action

A failed execution is a `raised` outcome in the journal. Scientific state is unchanged, and the outcome retains its error. Expected input rejections carry a `rejected` outcome:

```bash
curl -s http://localhost:8100/api/studies/$WORKSPACE_ID/timeline \
  | jq '.attempts[] | select(.record.attempt.outcome.status=="raised") | {seq: .record.seq, attempt: .record.attempt}'
```

Correct the scientific inputs and submit the changed call through its named action route. An identical resolved call returns its saved failure; changing only `reasoning` does not create another call. See the [action API contract](../../.agents/skills/nof1-study-api/SKILL.md).
