# Agent Quickstart

The service is headless: an external LLM agent navigates the episode machine
over HTTP, and the web viewer renders artifact versions selected by the
append-only transition log. This guide is the agent-side setup.

## Start the service

```bash
bun run integration:start
```

This brings up the Temporal dev server (port `7233`), the episode worker, the
tool server with the episode facade (port `8100`), and the web viewer (port
`3000`). See [agentic_integration_testing.md](agentic_integration_testing.md)
for details.

LLM-backed transitions read the ambient `OPENROUTER_API_KEY` from the service's
environment — credentials are infra config, never per-action parameters.

## Use the scientific actions

The interface is plain HTTP — the same tool server the web viewer uses
(`TOOL_SERVER_URL`, default `http://localhost:8100`). There is no SDK and no MCP
server; an agent navigates entirely with `curl`.

An agent working in this repo gets the `nof1-episode-api` skill automatically —
Claude Code loads it from `.claude/skills/` (a symlink) and Codex from
`.agents/skills/`. Its [`SKILL.md`](../../.agents/skills/nof1-episode-api/SKILL.md)
is the full reference: every endpoint, its body shape, and a copy-ready `curl`
line, generated from the tool server's
[OpenAPI spec](../../packages/api-types/schemas/openapi.json) so it never drifts
from the API. The loop in brief:

1. Read `GET /api/episodes/{workspace_id}` and
   `GET /api/episodes/{workspace_id}/revisions` for state and available inputs.
2. Submit `edit_model`, `prepare_data`, `fit`, or `simulate` at
   `POST /api/episodes/{workspace_id}/actions`. The
   [action contracts](../reference/scientific-actions.md) define revision pins,
   simulation designs and returned findings. Structure, measurements, parameters
   and laws may be interleaved; applicable cheap checks refresh on submission.
3. Dispatch returns HTTP `202` and only `{attempt_id}`. Poll
   `GET /api/episodes/{workspace_id}/actions/{attempt_id}` until `done` is true.
   Each poll returns `body` and all timestamped `messages` so far; the body is
   null while running and on failure. Keep polling the same attempt rather than resubmitting.
4. Read the scientific result from `body` and inspect message levels and labels.
   Warnings can accompany a successful result. Errors leave the scientific branch
   unchanged. The timeline retains completed attempts, results and messages.

Raw data enters by placing files under `data/{workspace_id}/input/` before
submitting `prepare_data` with `source: "files"`. Sources can be imported before
authoring a model. [Causal scenarios](../pipeline/analysis.md) use the same
`simulate` action and retain their results in the journal.

V2 is a read-only inspector. The agent submits all scientific actions through the
backend. The [integration testing guide](agentic_integration_testing.md) has curl
walkthroughs and service requirements.

## Publishing a workspace

The hosted viewer is read-only and serves exactly what was deliberately
published to the hosted store:

```bash
uv run --project apps/data-pipeline nof1-publish WORKSPACE_ID \
  --exclude raw_data --exclude input
```

Publishing is an idempotent file copy (the store is append-only), so
re-running it during a live episode gives the hosted viewer a live tail.
Raw N-of-1 data is personal data — exclude it unless the workspace is
synthetic.
